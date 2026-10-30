#!/usr/bin/env python3
"""SFT 验收闸门：不达标不进 RL（PROPOSAL §6）。

四道闸门 + 一条格式闸门，阈值全部来自 ``configs/default.yaml`` 的 ``sft.gate``：

    format_valid_rate  ≥ 0.99       <think>/<code>/<answer> 结构完整率
    tool_success_rate  ≥ 0.95       沙箱执行成功率（按每次调用计）
    tool_call_rate     ≥ 0.55       至少调用一次工具的样本比例（下限）
    over_call_rate     ≤ 0.25       不需要工具却调用的比例（上限）
    answer_change_rate ≥ 0.60       同一题 "给 gold 框" vs "不给工具" 答案不同的比例

输入是一个评测 jsonl，每行至少包含::

    {"id": ..., "dataset": ..., "condition": "gold_box" | "no_tool",
     "correct": bool, "n_calls": int, "tool_oks": [bool], ...
     "format_ok": bool             # 可选，缺省按 answer 非空估算
     "needs_tool": bool}           # 可选，缺省 True（无从判断时不计入 over_call）

``answer_change_rate`` 需要同 id 的两条记录（gold_box / no_tool）配对；只有单条时
该项记为 nan 并且**不算通过**——宁可拦下，也不要拿缺失指标放行。

用法::

    python scripts/check_gate.py --records data/eval/sft_gate.jsonl
    python scripts/check_gate.py --records r.jsonl --baseline base.jsonl --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medtaco.data.training_set import OBS_HEADER, OBS_TAIL
from medtaco.eval import metrics
from medtaco.rl.grader import normalize as _normalize_answer
from medtaco.utils.config import load_config
from medtaco.utils.io import atomic_write_json, iter_jsonl

#: 从 assistant 文本里抽 <answer>...</answer>（非贪婪，容忍前后多余的 think 文本）
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)

#: 工具**失败**的可靠判据（小写比较）。
#:
#: 依据（不改 ``training_set.py`` 的输出格式，只为解析侧提供共享判定）：
#:   - ``工具执行失败：{error}`` —— ``training_set.py`` 注入失败观测的**唯一**入口
#:     （render_trajectory，``if st.result and not st.result.ok`` 分支）。成功观测
#:     里绝不会出现这五个字，所以它是最可信的正向标记。
#:   - ``执行超时`` —— ``tools/sandbox.py`` 的 ok=False error 文本。
#:   - ``退化框`` —— ``tools/ops.py`` 的 degenerate_box 分支。
#:   - ``需要参数`` —— ``tools/ops.py`` 参数缺失分支（如 "measure 需要参数 box"）。
#:   - ``未知工具`` —— ``tools/ops.py`` 的工具名分发失败。
#:   - ``toolunavailable`` —— ops.execute 的 ToolUnavailable 异常类别。
#:
#: 旧实现只认 ``错误``/``error``，而上面这些串**一个都不含**这两个词，于是失败调用
#: 被全量误判为 ok=True，tool_success_rate 虚高。这里放宽到上述集合，并保留
#: 旧关键词以兼容外部产物（历史评测 jsonl 可能在 error 体里写字面量 "错误"）。
TOOL_FAILURE_MARKERS: Tuple[str, ...] = (
    "工具执行失败",
    "执行超时",
    "退化框",
    "需要参数",
    "未知工具",
    "toolunavailable",
    "错误",   # 旧关键词，向后兼容
    "error",  # 旧关键词，向后兼容
)


def obs_is_error(text: str) -> bool:
    """判断一条工具观测文本是否代表**失败**调用。

    只剥掉观测头尾（``【工具返回】`` / ``【工具返回结束】``）再匹配，不做其它改写：
    成功观测的主体是工具返回文本（如 ``12.34 mm``）或图片占位符，不会命中
    ``TOOL_FAILURE_MARKERS``。
    """
    body = str(text or "").replace(OBS_HEADER, " ").replace(OBS_TAIL, " ")
    low = body.lower()
    return any(m in low for m in TOOL_FAILURE_MARKERS)

# 与 configs/default.yaml: sft.gate 同名；config 缺字段时用这里的默认值
DEFAULT_GATE: Dict[str, float] = {    "min_format_valid_rate": 0.99,
    "min_tool_success_rate": 0.95,
    "min_tool_call_rate": 0.55,
    "max_over_call_rate": 0.25,
    "min_answer_change_rate": 0.60,
}


def _from_messages(rec: Dict[str, Any]) -> Dict[str, Any]:
    """从训练/推理产物（``messages``）里回推评测所需的扁平字段。

    这层适配不是多余的：同一批数据导出成 messages 是为了训练，评测脚本需要的是
    answer/n_calls/tool_oks。缺了这一步，训练数据直接喂给闸门会全读成 0/缺失，
    看起来像"模型没学会用工具"，实际只是读错了字段。

    能推出来的：
      - ``answer``：最后一条 assistant 文本里 ``<answer>..</answer>`` 的内容
        （没有标签就取整段）
      - ``n_calls``：带 ``<tool_call>`` 的 assistant 轮数
      - ``tool_oks``：每条工具返回（role=user + ``【工具返回】``）是否不含错误标记
    """
    msgs = rec.get("messages")
    if not isinstance(msgs, list):
        return {}
    answer = ""
    n_calls = 0
    oks: List[bool] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if isinstance(content, list):
            text = "".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
        else:
            text = str(content or "")
        role = m.get("role")
        if role == "assistant":
            if "<tool_call>" in text:
                n_calls += 1
            ans = _ANSWER_RE.search(text)
            answer = ans.group(1).strip() if ans else text.strip()
        elif role == "user" and "【工具返回】" in text:
            oks.append(not obs_is_error(text))
    out: Dict[str, Any] = {"answer": answer, "n_calls": n_calls}
    if oks:
        out["tool_oks"] = oks
    return out


def _flat(rec: Dict[str, Any]) -> Dict[str, Any]:
    """把 ``meta`` 里的字段摊到顶层（顶层优先）。

    训练/评测产物有两种形态：导出阶段的记录是 ``{messages, images, meta}``，
    评测阶段是扁平字段。不统一的话，同一份数据换个阶段产出，闸门就会从 0.95 掉成
    "指标缺失"，看起来像模型退化了。
    """
    if not isinstance(rec, dict):
        return {}
    merged = _from_messages(rec)
    meta = rec.get("meta")
    if isinstance(meta, dict):
        merged.update(meta)
    merged.update({k: v for k, v in rec.items() if k != "meta"})
    return merged


def _answer(rec: Dict[str, Any]) -> str:
    r = _flat(rec)
    for k in ("answer", "final_answer", "pred", "response"):
        if r.get(k) is not None:
            return str(r[k])
    return ""


def _cond(rec: Dict[str, Any]) -> str:
    r = _flat(rec)
    return str(r.get("condition") or r.get("view") or "").strip().lower()


def _strict_answer_ok(rec: Dict[str, Any]) -> Optional[bool]:
    """用严格 ``<answer>...</answer>`` 解析判断格式是否合格；拿不到 raw 时返回 None。

    输入记录若带原始模型输出 ``raw``（评测产物常见字段），就不该退回到"答案非空"
    这种粗估：答案非空只能证明模型说了话，证明不了它遵守了输出协议。
    """
    r = _flat(rec)
    raw = r.get("raw")
    if not isinstance(raw, str) or not raw.strip():
        return None
    m = _ANSWER_RE.search(raw)
    return bool(m and m.group(1).strip())


def format_valid_report(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """格式合格率及其来源。

    判定优先级：``format_ok`` 字段 > 原始 ``raw`` 的严格 ``<answer>`` 解析 > 答案非空。

    返回 ``{rate, n, n_explicit, n_strict_parsed, n_inferred, degraded, sources}``。
    ``degraded=True`` 表示至少有一条记录是靠"答案非空"推断出来的——这类记录绕过了
    真正的格式校验，报告必须暴露它，否则一个没写 format_ok 的评测文件会被读成满分。
    **闸门仍按推断值判定**：直接把缺失当 FAIL 会让所有没有该字段的合法产物全废。
    """
    n_explicit = 0
    n_strict = 0
    n_inferred = 0
    n_ok = 0
    for r in records:
        if isinstance(r, dict) and r.get("format_ok") is not None:
            n_explicit += 1
            ok = bool(r["format_ok"])
        else:
            strict = _strict_answer_ok(r)
            if strict is None:
                n_inferred += 1
                ok = bool(_answer(r).strip())
            else:
                n_strict += 1
                ok = strict
        n_ok += bool(ok)
    n = len(records)
    return {
        "rate": (n_ok / n) if n else float("nan"),
        "n": n,
        "n_explicit": n_explicit,
        "n_strict_parsed": n_strict,
        "n_inferred": n_inferred,
        "degraded": n_inferred > 0,
        "sources": {
            "explicit_format_ok": n_explicit,
            "strict_answer_parse": n_strict,
            "inferred_from_nonempty": n_inferred,
        },
    }


def format_valid_rate(records: Sequence[Dict[str, Any]]) -> float:
    """格式合格率（数值版，供外部单独调用；细节见 ``format_valid_report``）。

    优先用记录自带的 format_ok；没有的话先尝试严格解析原始 ``raw``，
    最后才退化为"答案非空"（推断来源会在报告里标 degraded）。
    """
    return float(format_valid_report(records)["rate"])


def answer_change_rate(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """同一 id 上 gold_box 与 no_tool 两路的答案差异率。

    比较口径两套，都报出来：

      - ``n_changed`` / ``rate``：过 ``medtaco.rl.grader.normalize`` 后再比。
        大小写、全角半角、标点、冠词、单复数、单位（``1.5 cm`` == ``15 mm``）的差异
        不算变化。这是闸门真正该用的口径——用原始字符串比，模型只要换个说法
        就被记成"工具改变了答案"，指标系统性偏乐观。
      - ``raw_changed`` / ``raw_rate``：原实现的裸字符串比较（strip + casefold），
        保留下来供对照，不作为判定依据。

    返回 ``{n_paired, n_changed, rate, raw_changed, raw_rate}``；没有可配对样本时
    rate=nan。
    """
    by_id: Dict[str, Dict[str, str]] = {}
    for r in records:
        rid = r.get("id")
        c = _cond(r)
        if rid is None or c not in ("gold_box", "no_tool"):
            continue
        by_id.setdefault(str(rid), {})[c] = _answer(r)

    n_paired = 0
    n_changed = 0
    n_raw_changed = 0
    for pair in by_id.values():
        if "gold_box" not in pair or "no_tool" not in pair:
            continue
        n_paired += 1
        a = pair["gold_box"].strip()
        b = pair["no_tool"].strip()
        if a.casefold() != b.casefold():
            n_raw_changed += 1
        if _normalize_answer(a) != _normalize_answer(b):
            n_changed += 1
    return {
        "n_paired": n_paired,
        "n_changed": n_changed,
        "rate": (n_changed / n_paired) if n_paired else float("nan"),
        "raw_changed": n_raw_changed,
        "raw_rate": (n_raw_changed / n_paired) if n_paired else float("nan"),
    }


def tool_call_rate(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """至少调用一次工具的记录比例，**分母排除 no_tool 条件**。

    混合文件（同一份 jsonl 里既有 gold_box 也有 no_tool）上，no_tool 行的正确行为
    就是 0 次调用；把它们算进分母会把 tool_call_rate 机械地拉低（两路各占一半时
    直接砍掉一半），读起来像"模型不会用工具"，其实只是口径错了。

    口径：
      - 有 ``condition`` 字段时，分母 = ``condition != "no_tool"`` 的行（即允许/应当
        用工具的那些行）；
      - 全文件都没有 condition 字段时按**全部行**算，保持对旧产物的向后兼容。

    返回 ``{rate, n, denominator, n_no_tool_excluded, n_conditioned}``。
    """
    flat = [_flat(r) for r in records]
    has_condition = any(f.get("condition") or f.get("view") for f in flat)
    if has_condition:
        judged = [f for f in flat if str(f.get("condition") or f.get("view")
                                         or "").strip().lower() != "no_tool"]
        denom = "condition!=no_tool"
    else:
        judged = flat
        denom = "all"
    n = len(judged)
    n_called = sum(1 for f in judged if metrics._record_calls(f) > 0)
    return {
        "rate": (n_called / n) if n else float("nan"),
        "n": n,
        "n_called": n_called,
        "denominator": denom,
        "n_no_tool_excluded": len(flat) - n if has_condition else 0,
        "n_conditioned": sum(1 for f in flat if f.get("condition") or f.get("view")),
    }


def over_call_rate(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """不需要工具的题里，仍然发起调用的比例。needs_tool 缺失的记录不计入分母。"""
    flat = [_flat(r) for r in records]
    judged = [r for r in flat if r.get("needs_tool") is not None]
    neg = [r for r in judged if not r["needs_tool"]]
    if not neg:
        return {"n": 0, "n_called": 0, "rate": float("nan"),
                "denominator": "needs_tool==False"}
    n_called = sum(1 for r in neg if metrics._record_calls(r) > 0)
    return {"n": len(neg), "n_called": n_called, "rate": n_called / len(neg),
            "denominator": "needs_tool==False", "n_without_needs_tool": len(flat) - len(judged)}


def _check(name: str, value: float, limit: float, lower_is_bad: bool) -> Dict[str, Any]:
    if value is None or value != value:  # nan
        return {"gate": name, "value": None, "limit": limit, "passed": False,
                "note": "指标缺失，按不通过处理"}
    ok = (value >= limit) if lower_is_bad else (value <= limit)
    return {"gate": name, "value": float(value), "limit": limit, "passed": bool(ok), "note": ""}


def evaluate_gate(records: Sequence[Dict[str, Any]], gate: Dict[str, float],
                  baseline: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """跑全部门禁，返回可直接打印/落盘的报告。"""
    g = dict(DEFAULT_GATE)
    g.update({k: float(v) for k, v in (gate or {}).items() if k in DEFAULT_GATE})

    flat = [_flat(r) for r in records]
    eff = metrics.efficiency_report(flat)
    chg = answer_change_rate(records)
    oc = over_call_rate(records)
    fmt_rep = format_valid_report(records)
    fmt = fmt_rep["rate"]
    cr = tool_call_rate(records)

    checks: List[Dict[str, Any]] = [
        _check("format_valid_rate", fmt, g["min_format_valid_rate"], True),
        _check("tool_success_rate", eff["tool_success_rate"], g["min_tool_success_rate"], True),
        # 调用率用排除 no_tool 的分母；gap 字段记录"不排除时"会是多少，便于对照
        _check("tool_call_rate", cr["rate"], g["min_tool_call_rate"], True),
        _check("over_call_rate", oc["rate"], g["max_over_call_rate"], False),
        _check("answer_change_rate", chg["rate"], g["min_answer_change_rate"], True),
    ]

    # 推断出来的 format_ok 必须显式曝光：闸门照判，但报告要标 degraded，
    # 否则"没写 format_ok 的文件"会被读成格式满分通过。
    if fmt_rep["degraded"]:
        for c in checks:
            if c["gate"] == "format_valid_rate":
                c["degraded"] = True
                c["note"] = (f"format_ok 缺失 {fmt_rep['n_inferred']}/{fmt_rep['n']} 条，"
                             f"已按 format_ok_source=inferred_from_nonempty 推断（非真实格式校验）")

    # 指标缺失时要说清是"数据不行"还是"输入不对"：训练导出里没有 needs_tool 和
    # condition，这两道闸门只能在**评测**产物上算。不区分的话，读训练文件得到的
    # FAIL 会被误读成模型问题。
    has_needs_tool = any(f.get("needs_tool") is not None for f in flat)
    has_condition = any(f.get("condition") for f in flat)
    for c in checks:
        if c["gate"] == "tool_call_rate":
            c["denominator"] = cr["denominator"]
            c["n"] = cr["n"]
        if "degraded" in c:
            continue  # 已有更具体的说明（如 format_valid_rate 的推断提示）
        if c["value"] is not None:
            continue
        if c["gate"] == "over_call_rate" and not has_needs_tool:
            c["note"] = "输入无 needs_tool 字段：该闸门需评测产物（每条标明是否需要工具）"
        elif c["gate"] == "answer_change_rate" and not has_condition:
            c["note"] = "输入无 condition 配对：需同一题 gold_box 与 no_tool 两条记录的评测产物"

    report: Dict[str, Any] = {
        "n": len(records),
        "checks": checks,
        "passed": all(c["passed"] for c in checks),
        "degraded": bool(fmt_rep["degraded"]),
        "details": {
            "format_valid_rate": fmt,
            "format_valid": fmt_rep,
            "format_ok_source": ("inferred_from_nonempty" if fmt_rep["degraded"] and not fmt_rep["n_explicit"]
                                 else ("explicit" if not fmt_rep["degraded"] else "mixed")),
            "tool_success_rate": eff["tool_success_rate"],
            "tool_call_rate": cr["rate"],
            "tool_call": cr,
            "avg_calls": eff["avg_calls"],
            "avg_tokens": eff["avg_tokens"],
            "over_call": oc,
            "answer_change": chg,
            "per_source": metrics.per_source_report(records),
        },
    }

    # 与 base 的配对比较：只比两路都有的题，避免拿不同题集自欺
    if baseline:
        a = {str(r["id"]): bool(r.get("correct")) for r in baseline
             if r.get("id") is not None and r.get("correct") is not None}
        b = {str(r["id"]): bool(r.get("correct")) for r in records
             if r.get("id") is not None and r.get("correct") is not None}
        ids = [i for i in b if i in a]
        if ids:
            report["details"]["vs_baseline"] = metrics.paired_report(
                [b[i] for i in ids], [a[i] for i in ids])
            report["details"]["vs_baseline"]["n_common"] = len(ids)
        else:
            report["details"]["vs_baseline"] = {"n_common": 0,
                                                "note": "无同 id 记录，无法配对比较"}

    return report


def _fmt(v: Any) -> str:
    if v is None:
        return "  n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def print_report(r: Dict[str, Any]) -> None:
    print(f"\n{'闸门':<24}{'实测':>10}{'阈值':>10}   结果")
    print("-" * 58)
    for c in r["checks"]:
        op = "<=" if c["gate"] == "over_call_rate" else ">="
        print(f"{c['gate']:<24}{_fmt(c['value']):>10}{c['limit']:>10.4f}   "
              f"{'PASS' if c['passed'] else 'FAIL'}  {op}")
        if c.get("note"):
            print(f"    ! {c['note']}")

    d = r["details"]
    print(f"\n样本 {r['n']}；平均调用 {_fmt(d['avg_calls'])} 次；平均 token {_fmt(d['avg_tokens'])}")
    chg = d["answer_change"]
    if chg["n_paired"]:
        print(f"gold_box vs no_tool 答案变化(归一化): {chg['n_changed']}/{chg['n_paired']}"
              f"；原始字符串比较: {chg['raw_changed']}/{chg['n_paired']}")
    else:
        print("gold_box vs no_tool 未配对（该项按 FAIL 处理）")

    tc = d.get("tool_call")
    if tc:
        print(f"tool_call_rate 分母口径: {tc['denominator']}（n={tc['n']}，"
              f"排除 no_tool 行 {d['tool_call'].get('n_no_tool_excluded', 0)} 条）")
    fv = d.get("format_valid")
    if fv and fv.get("degraded"):
        print(f"! format_valid_rate 含推断值 {fv['n_inferred']}/{fv['n']} 条"
              f"（format_ok 缺失，按答案非空推断，非真实格式校验）")

    vb = d.get("vs_baseline")
    if vb and vb.get("n_common"):
        print(f"vs base (n={vb['n_common']}): acc {vb['acc_a']:.4f} vs {vb['acc_b']:.4f}，"
              f"diff {vb['diff']:+.4f}，McNemar p={vb['mcnemar']['p']:.4g}")
    elif vb:
        print(f"vs base: {vb.get('note', '')}")

    print("\n结论:", "通过，可以进 RL" if r["passed"] else "未通过，先修数据/训练再进 RL")


def main() -> int:
    ap = argparse.ArgumentParser(description="SFT 验收闸门（PROPOSAL §6）")
    ap.add_argument("--records", required=True, help="SFT 模型在闸门集上的评测 jsonl")
    ap.add_argument("--baseline", default=None, help="base 模型同一评测集的 jsonl（可选）")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="K=V",
                    help="覆盖阈值，如 --set min_tool_call_rate=0.6")
    ap.add_argument("--json", default=None, help="把完整报告写到该路径")
    args = ap.parse_args()

    if not os.path.exists(args.records):
        print(f"记录文件不存在: {args.records}", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    gate = dict(cfg.get("sft.gate", {}) or {})
    for kv in args.set:
        k, _, v = kv.partition("=")
        gate[k.strip()] = float(v)

    records = list(iter_jsonl(args.records))
    baseline = list(iter_jsonl(args.baseline)) if args.baseline and os.path.exists(args.baseline) else None

    report = evaluate_gate(records, gate, baseline)
    print_report(report)
    if args.json:
        atomic_write_json(args.json, report)
        print(f"\n报告已写入 {args.json}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
