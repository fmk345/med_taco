#!/usr/bin/env python3
"""评测聚合：准确率 + 分源 + 效率 + VEG + oracle 上界 + 配对检验（PROPOSAL §8）。

本脚本**只做后处理**，不做推理：它读取推理产出的 jsonl（每个方法一个文件），
统一汇总成一张报告。这样评测可以脱离训练机跑，也便于反复重算指标而不用重跑模型。

记录约定（与 ``medtaco/eval/metrics.py`` 的 docstring 一致，字段全部可选）::

    {"id": "SLAKE-EN/42", "dataset": "SLAKE-EN", "source": null,
     "condition": "full" | "oracle_crop" | "gold_box" | "no_tool",
     "correct": true,                       # 正常条件是否答对
     "correct_corrupted": {"blackout": false, "shuffle_pixels": true, ...},
     "image": "/path/crop.jpg",             # VEG 重算时才需要
     "n_calls": 2, "tool_oks": [true, true],
     "tokens": 1834, "box": [...], "pred_box": [...], "needs_tool": true}

VEG 有两种来源，**优先用记录里已经算好的 correct_corrupted**（省掉重跑模型）：

  1. 预计算：记录带 ``correct_corrupted``，直接从它算 Acc_corrupted 与 VEG；
  2. 重算：只带 ``image`` 时走 ``metrics.run_veg``，需要传 ``--evaluate-cmd`` 才能重跑
     模型：其它情况直接报"无法计算 VEG"，**不拿 0 冒充**。

用法::

    python scripts/eval_all.py --records data/eval/sft.jsonl --name sft
    python scripts/eval_all.py --records sft.jsonl --baseline base.jsonl --markdown report.md
    python scripts/eval_all.py --records a.jsonl --records b.jsonl --name a --name b
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medtaco.eval import metrics
from medtaco.utils import stats
from medtaco.utils.config import load_config
from medtaco.utils.io import atomic_write_json, iter_jsonl

# VEG 接近 0 就说明"工具是装饰性的"（DeepEyes 83.3→83.8 的反例）
VEG_DECORATIVE_BAND = 0.02


# ---------------------------------------------------------------- 单方法指标


def veg_from_records(records: Sequence[Dict[str, Any]],
                     modes: Sequence[str], n_boot: int = 2000,
                     seed: int = 42) -> Dict[str, Any]:
    """从预计算的 ``correct_corrupted`` 算 VEG；与 metrics.run_veg 口径一致（严格配对）。

    配对规则：只有同时具备 ``correct`` 与某个 mode 的破坏结果的记录才计入该 mode，
    且各 mode 的分母各自独立——正常一路的 acc 也按同一子集重算，避免"全量 acc
    减子集 acc"这种不同底数的错配。
    """
    per_mode: Dict[str, Dict[str, Any]] = {}
    normal_all = [1.0 if r.get("correct") else 0.0 for r in records
                  if r.get("correct") is not None]

    for mode in modes:
        n_flags: List[float] = []
        c_flags: List[float] = []
        for r in records:
            cc = r.get("correct_corrupted")
            if not isinstance(cc, dict) or r.get("correct") is None or mode not in cc:
                continue
            if cc[mode] is None:
                continue
            n_flags.append(1.0 if r["correct"] else 0.0)
            c_flags.append(1.0 if cc[mode] else 0.0)
        if not n_flags:
            per_mode[mode] = {"n": 0, "acc_normal": float("nan"),
                              "acc_corrupted": float("nan"), "veg": float("nan")}
            continue
        acc_n = sum(n_flags) / len(n_flags)
        acc_c = sum(c_flags) / len(c_flags)
        # VEG 的方向是"正常 − 破坏"，所以 CI 也按这个方向算（a=正常，b=破坏）
        _, lo, hi = stats.paired_bootstrap(n_flags, c_flags, n_boot=n_boot, seed=seed)
        per_mode[mode] = {
            "n": len(n_flags),
            "acc_normal": acc_n,
            "acc_corrupted": acc_c,
            "veg": metrics.veg(acc_n, acc_c),
            "veg_ci": (lo, hi),  # 破坏一路 − 正常一路 的配对差值 CI
        }

    valid = [v["veg"] for v in per_mode.values() if v["n"] and not _isnan(v["veg"])]
    return {
        "n_records": len(records),
        "n_normal": len(normal_all),
        "acc_normal": (sum(normal_all) / len(normal_all)) if normal_all else float("nan"),
        "per_mode": per_mode,
        "veg_mean": (sum(valid) / len(valid)) if valid else float("nan"),
        "decorative": bool(valid) and abs(sum(valid) / len(valid)) <= VEG_DECORATIVE_BAND,
    }


def evaluate_one(records: Sequence[Dict[str, Any]], cfg: Any, name: str) -> Dict[str, Any]:
    """一个 ckpt 的全套指标。"""
    modes = list(cfg.get("eval.veg.corrupt_modes", metrics.CORRUPT_MODES))
    n_boot = int(cfg.get("eval.bootstrap_n", 2000))
    seed = int(cfg.get("seed", 42))

    out: Dict[str, Any] = {
        "name": name,
        "n": len(records),
        "acc": stats.summarize_acc([bool(r["correct"]) for r in records
                                    if r.get("correct") is not None], n_boot=n_boot),
        "per_source": metrics.per_source_report(records, n_boot=n_boot),
        "efficiency": metrics.efficiency_report(records),
        "veg": veg_from_records(records, modes, n_boot=n_boot, seed=seed),
        "oracle_ceiling": metrics.oracle_crop_ceiling(records, n_boot=n_boot, seed=seed),
    }
    boxes = [(r["pred_box"], r["box"]) for r in records
             if r.get("pred_box") and r.get("box")]
    if boxes:
        from medtaco.utils.schema import Box

        ious = []
        for pred, gold in boxes:
            try:
                ious.append(metrics.iou(Box(*[float(x) for x in pred]),
                                        Box(*[float(x) for x in gold])))
            except (TypeError, ValueError):
                continue
        if ious:
            out["iou"] = {"n": len(ious), "mean_iou": sum(ious) / len(ious)}
    return out


def compare(rec_a: Sequence[Dict[str, Any]], rec_b: Sequence[Dict[str, Any]],
            cfg: Any) -> Dict[str, Any]:
    """按 id 配对比较两个 ckpt（a = 新，b = 基线）。"""
    a = {str(r["id"]): bool(r["correct"]) for r in rec_a
         if r.get("id") is not None and r.get("correct") is not None}
    b = {str(r["id"]): bool(r["correct"]) for r in rec_b
         if r.get("id") is not None and r.get("correct") is not None}
    ids = [i for i in a if i in b]
    if not ids:
        return {"n_common": 0, "note": "无同 id 记录，无法配对比较"}
    rep = metrics.paired_report([a[i] for i in ids], [b[i] for i in ids],
                                n_boot=int(cfg.get("eval.bootstrap_n", 2000)),
                                icc=float(cfg.get("eval.icc", 0.0) or 0.0),
                                m=float(cfg.get("eval.m", 1.0) or 1.0),
                                seed=int(cfg.get("seed", 42)))
    rep["n_common"] = len(ids)
    rep["only_in_a"] = len(a) - len(ids)
    rep["only_in_b"] = len(b) - len(ids)
    return rep


# ---------------------------------------------------------------- 输出


def _isnan(v: Any) -> bool:
    try:
        return v is None or (isinstance(v, float) and math.isnan(v))
    except (TypeError, ValueError):
        return False


def _f(v: Any, nd: int = 4) -> str:
    if _isnan(v):
        return "n/a"
    if isinstance(v, (int, float)):
        return f"{v:.{nd}f}"
    return str(v)


def to_markdown(reports: List[Dict[str, Any]], comps: Dict[str, Any]) -> str:
    lines: List[str] = ["# MedTACO 评测报告", ""]

    lines += ["## 总览", "", "| ckpt | n | acc | 95% Wilson | VEG(均值) | 调用率 | 工具成功率 |",
              "|---|---|---|---|---|---|---|"]
    for r in reports:
        acc = r["acc"]
        w = f"[{_f(acc['wilson_lo'],3)}, {_f(acc['wilson_hi'],3)}]"
        lines.append(
            f"| {r['name']} | {r['n']} | {_f(acc['acc'])} | {w} | "
            f"{_f(r['veg']['veg_mean'])} | {_f(r['efficiency']['call_rate'],3)} | "
            f"{_f(r['efficiency']['tool_success_rate'],3)} |"
        )

    for r in reports:
        lines += ["", f"## {r['name']}", ""]

        v = r["veg"]
        if v["decorative"]:
            lines.append(f"> ⚠️ VEG 均值 {_f(v['veg_mean'])} 落在 ±{VEG_DECORATIVE_BAND} 内："
                         "破坏返回图后准确率几乎不变，**工具没有真正被用上**。")
        lines += ["### 忠实性（VEG，正数=破坏后掉分）", "",
                  "| mode | n | Acc 正常 | Acc 破坏 | VEG | 配对 CI |", "|---|---|---|---|---|---|"]
        for mode, m in v["per_mode"].items():
            ci = m.get("veg_ci")
            cis = f"[{_f(ci[0],3)}, {_f(ci[1],3)}]" if ci else "n/a"
            lines.append(f"| {mode} | {m['n']} | {_f(m['acc_normal'],3)} | "
                         f"{_f(m['acc_corrupted'],3)} | {_f(m['veg'])} | {cis} |")

        oc = r["oracle_ceiling"]
        lines += ["", "### oracle 上界（给 gold 框能涨多少）", "",
                  f"- full: n={oc['full']['n']} acc={_f(oc['full']['acc'])}",
                  f"- oracle_crop: n={oc['oracle_crop']['n']} acc={_f(oc['oracle_crop']['acc'])}",
                  f"- Δ(oracle−full) = {_f(oc['delta'])}，配对 n={oc['n_paired']}"]

        ps = r["per_source"]
        lines += ["", "### 分源（OmniMedVQA 已拆子源，禁止报混合均值）", "",
                  "| 源 | n | acc | 95% Wilson |", "|---|---|---|---|"]
        for label, s in ps["per_source"].items():
            lines.append(f"| {label} | {s['n']} | {_f(s['acc'])} | "
                         f"[{_f(s['wilson_lo'],3)}, {_f(s['wilson_hi'],3)}] |")
        if ps.get("mixed_excluded"):
            lines += ["", f"已剔除的混合源: {', '.join(ps['mixed_excluded'])}"]
        for w in ps.get("warnings", []):
            lines += [f"> {w}"]

        e = r["efficiency"]
        lines += ["", "### 效率", "",
                  f"- 调用率 {_f(e['call_rate'],3)}；平均调用 {_f(e['avg_calls'],2)} 次；"
                  f"工具成功率 {_f(e['tool_success_rate'],3)}；平均 token {_f(e['avg_tokens'],0)}"]
        if "iou" in r:
            lines.append(f"- 裁剪 IoU：n={r['iou']['n']}，mean IoU={_f(r['iou']['mean_iou'],3)}")

    for key, c in comps.items():
        lines += ["", f"## 配对比较：{key}", ""]
        if not c.get("n_common"):
            lines.append(c.get("note", "无可配对样本"))
            continue
        lines += [f"- n={c['n_common']}（仅 a 有 {c['only_in_a']}，仅 b 有 {c['only_in_b']}）",
                  f"- acc：a={_f(c['acc_a'])} b={_f(c['acc_b'])}，Δ={c['diff']:+.4f}",
                  f"- Δ 的配对 bootstrap CI："
                  f"[{_f(c['diff_boot_lo'])}, {_f(c['diff_boot_hi'])}]",
                  f"- McNemar：b={c['mcnemar']['b']} c={c['mcnemar']['c']} "
                  f"p={c['mcnemar']['p']:.4g}",
                  f"- 有效样本量 {_f(c['effective_n'],1)}（DEFF={_f(c['deff'],3)}）"]
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI


def main() -> int:
    ap = argparse.ArgumentParser(description="评测聚合（PROPOSAL §8）")
    ap.add_argument("--records", action="append", required=True,
                    help="评测 jsonl，可重复传多个")
    ap.add_argument("--name", action="append", default=[],
                    help="与 --records 一一对应的名字，缺省用文件名")
    ap.add_argument("--baseline", default=None, help="基线 ckpt 的 jsonl（用于配对比较）")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--json", default=None, help="完整报告落盘路径")
    ap.add_argument("--markdown", default=None, help="Markdown 报告落盘路径")
    ap.add_argument("--strict", action="store_true",
                    help="VEG 落在装饰性区间时返回非零（CI 里当断言用）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    names = list(args.name) or [os.path.splitext(os.path.basename(p))[0] for p in args.records]
    if len(names) < len(args.records):
        names += [os.path.splitext(os.path.basename(p))[0]
                  for p in args.records[len(names):]]

    all_recs: List[List[Dict[str, Any]]] = []
    reports: List[Dict[str, Any]] = []
    for path, name in zip(args.records, names):
        if not os.path.exists(path):
            print(f"记录文件不存在: {path}", file=sys.stderr)
            return 2
        recs = list(iter_jsonl(path))
        all_recs.append(recs)
        reports.append(evaluate_one(recs, cfg, name))
        print(f"[eval] {name}: {len(recs)} 条")

    base = None
    if args.baseline:
        if not os.path.exists(args.baseline):
            print(f"基线文件不存在: {args.baseline}", file=sys.stderr)
            return 2
        base = list(iter_jsonl(args.baseline))
        reports.append(evaluate_one(base, cfg, "baseline"))

    comps: Dict[str, Any] = {}
    for recs, name in zip(all_recs, names):
        if base is not None:
            comps[f"{name} vs baseline"] = compare(recs, base, cfg)
        elif len(all_recs) > 1:
            comps[f"{name} vs {names[0]}"] = compare(recs, all_recs[0], cfg)

    md = to_markdown(reports, comps)
    print("\n" + md)

    if args.json:
        atomic_write_json(args.json, {"reports": reports, "comparisons": comps})
        print(f"[eval] json -> {args.json}")
    if args.markdown:
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown)), exist_ok=True)
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"[eval] markdown -> {args.markdown}")

    if args.strict:
        bad = [r["name"] for r in reports
               if r["veg"]["decorative"] and not _isnan(r["veg"]["veg_mean"])]
        if bad:
            print(f"[eval] STRICT: 以下 ckpt 的 VEG 落在装饰性区间: {bad}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
