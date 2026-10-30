#!/usr/bin/env python3
"""§6 验收闸门的**推理链**：把题集跑成 ``check_gate.py`` 能吃的扁平记录。

为什么要单独一个脚本：``check_gate.py`` 只做算术，它需要的是每行已经带
``condition / answer / n_calls / tool_oks / format_ok / needs_tool / correct``
的评测产物。仓库里原先没有产出这份产物的东西（``pipeline.py`` 的 stage_eval
只 print，``serve_local.py`` 只有单轮 generate），所以闸门一直跑不起来。

两个条件（PROPOSAL §6 的 "给定 gold 框 vs 不给工具"）::

    no_tool  : user[原图 + 问题] -> 直接作答
    gold_box : user[原图 + 问题] -> assistant 发一次 crop(gold 框)
               -> user[金标裁图 + 【工具返回】…【工具返回结束】] -> 作答

为何 gold_box 的裁剪是**注入**而不是让模型自己挑：这个条件衡量的是
"看到金标区域能涨多少 / 答案会不会因此改变"，框必须是金标框；让模型自己
挑框就把问题变成了"模型定位得准不准"，两个条件的差异被定位误差污染。
因此 ``n_calls=1`` 指的是这次被注入的调用，``tool_oks`` 记录它真实的执行结果。

``model_fn(messages, images, max_tokens, temperature) -> str | dict`` 是唯一的
模型入口：离线测试注入假模型，真跑时 ``--model-endpoint`` 会构造一个走
urllib 的 OpenAI 兼容实现（不硬编码任何 SDK）。

用法::

    # 离线自检：只造题、渲染、统计，不调模型
    python scripts/eval_gate.py --dry-run --limit 20

    # 真跑（端点需是 OpenAI 兼容的 /v1/chat/completions）
    python scripts/eval_gate.py --model-endpoint http://127.0.0.1:18422/v1/chat/completions \\
        --out data/eval/sft_gate.jsonl
    python scripts/check_gate.py --records data/eval/sft_gate.jsonl
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medtaco.data.training_set import OBS_HEADER, OBS_TAIL  # noqa: E402
from medtaco.rl.grader import DeterministicGrader  # noqa: E402
from medtaco.synth.crops import DEFAULT_PAD, crop_boxes_for, crops_dir  # noqa: E402
from medtaco.tools.ops import crop_with_box, load_image  # noqa: E402
from medtaco.utils.config import load_config  # noqa: E402
from medtaco.utils.io import JsonlWriter, iter_jsonl, map_parallel  # noqa: E402
from medtaco.utils.schema import Box, Sample  # noqa: E402

__all__ = [
    "CONDITIONS",
    "build_gate_items",
    "render_condition",
    "run_condition",
    "grade_record",
    "run_gate",
    "main",
]

#: 支持的实验条件；顺序即输出顺序
CONDITIONS: Tuple[str, str] = ("gold_box", "no_tool")

#: 严格答案抽取：只有完整且非空的 ``<answer>…</answer>`` 才算格式合法。
#: 刻意**不用** ``grader.parse``：它会退化成"最后一行非空文本"，一旦拿它判
#: format_ok，任何带字的东西都算合法，格式闸门等于没有。
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)

#: 两条件共用的作答指令。必须一模一样，否则 answer_change 混入提示差异。
DEFAULT_ANSWER_INSTRUCTION = (
    "请先简述你的观察，然后把最终答案写在 <answer></answer> 标签里。"
    "答案尽量简短（病名 / 是或否 / 数值+单位）。"
)

#: gold_box 条件下给模型的额外提示：这里必须点明已提供裁剪图，
#: 否则模型会以为自己在裸答，两个条件就退化成同一件事。
DEFAULT_CROP_HINT = "上一轮裁剪图已返回，请结合裁剪图作答。"

#: 观测文本（不含首尾标记），描述这次裁剪本身
_FAIL_OBS = "工具执行失败：{err}"


def _cfg_get(cfg: Any, path: str, default: Any) -> Any:
    """兼容 Cfg（点号路径）与普通 dict / None 的取值。"""
    if cfg is None:
        return default
    cur: Any = cfg
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return default if cur is None else cur


def _safe(sid: str) -> str:
    """文件名安全化：与 crops.py 的 _safe 保持同一规则。"""
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(sid))[:120]


# ---------------------------------------------------------------- gold 裁剪


#: 金标裁剪的 role 优先级：越靠前越贴近"金标框"。
#:
#: 必须与 ``synth/crops.prepare_crops`` 产出的 role 一一对应：
#:   - ``context`` / ``tight``：G1/G5 的金标框（外扩 / 不外扩），两者是同一个框；
#:   - ``inner``：G2 ``scale_boxes`` 的**最内层**，也就是金标病灶框；
#:   - ``measure``：G4 唯一一张，就是金标框；
#:   - ``outer``：G2 的**最外层**大范围框，**不是**金标，只作兜底。
#:
#: 取现成裁剪图（``_crop_from_meta``）与现场重裁（``_gold_plan``）必须共用这一份
#: 优先级，否则两条路径会给出不同的框。踩过的坑：``_gold_plan`` 曾直接取
#: ``plan[0]``，对 G2 拿到的是 outer（真实数据里 outer/gold 面积比中位 4x、最大 16x），
#: gold_box 条件退化成"看整图"。
GOLD_ROLE_ORDER: Tuple[str, ...] = ("context", "tight", "inner", "measure", "outer")

#: 需要外扩的 role（与 ``prepare_crops`` 的 pad_ratio 规则保持一致）
PADDED_ROLES = frozenset({"context", "outer"})


def _crop_from_meta(sample: Sample) -> Optional[str]:
    """``meta['crops']`` 里现成的金标裁剪图（prepare_crops 的产物）。

    按 ``GOLD_ROLE_ORDER`` 挑：context/tight 是 G1/G5 的金标框，inner 是 G2 的
    金标病灶框，measure 是 G4 的唯一一张。都不匹配时退回第一张。文件已被清理的
    条目直接跳过——宁可现场重裁，也不要交给下游一个不存在的路径。
    """
    crops = (sample.meta or {}).get("crops") or []
    if not isinstance(crops, list):
        return None
    by_role: Dict[str, str] = {}
    first: Optional[str] = None
    for c in crops:
        if not isinstance(c, dict):
            continue
        p = c.get("path")
        if not p or not os.path.exists(str(p)):
            continue
        first = first or str(p)
        role = str(c.get("role") or "")
        if role in GOLD_ROLE_ORDER and role not in by_role:
            by_role[role] = str(p)
    for role in GOLD_ROLE_ORDER:
        if role in by_role:
            return by_role[role]
    return first


def _gold_plan(sample: Sample) -> Optional[Tuple[Box, float]]:
    """决定"金标框 + 外扩比例"。返回 None 表示这条题没有可注入的金标区域。

    ``meta['need_crop'] is False``（G3 负样本）是权威的"无裁剪"标记：
    即便样本上还残留 box，也不给裁剪。

    选框规则与 ``_crop_from_meta`` 共用 ``GOLD_ROLE_ORDER``：**挑最内层的金标框**。
    G2 的 plan 是 ``[("outer", 最外层), ("inner", 金标)]``，取 ``plan[0]`` 会拿到
    大范围框（实测最大到金标框面积的 16 倍），gold_box 条件就退化成 full_view；
    ``outer`` 只在没有其它金标 role 时才兜底。
    """
    if (sample.meta or {}).get("need_crop") is False:
        return None
    plan = crop_boxes_for(sample)
    if plan:
        by_role: Dict[str, Box] = {}
        for role, box in plan:
            by_role.setdefault(role, box)
        for role in GOLD_ROLE_ORDER:
            if role in by_role:
                return by_role[role], (DEFAULT_PAD if role in PADDED_ROLES else 0.0)
        # 未知 role（crop_boxes_for 将来新增）：沿用旧的"取第一个"行为，不静默丢题
        role, box = plan[0]
        return box, DEFAULT_PAD if role in PADDED_ROLES else 0.0
    if sample.boxes:
        return sample.boxes[0], DEFAULT_PAD
    return None


def _gate_crops_dir(cfg: Any) -> str:
    """闸门裁剪图的落盘目录：优先 ``<paths.processed>/_crops``，无配置时退到系统临时目录。

    配置缺失时 crops_dir(None) 会给出相对路径 ``_crops``——那会在当前工作目录里
    写文件，跑在仓库根目录就是把一堆 png 撒进代码树，所以这里显式退到 tmp。
    """
    if cfg is not None and _cfg_get(cfg, "paths.processed", None):
        try:
            return crops_dir(cfg)
        except Exception:  # noqa: BLE001 - 配置异常不应让整批失败
            pass
    import tempfile

    return os.path.join(tempfile.gettempdir(), "medtaco_gate_crops")


def _ensure_gold_crop(item: Dict[str, Any], cfg: Any) -> Optional[str]:
    """拿到（必要时现场生成）金标裁剪图路径；失败返回 None。

    幂等：目标文件已存在就直接复用，反复跑闸门不会重画 2000 张图。
    """
    cur = item.get("gold_crop")
    if cur and os.path.exists(str(cur)):
        return str(cur)
    sample: Sample = item["sample"]
    plan = item.get("_gold_plan")
    if plan is None:
        plan = _gold_plan(sample)
    if plan is None:
        return None
    box, pad = plan
    path = os.path.join(_gate_crops_dir(cfg), f"{_safe(sample.id)}_gate.png")
    try:
        if not os.path.exists(path):
            img = load_image(sample.image)
            arr, _eff = crop_with_box(img, box, pad_ratio=pad)
            from PIL import Image  # 延迟导入：PIL 已是既有依赖，这里只是缩窄作用域

            os.makedirs(os.path.dirname(path), exist_ok=True)
            Image.fromarray(arr).save(path)
        item["gold_crop"] = path
        return path
    except Exception:  # noqa: BLE001 - 单条裁不出来记为缺图，不让整批中止
        return None


# ---------------------------------------------------------------- 造题


def _match_traj_ids(trajs_path: str) -> Optional[set]:
    """读出 validated 轨迹覆盖的 sample id 集合；文件不存在返回 None（不过滤）。

    本函数**只筛覆盖范围，不筛 verdict**：verdict（能不能进训练）在上游就已经由
    validated.jsonl 这一文件名表达，这里再判一次只会引入第二套口径。

    匹配键收敛到 ``sample_id`` 一列（行首 ``sample_id`` 优先，缺了取 ``traj.sample_id``）。
    早先这里把 ``rec.id`` / ``traj.id`` / 去 ``traj-`` 前缀三种键混在一起全部并进集合，
    任一路径命中就放行：轨迹 id 形如 ``traj-{sid}``，可一旦某条 sample id 本身就以
    ``traj-`` 开头，剥前缀出来的短串会**额外**放行一批本不该进来的题。
    """
    if not trajs_path or not os.path.exists(trajs_path):
        return None
    ids = set()
    for rec in iter_jsonl(trajs_path):
        traj = rec.get("traj") if isinstance(rec.get("traj"), dict) else {}
        sid = rec.get("sample_id") or traj.get("sample_id")
        if sid:
            ids.add(str(sid))
    return ids


def build_gate_items(questions_path: str, trajs_path: Optional[str], cfg: Any,
                     limit: Optional[int] = None) -> List[dict]:
    """构造闸门题集：只保留 ``grader.gradable`` 的题，并准备好金标裁剪图。

    参数
      trajs_path : 给了就只保留 validated 轨迹覆盖的题（同一批题才谈得上与
                   SFT 数据同分布）。文件不存在/为空时**不过滤**。
      limit      : 保留前 N 条。先过滤再截断，且裁剪图在截断之后才生成，
                   因此 ``--limit 20`` 只画 20 张图。

    返回项::

        {id, sample_id, sample, gold, needs_tool, gold_crop, dataset, gold_box}

    剔除规则：``needs_tool=True`` 但既没有金标框/尺度框（``_gold_plan`` 为空）
    也没有现成金标裁图的题，直接在这里丢掉并报告条数。这种题是**数据缺陷**
    （need_crop 标了 True 却没有区域），模型在 gold_box 条件下一次都调不到工具；
    留着它只会让下游写出一条 n_calls=1/format_ok=False 的假记录，把数据问题
    记成模型问题（tool_success_rate 会直接掉到 0）。
    """
    grader = DeterministicGrader(cfg)
    keep_ids = _match_traj_ids(trajs_path) if trajs_path else None

    items: List[dict] = []
    n_need_but_no_gold = 0
    for rec in iter_jsonl(str(questions_path)):
        if not isinstance(rec, dict):
            continue
        try:
            sample = Sample.from_json(rec)
        except (TypeError, KeyError, ValueError):
            continue
        if not grader.gradable(sample):
            continue
        if keep_ids is not None and str(sample.id) not in keep_ids:
            continue
        meta = sample.meta or {}
        # need_crop 是权威标记（ChestX-ray14 正样本 True / G3 负样本 False）；
        # 缺失时退回"有没有框"这一保守判断。
        needs_tool = bool(meta["need_crop"]) if meta.get("need_crop") is not None \
            else bool(sample.boxes)

        plan = _gold_plan(sample)
        gold_crop = _crop_from_meta(sample)
        # 需要工具却既没有框也没有现成裁图：数据缺陷，剔除（见 docstring）
        if needs_tool and plan is None and not gold_crop:
            n_need_but_no_gold += 1
            continue
        item: dict = {
            "id": sample.id,
            "sample_id": str(meta.get("source_sample_id") or sample.id),
            "sample": sample,
            "gold": sample.answer,
            "needs_tool": needs_tool,
            "gold_crop": gold_crop,
            "dataset": sample.dataset,
            "gold_box": [b.to_json() for b in sample.boxes[:1]],
            "_gold_plan": plan,
        }
        # 只给需要工具的题画裁剪图；负样本本来就没有裁剪
        if needs_tool:
            _ensure_gold_crop(item, cfg)
        items.append(item)
        if limit is not None and limit > 0 and len(items) >= limit:
            break
    if n_need_but_no_gold:
        print(f"[gate] 剔除 need_crop=True 但无金标区域的题 {n_need_but_no_gold} 条"
              f"（数据缺陷：这类题 gold_box 条件无框可注入）", file=sys.stderr)
    return items


# ---------------------------------------------------------------- 渲染


def _msg(role: str, content: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"role": role, "content": content}


def _answer_instruction(cfg: Any) -> str:
    return str(_cfg_get(cfg, "eval.gate.answer_instruction", DEFAULT_ANSWER_INSTRUCTION))


def _render_tool_call(box: List[float], pad: float) -> str:
    """与 ``training_set._render_call`` 同形：``\\n<tool_call>crop(...)</tool_call>\\n``。

    参数用 ``box=[x0, y0, x1, y1]`` 的列表形式：训练数据里的 crop 调用是
    ``x0=…, y0=…`` 的扁平键，两种都见过；这里取列表形式是因为它和
    ``meta['crops'][i]['box']`` 一致，回放时不必再拼一遍。
    """
    coords = ", ".join(f"{float(v):.1f}" for v in box[:4])
    return f"\n<tool_call>crop(box=[{coords}], pad_ratio={float(pad)}):</tool_call>\n"


def render_condition(item: Dict[str, Any], condition: str, cfg: Any) -> Dict[str, Any]:
    """按条件把一条题渲染成多轮对话。

    返回::

        {id, condition, need_continue, messages, images, crop_ok, error}

    gold_box 的结构（4 条消息、2 张图）::

        user      [原图] + 问题 + 作答指令
        assistant think（没有就先验文本时不发）
        assistant <tool_call>crop(gold box)</tool_call>
        user      [金标裁图] + 【工具返回】…【工具返回结束】

    no_tool 的结构（1 条消息、1 张图）::

        user      [原图] + 问题 + 作答指令

    裁剪失败时仍保留 tool_call 轮，观测文本写明失败原因——把失败伪装成
    "没有调用"会让 tool_success_rate 假性满分。
    """
    condition = str(condition).strip().lower()
    if condition not in CONDITIONS:
        raise ValueError(f"未知条件: {condition!r}，可选 {CONDITIONS}")

    sample: Sample = item["sample"]
    instruction = _answer_instruction(cfg)
    user_text = f"{sample.question}\n\n{instruction}"
    messages: List[Dict[str, Any]] = [
        _msg("user", [{"type": "image", "image": sample.image},
                      {"type": "text", "text": user_text}]),
    ]
    images: List[str] = [sample.image]
    out: Dict[str, Any] = {
        "id": item["id"], "condition": condition, "need_continue": True,
        "messages": messages, "images": images, "crop_ok": None, "error": "",
    }

    if condition == "no_tool":
        # 单段对话，不存在"等工具返回后继续"的第二段
        out["need_continue"] = False
        return out

    # ---- gold_box
    plan = item.get("_gold_plan")
    if plan is None:
        plan = _gold_plan(sample)
    if plan is None and item.get("gold_crop"):
        plan = (None, 0.0)  # 只有现成图、没有框：直接注入，不渲染具体坐标
    if plan is None:
        # 没有金标区域可给：这一条件对该题不成立，由 run_condition 记失败
        out.update(need_continue=False, error="no_gold_box")
        return out

    box, pad = plan
    crop_path = _ensure_gold_crop(item, cfg)
    think = str(item.get("prethink") or "")
    if think.strip():
        messages.append(_msg("assistant", [{"type": "text", "text": think}]))

    if box is not None:
        call_text = _render_tool_call(box.to_json(), pad)
    else:
        call_text = "\n<tool_call>crop(gold_box)</tool_call>\n"
    messages.append(_msg("assistant", [{"type": "text", "text": call_text}]))

    crop_hint = str(_cfg_get(cfg, "eval.gate.crop_hint", DEFAULT_CROP_HINT))
    if crop_path:
        obs_parts: List[Dict[str, Any]] = [{"type": "image", "image": crop_path}]
        images.append(crop_path)
        obs_text = crop_hint
        out["crop_ok"] = True
    else:
        obs_parts = []
        err = out.get("error") or "金标裁图生成失败"
        obs_text = _FAIL_OBS.format(err=err)
        out["crop_ok"] = False
        out["error"] = err
    messages.append(_msg("user", [
        *obs_parts,
        {"type": "text", "text": OBS_HEADER + obs_text + OBS_TAIL},
    ]))
    return out


# ---------------------------------------------------------------- 执行


def _strict_answer(raw: str) -> Tuple[str, bool]:
    """严格抽 ``<answer>``：取最后一个完整的非空块，抽不到返回 ("", False)。"""
    if not raw:
        return "", False
    matches = ANSWER_RE.findall(str(raw))
    for m in reversed(matches):
        ans = " ".join(str(m).split()).strip()
        if ans:
            return ans, True
    return "", False


def _normalize_model_out(out: Any) -> Tuple[str, int, bool]:
    """把 model_fn 的返回值归一成 ``(text, tokens, estimated)``。

    允许直接返回 str（测试替身最常用），也允许 ``{"text":…, "tokens":…}``
    （真实端点能拿到 usage）。拿不到 usage 时按字符数粗估，并置
    ``estimated=True``——闸门只看 avg_tokens 的量级，不追求精确，但估出来的
    值不能伪装成账单数字。
    """
    if isinstance(out, dict):
        text = str(out.get("text") or out.get("raw") or out.get("content") or "")
        tok = out.get("tokens")
        if tok is not None and int(tok) > 0:
            return text, int(tok), False
    else:
        text = "" if out is None else str(out)
    return text, max(1, len(text) // 4), True


def run_condition(model_fn: Callable[..., Any], item: Dict[str, Any], condition: str,
                  cfg: Any) -> dict:
    """跑一个条件，返回 ``check_gate`` 需要的扁平字段（外加 messages 备查）。

    ``model_fn(messages, images, max_tokens, temperature) -> str | dict``。

    字段口径：
      - ``answer``     : 从 ``<answer>`` 抽出的净答案；抽不到就是 ""
      - ``format_ok``  : **严格**——有非空 ``<answer>`` 块才算 True
      - ``n_calls``    : gold_box 恒 1（注入的那次 crop）；no_tool 从模型 raw 里
                         数 ``<tool_call>``（与 ``check_gate._from_messages`` 同一判据）
      - ``tool_oks``   : gold_box 里裁剪的实际成败；no_tool 恒 []（没有真实执行过
                         任何工具，伪造 True 会把 tool_success_rate 灌水）
      - ``correct``    : 由 grade_record 填（这里先留 None）

    no_tool 的 ``n_calls`` 必须从 raw 反推：硬编码 0 会让 over_call_rate 在数据流里
    没有任何入口可观测，"不需要工具却自发调用"这道闸门于是永远 PASS。
    """
    condition = str(condition).strip().lower()
    rendered = render_condition(item, condition, cfg)
    messages = rendered["messages"]
    images = rendered["images"]
    max_tokens = int(_cfg_get(cfg, "eval.gate.max_tokens", 512))
    temperature = float(_cfg_get(cfg, "eval.gate.temperature", 0.0))

    # 只有"gold_box 连金标区域都没有"这种情况才不调模型；裁剪执行失败仍然要
    # 让模型在带失败观测的上下文里作答（那正是 tool_success_rate 想统计的东西）。
    can_run = rendered.get("crop_ok") is not None or not rendered.get("error")
    raw = ""
    tokens = 0
    estimated = False
    if can_run:
        raw, tokens, estimated = _normalize_model_out(
            model_fn(messages, images, max_tokens, temperature))
    answer, format_ok = _strict_answer(raw)

    # 条件对该题不成立（no_gold_box）：模型一次都没被调用，绝不能写成
    # n_calls=1 / tool_oks=[False] 的"模型答错且工具失败"记录。
    invalid_item = bool(rendered.get("error") == "no_gold_box")
    if invalid_item:
        n_calls = 0
        tool_oks = []
    elif condition == "gold_box":
        n_calls = 1
        tool_oks = [bool(rendered.get("crop_ok"))]
    else:
        # no_tool 没有任何被执行的调用；模型若自己发 <tool_call>，只能从 raw 里看
        n_calls = str(raw or "").count("<tool_call>")
        tool_oks = []

    return {
        # uid 是**落盘主键**：同一题会写 gold_box 与 no_tool 两条，id 相同；
        # JsonlWriter 按 id 去重会让第二条 write() 静默返回 False，产物只剩一半。
        "uid": f"{item['id']}|{condition}",
        "id": item["id"],
        "condition": condition,
        "answer": answer,
        "raw": raw,
        "n_calls": n_calls,
        "tool_oks": tool_oks,
        "format_ok": bool(format_ok),
        "needs_tool": bool(item.get("needs_tool")),
        "dataset": item.get("dataset") or "",
        "tokens": int(tokens),
        "tokens_estimated": bool(estimated),
        "correct": None,
        "crop_ok": rendered.get("crop_ok"),
        "error": rendered.get("error") or "",
        "invalid_item": invalid_item,
        "messages": messages,
    }


def grade_record(out: dict, item: Dict[str, Any], grader: DeterministicGrader) -> dict:
    """填 ``correct``：只有 grader 给出 1 才算对（-1/0 都是错）。

    ``invalid_item`` 的记录（条件对该题不成立，模型根本没被调用）不打分：
    留 None 让下游按"缺测"处理，而不是按"答错"处理。
    """
    if out.get("invalid_item"):
        return out
    answer = str(out.get("answer") or "")
    out["correct"] = bool(grader.score(item["sample"], answer) == 1)
    return out


def run_gate(items: Sequence[dict], cfg: Any, model_fn: Callable[..., Any],
             conditions: Sequence[str] = CONDITIONS, workers: int = 8,
             grader: Optional[DeterministicGrader] = None,
             progress: bool = False) -> List[dict]:
    """并发跑全部 (题, 条件) 组合，返回扁平记录。

    跳过规则：``needs_tool is False`` 的题不跑 ``gold_box``——那条题根本没有
    金标区域，"给 gold 框"不成立；硬塞一条 n_calls=1 的记录会让 over_call_rate
    把注入的调用算成模型过度调用（负样本因此全被判成 over-call）。

    带 ``invalid_item`` 的记录（条件不成立、模型未被调用）不返回：它们既不是
    模型的答案也不是模型的调用，混进指标里就是把数据缺陷记成模型缺陷。
    build_gate_items 已经剔掉了这类题，这里是第二道防线（例如手工构造的 items）。
    """
    grader = grader or DeterministicGrader(cfg)
    tasks: List[Tuple[dict, str]] = []
    for it in items:
        for cond in conditions:
            if cond == "gold_box" and not it.get("needs_tool"):
                continue
            tasks.append((it, cond))
    if not tasks:
        return []

    def _one(task: Tuple[dict, str]) -> dict:
        it, cond = task
        return grade_record(run_condition(model_fn, it, cond, cfg), it, grader)

    results = map_parallel(
        tasks, _one, workers=max(1, int(workers)), desc="gate" if progress else "",
        on_error=lambda t, e: print(f"[gate] {t[0].get('id')} {t[1]} 失败: {e}",
                                    file=sys.stderr),
    )
    kept = [r for r in results if r is not None and not r.get("invalid_item")]
    n_invalid = sum(1 for r in results if r is not None and r.get("invalid_item"))
    if n_invalid:
        print(f"[gate] 丢弃条件不成立（no_gold_box）的记录 {n_invalid} 条", file=sys.stderr)
    return kept


# ---------------------------------------------------------------- 真实端点


def make_http_model_fn(endpoint: str, model: str, timeout: float = 180.0,
                       jpeg_quality: int = 90) -> Callable[..., Dict[str, Any]]:
    """OpenAI 兼容端点的 model_fn（只用标准库，不依赖任何 SDK）。

    图以 data URL 的 base64 JPEG 塞进 content；``images`` 与消息里的
    ``{"type":"image"}`` 块按出现顺序一一对应。
    """
    url = endpoint.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url + "/chat/completions"

    def _data_url(path: str) -> str:
        from PIL import Image

        img = load_image(str(path))
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=int(jpeg_quality))
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"

    def _content(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = []
        for b in msg.get("content") or []:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "image":
                parts.append({"type": "image_url",
                              "image_url": {"url": _data_url(str(b["image"]))}})
            elif b.get("type") == "text":
                parts.append({"type": "text", "text": str(b.get("text") or "")})
        return parts

    def fn(messages: List[Dict[str, Any]], images: List[str], max_tokens: int,
           temperature: float) -> Dict[str, Any]:
        body = json.dumps({
            "model": model,
            "messages": [{"role": m.get("role", "user"), "content": _content(m)}
                         for m in messages],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
        text = str(data["choices"][0]["message"].get("content") or "")
        usage = data.get("usage") or {}
        tok = usage.get("completion_tokens") or usage.get("output_tokens")
        return {"text": text, "tokens": int(tok) if tok is not None else -1}

    return fn


# ---------------------------------------------------------------- CLI


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="§6 验收闸门的推理链（产出 check_gate 的输入）")
    ap.add_argument("--questions", default=None,
                    help="题目 jsonl，默认 <paths.questions>/questions.jsonl")
    ap.add_argument("--trajs", default=None,
                    help="validated 轨迹 jsonl；给了就只跑被覆盖的题，默认自动探测")
    ap.add_argument("--out", default=None, help="记录输出 jsonl，默认 <paths.logs>/sft_gate.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 题（先过滤后截断）")
    ap.add_argument("--conditions", default=",".join(CONDITIONS),
                    help="逗号分隔，可选 gold_box,no_tool")
    ap.add_argument("--model-endpoint", default=None,
                    help="OpenAI 兼容端点（base 或完整 /chat/completions 路径）")
    ap.add_argument("--model", default=None, help="端点上的模型名")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dry-run", action="store_true", help="只造题/渲染/统计，不调模型")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = load_config(args.config)

    # limit<=0 曾经被当成"不限量"静默跑全量（真实数据上就是数十分钟到数小时的
    # 意外开销）。这里直接拒绝：想要全量就别传 --limit。
    if args.limit is not None and args.limit <= 0:
        print(f"--limit 必须是正整数（收到 {args.limit}）；全量请省略该参数",
              file=sys.stderr)
        return 2

    questions = args.questions or os.path.join(str(cfg.paths.questions), "questions.jsonl")
    if not os.path.exists(questions):
        print(f"题目文件不存在: {questions}", file=sys.stderr)
        return 2
    trajs = args.trajs
    if trajs is None:
        cand = os.path.join(str(cfg.paths.trajectories), "validated.jsonl")
        trajs = cand if os.path.exists(cand) else None

    conditions = [c.strip().lower() for c in str(args.conditions).split(",") if c.strip()]
    bad = [c for c in conditions if c not in CONDITIONS]
    if bad:
        print(f"未知条件: {bad}，可选 {CONDITIONS}", file=sys.stderr)
        return 2

    items = build_gate_items(questions, trajs, cfg, limit=args.limit)
    n_need = sum(1 for it in items if it.get("needs_tool"))
    print(f"[gate] 题集 {len(items)} 条（需要工具 {n_need}；有金标裁图 "
          f"{sum(1 for it in items if it.get('gold_crop'))}）")

    if args.dry_run:
        # 注意：这里不能拿 need_continue 当"可渲染"判据 —— no_tool 是单段对话，
        # need_continue 恒为 False，会统计成 0 条。可渲染 = 渲染过程没有 error。
        for cond in conditions:
            n_ok = 0
            n_tried = 0
            for it in items:
                if cond == "gold_box" and not it.get("needs_tool"):
                    continue
                n_tried += 1
                r = render_condition(it, cond, cfg)
                n_ok += not r.get("error")
            print(f"[gate] dry-run {cond}: 可渲染 {n_ok}/{n_tried} 条")
        return 0

    if not args.model_endpoint:
        print("非 dry-run 必须给 --model-endpoint（或用假 model_fn 在代码里调用 run_gate）",
              file=sys.stderr)
        return 2
    model = args.model or str(_cfg_get(cfg, "eval.gate.model",
                                       _cfg_get(cfg, "sft.base_model", "Qwen3-VL-8B-Instruct")))
    model_fn = make_http_model_fn(args.model_endpoint, model)

    records = run_gate(items, cfg, model_fn, conditions=tuple(conditions),
                       workers=args.workers, progress=True)
    out_path = args.out or os.path.join(str(cfg.paths.logs), "sft_gate.jsonl")
    # 主键用 uid（= id|condition）：同一题要写 gold_box 与 no_tool 两条，两条 id
    # 相同，按 id 去重会让第二条 write() 静默返回 False —— 产物只剩一半（条件分布
    # 变成 3:1），answer_change 永远配不上对。所以 id_key 必须是 uid，且每条
    # write() 都要断言成功，不能让去重把记录吃掉。
    # resume=False：闸门产物是"这一版模型的一次评测"，与旧记录混在一起会让
    # answer_change 配到两次运行的答案上。每次整份重写。
    written = 0
    with JsonlWriter(out_path, id_key="uid", resume=False) as w:
        for r in records:
            if not w.write(r):
                print(f"[gate] 记录被去重丢弃（uid 冲突）: {r.get('uid')}", file=sys.stderr)
                return 1
            written += 1
    if written != len(records):
        print(f"[gate] 落盘 {written} 条 != 内存 {len(records)} 条", file=sys.stderr)
        return 1
    print(f"[gate] {len(records)} 条记录写入 {out_path}")

    from scripts.check_gate import evaluate_gate, print_report

    gate = dict(cfg.get("sft.gate", {}) or {})
    # 报告用**刚落盘的**那份读回来算：内存对象与文件内容一旦分叉（去重、序列化
    # 丢字段），PASS 就成了谎报。evaluate_gate 需要能按 id+condition 配对的扁平记录。
    persisted = list(iter_jsonl(out_path))
    report = evaluate_gate(persisted, gate)
    # 口径必须写在报告里，否则读报告的人会以为 no_tool 的调用量是实测出来的：
    # gold_box 的 n_calls=1 是**注入**的那次 crop（不是模型自发调用），
    # no_tool 的 n_calls 是模型 raw 里 ``<tool_call>`` 的出现次数（自发调用）。
    oc = report.get("details", {}).get("over_call")
    if isinstance(oc, dict):
        oc["n_calls_source"] = ("gold_box=注入的 crop 调用（恒 1）；"
                                "no_tool=raw 中 <tool_call> 出现次数（模型自发）")
    print_report(report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
