"""三项防 bypassing 验证（PROPOSAL §5.3）。

直接让教师模型写轨迹的最大风险是「教师不看裁剪也能答对」，学生于是学到
「调工具是装饰」。本模块用构造性验证把这种轨迹筛掉：

1. ``premise_perturbation``（核心）：把轨迹里的第一次裁剪替换成**面积匹配的随机区域**，
   再让模型只回答最终问题。答案不变 → 裁剪是装饰性的 → ``reason="decorative_crop"``。
   没有裁剪的轨迹（G3 负样本）→ ``reason="no_tool_trajectory"``，是否通过取决于
   无工具消融（负样本本就应该「不给工具也能答对」）。
2. ``no_tool_ablation``：不给任何工具直接问全文题。答对 → 工具非必需
   → ``passed=False, reason="solved_without_tool"``。
3. ``tool_necessity``：结合前两项的读数给出最终判定——只有「扰动后答案变化」
   且「无工具答错」才认为裁剪是承重的。

所有对外函数都只吃 ``Trajectory`` / ``Sample`` / cfg，模型调用一律通过注入的
``ask_fn(prompt, image) -> str``，本模块不 import 任何 SDK。
"""
from __future__ import annotations

import math
import random
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from medtaco.utils.schema import Box, Sample, Step, Trajectory, Verdict

# ask_fn(prompt, image) -> 模型回答文本。image 是 HxWx3 uint8 的 numpy 数组。
AskFn = Callable[[str, np.ndarray], str]

# ---------------------------------------------------------------- 常量

#: 认为「是同一张裁剪」所需的最小面积（像素）
_MIN_BOX_SIDE: float = 4.0
#: 随机框重采样上限
_MAX_RESAMPLE: int = 50
#: 裁剪类工具名（crop 与 zoom 都按「裁剪」处理）
_CROP_TOOLS: Tuple[str, ...] = ("crop", "zoom")

#: 判分用的肯定/否定词表（只做轻量等价，正式判分以 rl/grader.py 为准）
_AFFIRM = {"yes", "y", "true", "present", "positive", "是", "有", "存在"}
_DENY = {"no", "n", "false", "absent", "negative", "否", "没有", "不存在", "无"}

_PUNCT = " \t\r\n.,;:!?\"'`()[]{}<>“”‘’。"
#: 否定前缀（用于「no effusion」≠「effusion」）；必须带词边界，避免 nodule 被 no 误吞
_NEG_PREFIX = re.compile(
    r"^(?:no|not|none|never|without|negative for|absence of)(?:\s|$)|^(?:无|没有|未见|不存在)"
)
#: 比较时忽略的虚词
_STOPWORDS = frozenset({"a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "at"})


# ---------------------------------------------------------------- 配置读取


def _cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    """按点号路径取配置；Cfg 与普通 dict 都能用，缺失返回 default。"""
    cur: Any = cfg
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return default if cur is None else cur


def _sub_cfg(cfg: Any, path: str) -> Any:
    """取一个子配置块（拿不到就返回空 dict）。"""
    v = _cfg_get(cfg, path, None)
    return v if isinstance(v, dict) else {}


# ---------------------------------------------------------------- 基础几何


def iou(a: Box, b: Box) -> float:
    """两个框的 IoU；任一面积为 0 时返回 0。"""
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _to_box(value: Any) -> Optional[Box]:
    """把 [x0,y0,x1,y1]/dict/Box 统一成 Box；非法返回 None。"""
    if value is None:
        return None
    if isinstance(value, Box):
        return value
    try:
        if isinstance(value, dict):
            return Box(float(value["x0"]), float(value["y0"]),
                       float(value["x1"]), float(value["y1"]),
                       str(value.get("label", "")))
        seq = list(value)
        if len(seq) < 4:
            return None
        return Box(float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3]),
                   str(seq[4]) if len(seq) > 4 else "")
    except (TypeError, ValueError, KeyError):
        return None


def area_matched_random_box(
    ref: Box,
    img_w: int,
    img_h: int,
    area_tol: float = 0.25,
    max_iou: float = 0.1,
    rng: Optional[random.Random] = None,
    max_tries: int = _MAX_RESAMPLE,
    avoid: Sequence[Box] = (),
) -> Optional[Box]:
    """采一个与 ``ref`` 面积匹配、且与 ``avoid`` 里所有框 IoU ≤ ``max_iou`` 的随机框。

    做法：均匀采中心，宽高比在 [0.5, 2] 内均匀采，按目标面积反推宽高；
    越界或约束不满足就重采样，最多 ``max_tries`` 次。失败返回 None。

    ``avoid`` 是对比对象集合（通常含 gold box），为空时退化为只对比 ``ref``。
    """
    if rng is None:
        rng = random.Random(0)
    ref_area = float(ref.area)
    if ref_area <= 0 or img_w <= 0 or img_h <= 0:
        return None
    obstacles: List[Box] = [b for b in (_to_box(v) for v in avoid) if b is not None]
    if not obstacles:
        obstacles = [ref]

    lo, hi = 1.0 - float(area_tol), 1.0 + float(area_tol)
    for _ in range(int(max_tries)):
        target_area = ref_area * rng.uniform(lo, hi)
        aspect = rng.uniform(0.5, 2.0)          # w / h
        bh = math.sqrt(target_area / aspect)
        bw = target_area / bh
        if bw > img_w or bh > img_h:
            continue
        # 中心均匀分布在「框完整落在图内」的可行区间上
        cx = rng.uniform(bw / 2.0, img_w - bw / 2.0)
        cy = rng.uniform(bh / 2.0, img_h - bh / 2.0)
        box = Box(cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0)
        box = box.clip(img_w, img_h)
        if box.w < _MIN_BOX_SIDE or box.h < _MIN_BOX_SIDE:
            continue
        ratio = box.area / ref_area
        if not (lo <= ratio <= hi):
            continue
        if max(iou(box, o) for o in obstacles) > max_iou:
            continue
        return box
    return None


# ---------------------------------------------------------------- 图像与轨迹


def _load_rgb(path: str) -> Optional[np.ndarray]:
    """读图成 HxWx3 uint8；失败返回 None（单测里不会走到磁盘）。"""
    try:
        from PIL import Image  # 延迟导入，便于纯逻辑测试

        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


def resolve_image(tool_ctx: Any, sample: Optional[Sample] = None) -> Optional[np.ndarray]:
    """从 tool_ctx 取全图数组。

    支持：直接给 np.ndarray / ToolContext（有 ``.image``）/ dict（键 ``image``）/
    文件路径。拿不到时退回 ``sample.image``。
    """
    img: Any = None
    if isinstance(tool_ctx, np.ndarray):
        img = tool_ctx
    elif isinstance(tool_ctx, str):
        img = tool_ctx
    elif isinstance(tool_ctx, dict):
        img = tool_ctx.get("image")
    elif tool_ctx is not None:
        img = getattr(tool_ctx, "image", None)

    if isinstance(img, str):
        img = _load_rgb(img)
    if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] >= 3:
        return img
    if isinstance(img, np.ndarray) and img.ndim == 2:      # 灰度兜底
        return np.stack([img] * 3, axis=-1)

    if sample is not None and sample.image:
        return _load_rgb(sample.image)
    return None


def crop_array(img: np.ndarray, box: Box) -> np.ndarray:
    """按像素框裁剪数组（整数取整 + 至少保留 1 像素）。"""
    h, w = img.shape[:2]
    x0 = int(round(max(0.0, min(float(box.x0), w - 1))))
    y0 = int(round(max(0.0, min(float(box.y0), h - 1))))
    x1 = int(round(max(x0 + 1, min(float(box.x1), w))))
    y1 = int(round(max(y0 + 1, min(float(box.y1), h))))
    return img[y0:y1, x0:x1]


def first_crop_step(traj: Trajectory) -> Optional[Step]:
    """轨迹里第一次裁剪类工具调用（crop / zoom）。"""
    for step in traj.steps:
        if step.call is not None and step.call.name in _CROP_TOOLS:
            return step
    return None


def crop_box_of(step: Step) -> Optional[Box]:
    """取一次裁剪调用的实际区域：优先 result.meta['crop_box']，退回 call.args['box']。"""
    if step.result is not None and isinstance(step.result.meta, dict):
        box = _to_box(step.result.meta.get("crop_box"))
        if box is not None:
            return box
    if step.call is not None and isinstance(step.call.args, dict):
        return _to_box(step.call.args.get("box"))
    return None


# ---------------------------------------------------------------- 答案比较


def normalize_answer(text: Any) -> str:
    """轻量归一化：小写、压缩空白、去首尾标点。仅用于「是否变化」判断。"""
    if text is None:
        return ""
    s = str(text).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(_PUNCT)


def _is_negative(text: str) -> bool:
    """归一化后的答案是否带否定前缀（no / not / 无 / 不 ...）。"""
    s = text.strip()
    if s in _DENY:
        return True
    return bool(_NEG_PREFIX.match(s))


def answers_match(a: Any, b: Any) -> bool:
    """两条答案是否等价。

    规则（故意保守）：
      1. 归一化后完全相等；
      2. 否定极性必须一致：「no effusion」不得匹配「effusion」；
      3. 肯定/否定词表内互相等价；
      4. 较短的一条的实词全部落进较长一条（「a small nodule」⟷「nodule」）。
    """
    na, nb = normalize_answer(a), normalize_answer(b)
    if not na or not nb:
        return na == nb
    if na == nb:
        return True
    if _is_negative(na) != _is_negative(nb):
        return False
    if na in _AFFIRM and nb in _AFFIRM:
        return True
    if na in _DENY and nb in _DENY:
        return True
    short, long_ = (na, nb) if len(na) <= len(nb) else (nb, na)
    # 「a small nodule」⟷「nodule」：短答案的词全部落进长答案（忽略冠词等虚词）
    stoks = [t for t in re.findall(r"\w+", short) if t not in _STOPWORDS]
    ltoks = set(re.findall(r"\w+", long_))
    if len(short) >= 3 and stoks and all(t in ltoks for t in stoks):
        return True
    return False


# ---------------------------------------------------------------- 提示词


def _answer_only_prompt(sample: Sample, cfg: Any, template_key: str) -> str:
    """只问最终问题的提示词（扰动/消融都用它，不给推理、不给工具）。"""
    tmpl = _cfg_get(cfg, template_key, None)
    if isinstance(tmpl, str) and tmpl:
        try:
            return tmpl.format(question=sample.question)
        except (KeyError, IndexError, ValueError):
            return tmpl
    return (
        "Answer the question using only the provided image. "
        "Do not explain; output the final answer on a single line.\n"
        f"Question: {sample.question}\n"
        "Answer:"
    )


# ---------------------------------------------------------------- 结果存取


def _store_check(traj: Trajectory, key: str, verdict: Verdict) -> None:
    """把子检查写进 traj.meta['validate'][key]，供 tool_necessity 复用。"""
    bucket = traj.meta.get("validate")
    if not isinstance(bucket, dict):
        bucket = {}
        traj.meta["validate"] = bucket
    bucket[key] = verdict.to_json()


def _read_check(traj: Trajectory, key: str) -> Optional[Dict[str, Any]]:
    """读回子检查（dict 或 Verdict 都接受），统一成 dict。"""
    bucket = traj.meta.get("validate")
    if not isinstance(bucket, dict):
        return None
    v = bucket.get(key)
    if v is None:
        return None
    if isinstance(v, Verdict):
        return v.to_json()
    if isinstance(v, dict):
        return v
    return None


def _check_skipped(check: Optional[Dict[str, Any]]) -> bool:
    """该子检查是否被配置关掉了。"""
    if not isinstance(check, dict):
        return False
    checks = check.get("checks") or {}
    return bool(checks.get("skipped"))


# ---------------------------------------------------------------- 检查一：前提扰动


def premise_perturbation(
    traj: Trajectory,
    sample: Sample,
    tool_ctx: Any,
    ask_fn: Optional[AskFn],
    cfg: Any,
) -> Verdict:
    """把第一次裁剪换成面积匹配的随机裁剪，看答案是否变化（PROPOSAL §5.3 第 2 项）。

    返回的 Verdict.passed 语义是「这条轨迹通过了前提扰动检查」：
      - 有裁剪：至少 ``min_answer_change`` 次试验里答案发生变化；
      - 无裁剪（G3 负样本）：取决于无工具消融——负样本本来就该「不给工具也能答对」。
    原始读数（每个随机框、每条答案、变化率）都留在 ``checks`` 里。
    """
    pcfg = _sub_cfg(cfg, "validate.premise_perturbation")
    enabled = bool(pcfg.get("enabled", True))
    area_tol = float(pcfg.get("area_match_tol", 0.25))
    n_trials = int(pcfg.get("n_replacement_trials", 3))
    min_change = int(pcfg.get("min_answer_change", 1))
    max_iou = float(pcfg.get("max_iou", 0.1))
    seed = int(pcfg.get("seed", _cfg_get(cfg, "seed", 42)))

    base_checks: Dict[str, Any] = {
        "enabled": enabled,
        "n_trials": n_trials,
        "min_answer_change": min_change,
        "area_match_tol": area_tol,
        "max_iou": max_iou,
        "seed": seed,
    }

    step = first_crop_step(traj)

    # ---- 分支 A：没有裁剪（G3 负样本）
    if step is None:
        ab = _read_check(traj, "no_tool_ablation")
        if ab is None:
            ab_v = no_tool_ablation(traj, sample, ask_fn, cfg)
            ab = ab_v.to_json()
            _store_check(traj, "no_tool_ablation", ab_v)
        base_checks["no_tool_ablation"] = ab
        if _check_skipped(ab) or not _cfg_get(cfg, "validate.no_tool_ablation.enabled", True):
            # 消融被关掉：负样本无从验证，按「不做判断、放行」处理并把原因记清楚
            return Verdict(traj.id, True, "no_tool_trajectory", base_checks)
        # 负样本成立 ⇔ 模型不给工具也能答对 ⇔ 消融判定为「工具非必需」
        solved = bool((ab.get("checks") or {}).get("answered_without_tool", False))
        return Verdict(traj.id, solved, "no_tool_trajectory", base_checks)

    # ---- 分支 B：有裁剪，做替换试验
    if not enabled:
        base_checks["skipped"] = True
        return Verdict(traj.id, True, "disabled", base_checks)

    crop_box = crop_box_of(step)
    if crop_box is None:
        base_checks["skipped"] = True
        base_checks["error"] = "crop_box_missing"
        return Verdict(traj.id, False, "crop_box_missing", base_checks)

    img = resolve_image(tool_ctx, sample)
    if img is None:
        base_checks["error"] = "image_unavailable"
        return Verdict(traj.id, False, "image_unavailable", base_checks)

    img_h, img_w = int(img.shape[0]), int(img.shape[1])
    crop_box = crop_box.clip(img_w, img_h)
    if crop_box.w < _MIN_BOX_SIDE or crop_box.h < _MIN_BOX_SIDE:
        base_checks["error"] = "crop_box_degenerate"
        return Verdict(traj.id, False, "crop_box_degenerate", base_checks)

    if ask_fn is None:
        base_checks["error"] = "ask_fn_missing"
        return Verdict(traj.id, False, "ask_fn_missing", base_checks)

    # IoU 约束的参照：轨迹裁剪框 + 样本里落在图内的 gold box
    obstacles: List[Box] = [crop_box]
    for b in sample.boxes:
        try:
            cb = b.clip(img_w, img_h)
        except (TypeError, ValueError):
            continue
        if cb.area >= _MIN_BOX_SIDE * _MIN_BOX_SIDE:
            obstacles.append(cb)

    baseline = traj.final_answer if normalize_answer(traj.final_answer) else sample.answer
    prompt = _answer_only_prompt(sample, cfg, "validate.premise_perturbation.prompt_template")
    rng = random.Random(seed)

    boxes: List[List[float]] = []
    answers: List[str] = []
    changed: List[bool] = []
    errors: List[str] = []

    for _ in range(max(0, n_trials)):
        rbox = area_matched_random_box(
            crop_box, img_w, img_h, area_tol=area_tol, max_iou=max_iou,
            rng=rng, max_tries=_MAX_RESAMPLE, avoid=obstacles,
        )
        if rbox is None:
            errors.append("resample_exhausted")
            continue
        patch = crop_array(img, rbox)
        try:
            ans = ask_fn(prompt, patch)
        except Exception as exc:                      # 模型侧异常不该炸掉整批
            errors.append(f"ask_fn_error:{type(exc).__name__}")
            continue
        ans = "" if ans is None else str(ans)
        boxes.append([rbox.x0, rbox.y0, rbox.x1, rbox.y1])
        answers.append(ans)
        changed.append(not answers_match(ans, baseline))

    n_valid = len(changed)
    n_change = int(sum(changed))
    base_checks.update({
        "crop_box": list(crop_box.as_tuple()),
        "image_size": [img_w, img_h],
        "baseline_answer": baseline,
        "perturbed_boxes": boxes,
        "perturbed_answers": answers,
        "answer_changed": changed,
        "n_valid_trials": n_valid,
        "n_answer_change": n_change,
        "change_rate": (n_change / n_valid) if n_valid else 0.0,
        "errors": errors,
    })

    if n_valid == 0:
        # 一个有效替换都没造出来：不能宣称通过（保守失败）
        return Verdict(traj.id, False, "perturbation_failed", base_checks)

    passed = n_change >= max(1, min_change)
    return Verdict(traj.id, passed, "" if passed else "decorative_crop", base_checks)


# ---------------------------------------------------------------- 检查二：无工具消融


def no_tool_ablation(
    traj: Trajectory,
    sample: Sample,
    ask_fn: Optional[AskFn],
    cfg: Any,
) -> Verdict:
    """不给工具直接问整图（PROPOSAL §5.3）。

    ``passed=True`` 表示「工具是必需的」（模型裸答答错），``passed=False`` 表示
    「工具非必需」（裸答就答对，``reason="solved_without_tool"``）。
    比较基准是 ``sample.answer``，为空时退回轨迹的 final_answer。
    """
    acfg = _sub_cfg(cfg, "validate.no_tool_ablation")
    enabled = bool(acfg.get("enabled", True))
    checks: Dict[str, Any] = {"enabled": enabled, "model": acfg.get("model", "")}

    if not enabled:
        checks["skipped"] = True
        return Verdict(traj.id, True, "disabled", checks)
    if ask_fn is None:
        checks["skipped"] = True
        checks["error"] = "ask_fn_missing"
        return Verdict(traj.id, True, "disabled", checks)

    img = resolve_image(None, sample)
    if img is None:
        checks["error"] = "image_unavailable"
        return Verdict(traj.id, False, "image_unavailable", checks)

    reference = sample.answer if normalize_answer(sample.answer) else traj.final_answer
    if not normalize_answer(reference):
        checks["error"] = "missing_reference_answer"
        return Verdict(traj.id, False, "missing_reference", checks)

    prompt = _answer_only_prompt(sample, cfg, "validate.no_tool_ablation.prompt_template")
    try:
        raw = ask_fn(prompt, img)
    except Exception as exc:
        checks["error"] = f"ask_fn_error:{type(exc).__name__}"
        return Verdict(traj.id, False, "ask_fn_error", checks)

    answer = "" if raw is None else str(raw)
    solved = answers_match(answer, reference)
    checks.update({
        "answer": answer,
        "reference_answer": reference,
        "answered_without_tool": solved,
    })
    if solved:
        return Verdict(traj.id, False, "solved_without_tool", checks)
    return Verdict(traj.id, True, "", checks)


# ---------------------------------------------------------------- 检查三：工具必要性判定


def sample_needs_crop(sample: Sample) -> bool:
    """样本是否"理论上应当裁剪"。

    ``meta["need_crop"]`` 是 qgen 写下的权威标记；缺这个键时（老数据、单测里的
    手工样本）退回"有没有金框"。带金框却一步没裁，是轨迹的问题，不是负样本。
    """
    meta = getattr(sample, "meta", None) or {}
    if "need_crop" in meta:
        return bool(meta.get("need_crop"))
    return bool(getattr(sample, "boxes", None))


def tool_necessity(traj: Trajectory, sample: Sample, cfg: Any) -> Verdict:
    """综合前两项读数给出最终判定（不重新调模型，只读 ``traj.meta['validate']``）。

    判定规则：
      - 有裁剪：``premise.passed and ablation.passed`` —— 扰动后答案变化 **且**
        无工具答错，裁剪才算承重；
      - 无裁剪（负样本）：模型必须**裸答答对**，否则该负样本与题面矛盾，丢弃。
    ``checks`` 里原样保留三项的原始数值，便于事后分析。
    """
    prem = _read_check(traj, "premise_perturbation")
    abl = _read_check(traj, "no_tool_ablation")
    checks: Dict[str, Any] = {
        "premise_perturbation": prem,
        "no_tool_ablation": abl,
    }
    if prem is None:
        return Verdict(traj.id, False, "missing_premise_check", checks)

    prem_checks = prem.get("checks") or {}
    if prem_checks.get("skipped"):
        checks["decision"] = "disabled"
        return Verdict(traj.id, True, "disabled", checks)
    no_crop = prem.get("reason") == "no_tool_trajectory"

    # ---- 该裁却没裁：样本带金框（need_crop=True）而教师一步工具都没调。
    # 不能再走负样本分支——那里要求「裸答答对」，可这类样本恰恰是"裸答答不对、
    # 才需要裁剪"的正样本，落到 negative_sample_unsolved 会把原因指向样本，
    # 实际责任在轨迹。方向上都判失败，只是原因必须说准，否则没法按原因归因。
    if no_crop and sample_needs_crop(sample):
        checks["decision"] = "expected_crop_missing"
        return Verdict(traj.id, False, "expected_crop_missing", checks)

    # ---- 负样本（G3）：裸答答对才是好负样本
    if no_crop:
        abl_checks = (abl or {}).get("checks") or {}
        if not abl_checks:
            # 只有独立调用过 no_tool_ablation 时 prem.checks 里才带着这份快照
            snap = prem_checks.get("no_tool_ablation")
            if isinstance(snap, dict):
                abl_checks = snap.get("checks") or {}
        if abl_checks.get("skipped"):
            checks["decision"] = "negative_sample_unverified"
            return Verdict(traj.id, True, "negative_sample_unverified", checks)
        solved = bool(abl_checks.get("answered_without_tool", False))
        checks["decision"] = "negative_sample" if solved else "negative_sample_unsolved"
        if solved:
            return Verdict(traj.id, True, "negative_sample", checks)
        return Verdict(traj.id, False, "negative_sample_unsolved", checks)

    # ---- 正样本：两道关都要过
    prem_passed = bool(prem.get("passed"))
    abl_checks = (abl or {}).get("checks") or {}
    abl_skipped = bool(abl_checks.get("skipped")) or abl is None
    abl_passed = bool((abl or {}).get("passed", True))

    checks["premise_passed"] = prem_passed
    checks["ablation_skipped"] = abl_skipped
    checks["ablation_passed"] = abl_passed
    checks["change_rate"] = prem_checks.get("change_rate", 0.0)

    # 该不裁却裁了（G3 负样本）。这一条排在最前：对这类样本，"扰动后答案没变"
    # 本来就是意料之中，拿 decorative_crop 去解释会把真正的问题（轨迹教学生
    # 看整图也要先裁一刀）掩盖掉。
    if not sample_needs_crop(sample):
        checks["decision"] = "unexpected_crop"
        return Verdict(traj.id, False, "unexpected_crop", checks)

    if not prem_passed:
        checks["decision"] = "decorative_crop"
        return Verdict(traj.id, False, prem.get("reason") or "decorative_crop", checks)
    if not abl_skipped and not abl_passed:
        checks["decision"] = "solved_without_tool"
        return Verdict(traj.id, False, "solved_without_tool", checks)

    checks["decision"] = "tool_necessary"
    return Verdict(traj.id, True, "", checks)


# ---------------------------------------------------------------- 组合入口


def validate_all(
    traj: Trajectory,
    sample: Sample,
    tool_ctx: Any,
    ask_fn: Optional[AskFn],
    cfg: Any,
) -> Verdict:
    """跑完三项检查并返回最终 Verdict（checks 里保留三份原始读数）。

    执行顺序：无工具消融 → 前提扰动（会复用已算好的消融结果，不重复调模型）
    → 工具必要性判定。子检查同时写进 ``traj.meta['validate']``，方便落盘复盘。
    """
    abl = no_tool_ablation(traj, sample, ask_fn, cfg)
    _store_check(traj, "no_tool_ablation", abl)

    prem = premise_perturbation(traj, sample, tool_ctx, ask_fn, cfg)
    _store_check(traj, "premise_perturbation", prem)

    final = tool_necessity(traj, sample, cfg)

    final.checks["premise_perturbation"] = prem.to_json()
    final.checks["no_tool_ablation"] = abl.to_json()
    final.checks["final"] = {
        "decision": final.checks.get("decision"),
        "reason": final.reason,
        "passed": final.passed,
    }
    _store_check(traj, "tool_necessity", final)
    return final
