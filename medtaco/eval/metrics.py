"""评测指标：准确性 + 忠实性 VEG + 配对报告（PROPOSAL §8）。

设计要点（与 PROPOSAL §8.1/§8.3 对齐）
------------------------------------------------
1. 准确性只报比例不足以证明"工具真的被用上了"（DeepEyes 83.3→83.8 的反例），
   因此同时报 VEG = Acc_正常 − Acc_扰动。**VEG≈0 即结构不生效**。
2. OmniMedVQA 是聚合数据集，必须按子源分项报告，禁止报混合均值。
   `per_source_report` 会把 `OmniMedVQA` 拆成 `OmniMedVQA/<子源>`；
   缺失子源的桶标记为 `mixed=True` 并**从 overall 中剔除**。
3. 所有区间/检验调用 `medtaco.utils.stats`，不自己造第二套统计实现。
4. 随机性全部走显式 seed（corrupt / bootstrap 各自的 seed 分开）。

记录（record）约定
------------------------------------------------
本模块只消费普通的 dict，不强制 schema。识别的字段（全部可选，缺失则相应指标为 nan）：

    id            str        题目 id（配对用）
    dataset       str        数据集名（分源主键，默认 source_key）
    source        str        子源名（如 OmniMedVQA 的 Kvasir），用于拆混合源
    correct       bool       该记录在"正常条件"下是否答对
    image         str|ndarray 返回图路径或数组（VEG 需要）
    n_calls       int        工具调用次数
    tokens        int        视觉/总 token 数
    tool_ok       bool       该记录的工具调用是否全部成功
    tool_oks      list[bool] 逐次调用的成功标记（优先于 tool_ok）
    box           list[4]    gold 框
    pred_box      list[4]    预测框
    condition     str        'full' | 'oracle_crop'（oracle 上界对比用）

契约签名（INTERFACES.md 要求逐字一致）
------------------------------------------------
    iou(a, b) / veg(acc_normal, acc_corrupted)
    oracle_crop_ceiling(...) / paired_report(model_a, model_b, n_boot, icc, m)
    per_source_report(records, source_key="dataset")
"""
from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from medtaco.utils import stats
from medtaco.utils.schema import Box

# 三种破坏模式（与 configs/default.yaml: eval.veg.corrupt_modes 一致）
CORRUPT_MODES: Tuple[str, ...] = ("random_crop_area_matched", "shuffle_pixels", "blackout")

ImageLike = Union[np.ndarray]
RngLike = Union[np.random.Generator, int, None]


# ============================================================ 基础：IoU / VEG


def iou(a: Box, b: Box) -> float:
    """标准 IoU（交并比）。退化框（宽或高 <= 0）一律返回 0.0。

    退化定义按"面积非正"判，而不是按 Box.__post_init__ 的合法性判：
    Box(5, 5, 5, 9) 在 schema 里是合法框（x1 >= x0），但宽度为 0，
    任何 IoU 计算的分母都可能为 0，因此这里直接归零。
    """
    if a.w <= 0 or a.h <= 0 or b.w <= 0 or b.h <= 0:
        return 0.0
    iw = min(a.x1, b.x1) - max(a.x0, b.x0)
    ih = min(a.y1, b.y1) - max(a.y0, b.y0)
    if iw <= 0 or ih <= 0:  # 只共享一条边不算相交
        return 0.0
    inter = iw * ih
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def veg(acc_normal: float, acc_corrupted: float) -> float:
    """Visual Evidence Gain = Acc_正常 − Acc_扰动。

    符号语义：**VEG > 0 表示破坏返回图后准确率下降**，即模型确实用到了这张裁剪图。
    VEG ≈ 0 → 工具是装饰性的；VEG < 0 → 破坏反而更好（通常是模型被无关内容干扰）。
    """
    if acc_normal is None or acc_corrupted is None:
        return float("nan")
    if math.isnan(acc_normal) or math.isnan(acc_corrupted):
        return float("nan")
    return float(acc_normal) - float(acc_corrupted)


# ============================================================ 图像破坏


def _as_rng(rng: RngLike) -> np.random.Generator:
    """把 seed / Generator / None 统一成 numpy Generator。"""
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None:
        # 不静默引入随机性：调用方不给 seed 时用固定的 0，保证可复现
        return np.random.default_rng(0)
    return np.random.default_rng(int(rng))


def _to_pil(arr: np.ndarray):
    """仅在本模块内部做 resize 用，避免强依赖 PIL 的顶层 import。"""
    from PIL import Image

    if arr.dtype != np.uint8:
        a = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        a = arr
    # 不传 mode：新版 Pillow 已废弃该参数，颜色模式由数组 shape 自动推断
    return Image.fromarray(a)


def _resize(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """双线性 resize 到 (h, w)，保持通道与 dtype。"""
    im = _to_pil(arr).resize((w, h), resample=1)  # 1 == Image.BILINEAR
    out = np.asarray(im)
    if out.dtype != arr.dtype:
        out = out.astype(arr.dtype)
    return out


def _random_area_crop(arr: np.ndarray, rng: np.random.Generator,
                      source: Optional[np.ndarray] = None) -> np.ndarray:
    """"面积匹配的随机裁剪"：取一块与 arr 面积相同、但位置随机的区域。

    - 给了 source（全图）时：在 source 上按同面积、同长宽比随机取框，再缩放回 arr 的尺寸。
      这是 PROPOSAL §5.3 / §8.2 的口径——"多看了同一大小的一块，但看错了地方"。
    - 没给 source 时：退化为对 arr 做非零整数位移的环绕平移（wrap-around shift）。
      语义仍是"面积不变、内容错位"，只是内容来自自身，强度较弱，会在返回值里说明。
    """
    h, w = arr.shape[:2]
    if h <= 0 or w <= 0:
        return arr.copy()

    if source is None:
        dy = int(rng.integers(1, h)) if h > 1 else 0
        dx = int(rng.integers(0, w)) if w > 1 else 0
        if dy == 0 and dx == 0:
            dx = 1 if w > 1 else 0
        return np.roll(arr, (dy, dx), axis=(0, 1)).copy()

    src = np.asarray(source)
    sh, sw = src.shape[:2]
    area = float(h * w)
    # 同长宽比 → 由面积反推高宽
    aspect = w / h
    ch = int(round(math.sqrt(area / aspect)))
    cw = int(round(area / max(ch, 1)))
    if ch > sh or cw > sw or ch <= 0 or cw <= 0:
        # 全图比目标区域还小：无信息可替换，退回位移
        dy = int(rng.integers(1, h)) if h > 1 else 0
        dx = int(rng.integers(0, w)) if w > 1 else 0
        return np.roll(arr, (dy, max(dx, 1)), axis=(0, 1)).copy()
    y0 = int(rng.integers(0, sh - ch + 1))
    x0 = int(rng.integers(0, sw - cw + 1))
    patch = src[y0:y0 + ch, x0:x0 + cw]
    return _resize(patch, h, w)


def corrupt_crop(img: np.ndarray, mode: str, rng: RngLike,
                 source: Optional[np.ndarray] = None) -> np.ndarray:
    """对工具返回的裁剪图做可控破坏，返回新数组（不改原数组）。

    mode:
      - ``random_crop_area_matched``：同面积随机区域替换（有 source 时从全图取，最推荐）。
      - ``shuffle_pixels``：全局像素置换（保留强度直方图，摧毁空间结构）。
      - ``blackout``：全零（保留尺寸；对归一化输入等价于"黑图"）。
    其余 mode 抛 ValueError（不要静默返回原图，否则 VEG 会被系统性低估）。

    rng 接受 numpy Generator / int seed / None；**同一 seed 结果完全一致**。
    """
    if mode not in CORRUPT_MODES:
        raise ValueError(f"未知破坏模式: {mode!r}，可选 {CORRUPT_MODES}")
    arr = np.asarray(img)
    if arr.ndim < 2:
        raise ValueError(f"图像维度非法: shape={arr.shape}")
    g = _as_rng(rng)

    if mode == "blackout":
        return np.zeros_like(arr)

    if mode == "shuffle_pixels":
        h, w = arr.shape[:2]
        flat = arr.reshape(h * w, -1)
        perm = g.permutation(h * w)
        return flat[perm].reshape(arr.shape).copy()

    return _random_area_crop(arr, g, source)


# ============================================================ run_veg


def _as_correct(v: Any) -> bool:
    """把 evaluate_fn 的返回值归一成 bool。支持 bool / 0-1 / {'correct': ...}。"""
    if isinstance(v, dict):
        for k in ("correct", "ok", "pass", "is_correct"):
            if k in v:
                return bool(v[k])
        raise KeyError("evaluate_fn 返回的 dict 缺少 correct/ok/pass 字段")
    return bool(v)


def _get_image(rec: Dict[str, Any], image_key: str = "image") -> Optional[np.ndarray]:
    """从记录里取图像数组：优先内存数组，其次存在的文件路径。"""
    for key in (image_key, "image_array", "arr", "pixels", "crop"):
        v = rec.get(key)
        if isinstance(v, np.ndarray):
            return v
        if isinstance(v, (str, os.PathLike)) and os.path.exists(v):
            from PIL import Image

            with Image.open(v) as im:
                return np.asarray(im.convert("RGB"))
    return None


def run_veg(records: Sequence[Dict[str, Any]],
            corrupt_fn: Callable[..., np.ndarray],
            evaluate_fn: Callable[[Dict[str, Any], np.ndarray], Any],
            modes: Optional[Sequence[str]] = None,
            n_boot: int = 2000,
            seed: int = 42,
            image_key: str = "image") -> Dict[str, Any]:
    """遍历记录，每条跑"正常 / 破坏"两路，给出每 mode 的 acc 与 VEG + bootstrap CI。

    参数
      records      : 记录列表（字段见模块 docstring）。
      corrupt_fn   : ``(img, mode, rng) -> img``，通常就是本模块的 ``corrupt_crop``。
      evaluate_fn  : ``(record, img) -> bool | {'correct': bool}``。
                     正常一路若 record 自带 ``correct`` 则不再调用它（省一次推理）。
      modes        : 默认三种 ``CORRUPT_MODES``。
      seed         : 每条记录 × 每个 mode 用 [seed, record_idx, mode_idx] 派生 rng，
                     因此结果与记录顺序无关、可完全复现。

    返回
      ``{n, n_skipped, n_degenerate, acc_normal, acc_normal_wilson, acc_normal_boot,
         modes, per_mode: {mode: {n, acc_corrupted, acc_corrupted_boot, veg, veg_ci, ...}},
         veg: {mode: float}, veg_mean}``

    跳过规则：取不到图像、或拿不到正常一路的判定 → 计入 ``n_skipped``（三路同时跳过，
    保证各 mode 的样本集合与正常一路严格配对）。
    """
    mode_list = list(modes) if modes is not None else list(CORRUPT_MODES)
    normal_flags: List[float] = []
    corr_flags: Dict[str, List[float]] = {m: [] for m in mode_list}
    n_skipped = 0
    n_degenerate = 0

    for i, rec in enumerate(records):
        img = _get_image(rec, image_key)
        if img is None:
            n_skipped += 1
            continue
        if "correct" in rec and rec["correct"] is not None:
            normal = bool(rec["correct"])
        elif evaluate_fn is not None:
            normal = _as_correct(evaluate_fn(rec, img))
        else:
            n_skipped += 1
            continue
        normal_flags.append(1.0 if normal else 0.0)

        for j, mode in enumerate(mode_list):
            rng = np.random.default_rng([int(seed), i, j])
            corrupted = corrupt_fn(img, mode, rng)
            if np.array_equal(corrupted, img):
                n_degenerate += 1  # 破坏没生效 → 该样本无信息量，需人工检查
            if evaluate_fn is None:
                # 无评估函数：只允许记录里预存了逐 mode 的结果
                pre = rec.get("corrupted_correct") or {}
                if mode not in pre:
                    raise ValueError("evaluate_fn 为 None 时必须提供 record['corrupted_correct']")
                ok = _as_correct(pre[mode])
            else:
                ok = _as_correct(evaluate_fn(rec, corrupted))
            corr_flags[mode].append(1.0 if ok else 0.0)

    n = len(normal_flags)
    if n == 0:
        return {
            "n": 0, "n_skipped": n_skipped, "n_degenerate": n_degenerate,
            "acc_normal": float("nan"), "acc_normal_wilson": (float("nan"),) * 2,
            "acc_normal_boot": (float("nan"),) * 3,
            "modes": mode_list, "per_mode": {}, "veg": {},
            "veg_mean": float("nan"),
        }

    acc_normal = float(sum(normal_flags) / n)
    k_normal = int(sum(normal_flags))
    per_mode: Dict[str, Dict[str, Any]] = {}
    veg_by_mode: Dict[str, float] = {}

    for mode in mode_list:
        flags = corr_flags[mode]
        acc_corr = float(sum(flags) / len(flags)) if flags else float("nan")
        est_b, blo_b, bhi_b = stats.bootstrap_ci(flags, n_boot=n_boot, seed=int(seed))
        # 配对 delta 的 bootstrap：同一批题，保留配对结构
        est_d, blo_d, bhi_d = stats.paired_bootstrap(normal_flags, flags,
                                                     n_boot=n_boot, seed=int(seed))
        wlo, whi = stats.wilson_ci(int(sum(flags)), len(flags))
        v = veg(acc_normal, acc_corr)
        veg_by_mode[mode] = v
        per_mode[mode] = {
            "n": len(flags),
            "acc_corrupted": acc_corr,
            "acc_corrupted_wilson": (wlo, whi),
            "acc_corrupted_boot": (est_b, blo_b, bhi_b),
            "veg": v,
            "veg_ci": (est_d, blo_d, bhi_d),
            "flips_down": int(sum(1 for a, b in zip(normal_flags, flags) if a > b)),
            "flips_up": int(sum(1 for a, b in zip(normal_flags, flags) if a < b)),
        }

    finite = [v for v in veg_by_mode.values() if not math.isnan(v)]
    wlo_n, whi_n = stats.wilson_ci(k_normal, n)
    return {
        "n": n,
        "n_skipped": n_skipped,
        "n_degenerate": n_degenerate,
        "acc_normal": acc_normal,
        "acc_normal_wilson": (wlo_n, whi_n),
        "acc_normal_boot": stats.bootstrap_ci(normal_flags, n_boot=n_boot, seed=int(seed)),
        "modes": mode_list,
        "per_mode": per_mode,
        "veg": veg_by_mode,
        "veg_mean": float(sum(finite) / len(finite)) if finite else float("nan"),
    }


# ============================================================ 配对报告


def paired_report(model_a: Sequence[bool], model_b: Sequence[bool],
                  n_boot: int = 2000, icc: float = 0.0, m: float = 1.0,
                  seed: int = 42) -> Dict[str, Any]:
    """同一批题上比较两个模型（配对）：acc + Wilson CI + McNemar p + 差值 CI + 有效样本量。

    参数
      model_a/model_b : 等长的 0/1 正确性序列，第 i 位对应同一道题。
      icc             : 组内相关系数（一图多问的聚类效应），DEFF = 1+(m-1)·ICC。
      m               : 每簇平均题数（一图平均几问）。

    McNemar 的 b/c 定义沿用 utils.stats：b = A 对 B 错，c = A 错 B 对。
    """
    a = [1.0 if x else 0.0 for x in model_a]
    b = [1.0 if x else 0.0 for x in model_b]
    if len(a) != len(b):
        raise ValueError(f"配对比较要求等长: {len(a)} != {len(b)}")
    n = len(a)
    if n == 0:
        return {"n": 0, "acc_a": float("nan"), "acc_b": float("nan"),
                "diff": float("nan"), "mcnemar": {"b": 0, "c": 0, "chi2": 0.0, "p": 1.0},
                "effective_n": 0.0, "deff": 1.0, "icc": icc, "m": m}

    ka = int(sum(a))
    kb = int(sum(b))
    b01 = int(sum(1 for x, y in zip(a, b) if x > y))  # A 对 B 错
    c10 = int(sum(1 for x, y in zip(a, b) if x < y))  # A 错 B 对
    chi2, p = stats.mcnemar(b01, c10)
    est, lo, hi = stats.paired_bootstrap(a, b, n_boot=n_boot, seed=int(seed))
    d = stats.deff(m, icc)
    eff_n = stats.effective_n(n, m, icc)
    a_lo, a_hi = stats.wilson_ci(ka, n)
    b_lo, b_hi = stats.wilson_ci(kb, n)

    return {
        "n": n,
        "n_boot": n_boot,
        "seed": seed,
        "acc_a": ka / n,
        "acc_a_wilson": (a_lo, a_hi),
        "k_a": ka,
        "acc_b": kb / n,
        "acc_b_wilson": (b_lo, b_hi),
        "k_b": kb,
        "diff": ka / n - kb / n,
        "diff_boot_ci": (est, lo, hi),
        "diff_boot_lo": lo,
        "diff_boot_hi": hi,
        "mcnemar": {"b": b01, "c": c10, "discordant": b01 + c10, "chi2": chi2, "p": p},
        "icc": icc,
        "m": m,
        "deff": d,
        "effective_n": eff_n,
        # 名义 N 下的 95% 半宽 vs 有效 N 下的半宽（有效 N 更小 → 区间更宽）
        "halfwidth_nominal": (a_hi - a_lo) / 2,
        "halfwidth_effective": (a_hi - a_lo) / 2 * math.sqrt(n / eff_n) if eff_n > 0 else float("nan"),
    }


# ============================================================ 分源报告


def _resolve_source(rec: Dict[str, Any], source_key: str) -> Tuple[str, bool]:
    """返回 (源标签, 是否混合源)。

    规则：主键取 source_key，子源取 source / sub_source / source_dataset。
    主键 == 'OmniMedVQA' 且没有子源时，标签为 'OmniMedVQA/unknown' 并标记 mixed。
    """
    base = rec.get(source_key) or rec.get("dataset") or "unknown"
    base = str(base)
    sub = None
    for k in ("source", "sub_source", "source_dataset", "sub_dataset"):
        v = rec.get(k)
        if v:
            sub = str(v)
            break
    if sub and sub != base:
        return f"{base}/{sub}", False
    if base == "OmniMedVQA":
        # 聚合集不能出单个数字：没有子源就单独成未知桶，并禁止进 overall
        return f"{base}/unknown", True
    return base, False


def _record_correct(rec: Dict[str, Any]) -> Optional[bool]:
    if "correct" in rec and rec["correct"] is not None:
        return bool(rec["correct"])
    if "is_correct" in rec and rec["is_correct"] is not None:
        return bool(rec["is_correct"])
    return None


def per_source_report(records: Sequence[Dict[str, Any]], source_key: str = "dataset",
                      n_boot: int = 2000) -> Dict[str, Any]:
    """按源分项统计。OmniMedVQA 必须拆到子源，**禁止报混合均值**。

    返回
      ``{source_key, n, per_source: {label: {n, k, acc, wilson_lo/hi, boot_lo/hi, mixed}},
         overall: {...}, mixed_excluded: [...], warnings: [...]}``

    overall 只聚合非混合桶；被剔除的桶列在 ``mixed_excluded`` 里，避免有人直接
    拿 overall 当"医学平均"。
    """
    groups: Dict[str, List[float]] = {}
    mixed: Dict[str, bool] = {}
    n_bad = 0
    for rec in records:
        label, is_mixed = _resolve_source(rec, source_key)
        c = _record_correct(rec)
        if c is None:
            n_bad += 1
            continue
        groups.setdefault(label, []).append(1.0 if c else 0.0)
        mixed[label] = mixed.get(label, False) or is_mixed

    per_source: Dict[str, Dict[str, Any]] = {}
    clean: List[float] = []
    excluded: List[str] = []
    for label, flags in groups.items():
        entry = stats.summarize_acc([bool(x) for x in flags], n_boot=n_boot)
        entry["mixed"] = bool(mixed.get(label, False))
        per_source[label] = entry
        if entry["mixed"]:
            excluded.append(label)
        else:
            clean.extend(flags)

    warnings: List[str] = []
    if excluded:
        warnings.append(
            "以下桶缺少子源、被判定为混合源，已从 overall 中剔除（禁止报混合均值）: "
            + ", ".join(sorted(excluded))
        )
    if n_bad:
        warnings.append(f"{n_bad} 条记录缺少 correct 字段，已跳过")

    overall = stats.summarize_acc([bool(x) for x in clean], n_boot=n_boot) if clean else {
        "n": 0, "k": 0, "acc": float("nan"), "wilson_lo": float("nan"),
        "wilson_hi": float("nan"), "boot_lo": float("nan"), "boot_hi": float("nan"),
    }

    return {
        "source_key": source_key,
        "n": sum(len(v) for v in groups.values()),
        "n_sources": len(per_source),
        "per_source": per_source,
        "overall": overall,
        "mixed_excluded": excluded,
        "warnings": warnings,
    }


# ============================================================ 效率指标


def _record_calls(rec: Dict[str, Any]) -> int:
    """取调用次数：n_calls > calls 列表 > steps 里的工具轮。"""
    if "n_calls" in rec and rec["n_calls"] is not None:
        return int(rec["n_calls"])
    calls = rec.get("calls")
    if isinstance(calls, (list, tuple)):
        return len(calls)
    steps = rec.get("steps")
    if isinstance(steps, (list, tuple)):
        return sum(1 for s in steps if isinstance(s, dict) and s.get("call"))
    return 0


def call_rate(records: Sequence[Dict[str, Any]]) -> float:
    """至少发起过一次工具调用的记录比例。"""
    if not records:
        return float("nan")
    n = sum(1 for r in records if _record_calls(r) > 0)
    return n / len(records)


def avg_tokens(records: Sequence[Dict[str, Any]], key: str = "tokens") -> float:
    """平均 token 数（默认 total tokens；可改 'visual_tokens'）。缺字段的记录不计入。"""
    vals = [float(r[key]) for r in records
            if key in r and r[key] is not None]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def _record_tool_oks(rec: Dict[str, Any]) -> List[bool]:
    oks = rec.get("tool_oks")
    if isinstance(oks, (list, tuple)):
        return [bool(x) for x in oks]
    if "tool_ok" in rec and rec["tool_ok"] is not None:
        # 记录级标记：按调用次数展开，避免"一次成功一次失败"被重复计数
        return [bool(rec["tool_ok"])] * max(_record_calls(rec), 1)
    return []


def tool_success_rate(records: Sequence[Dict[str, Any]]) -> float:
    """工具调用成功率（按**每次调用**计）。没有任何调用时返回 nan，而不是 0。"""
    oks: List[bool] = []
    for r in records:
        if _record_calls(r) > 0:
            oks.extend(_record_tool_oks(r))
    if not oks:
        return float("nan")
    return sum(1 for x in oks if x) / len(oks)


def efficiency_report(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """三个效率指标打包：调用率 / 平均 token / 工具成功率（PROPOSAL §8.1.3）。"""
    return {
        "n": len(records),
        "call_rate": call_rate(records),
        "avg_tokens": avg_tokens(records),
        "avg_calls": (sum(_record_calls(r) for r in records) / len(records))
        if records else float("nan"),
        "tool_success_rate": tool_success_rate(records),
    }


# ============================================================ oracle 上界


def _cond_of(rec: Dict[str, Any]) -> str:
    c = rec.get("condition") or rec.get("view") or ""
    return str(c).strip().lower()


def oracle_crop_ceiling(records: Sequence[Dict[str, Any]],
                        oracle_records: Optional[Sequence[Dict[str, Any]]] = None,
                        n_boot: int = 2000, seed: int = 42) -> Dict[str, Any]:
    """比较 full-view 与 oracle-crop 两个条件的准确率（SLAKE 81.97%/+9.99 的口径）。

    两种用法：
      1. ``records`` 里带 ``condition``（'full' / 'oracle_crop'），按字段分组；
      2. 显式传 ``oracle_records``，与 records 分开给。

    有 ``id`` 时按 id 求交集做**配对** bootstrap；无 id 且长度相等时按位置配对。
    差值方向固定为 ``oracle − full``：这是"看到金标框能涨多少"的上界估计。
    """
    if oracle_records is None:
        full_recs: List[Dict[str, Any]] = []
        orc_recs: List[Dict[str, Any]] = []
        unk = 0
        for r in records:
            c = _cond_of(r)
            if c in ("full", "full_view", "fullview", "full-view"):
                full_recs.append(r)
            elif c in ("oracle_crop", "oracle", "gold_crop", "oracle-crop"):
                orc_recs.append(r)
            else:
                unk += 1
    else:
        full_recs = list(records)
        orc_recs = list(oracle_records)
        unk = 0

    def _flags(rs: Sequence[Dict[str, Any]]) -> Tuple[List[float], List[Any]]:
        fs: List[float] = []
        ids: List[Any] = []
        for r in rs:
            c = _record_correct(r)
            if c is None:
                continue
            fs.append(1.0 if c else 0.0)
            ids.append(r.get("id"))
        return fs, ids

    f_full, id_full = _flags(full_recs)
    f_orc, id_orc = _flags(orc_recs)

    def _block(fs: List[float]) -> Dict[str, Any]:
        n = len(fs)
        k = int(sum(fs))
        lo, hi = stats.wilson_ci(k, n)
        est, blo, bhi = stats.bootstrap_ci(fs, n_boot=n_boot, seed=int(seed))
        return {"n": n, "k": k, "acc": k / n if n else float("nan"),
                "wilson_lo": lo, "wilson_hi": hi, "boot_lo": blo, "boot_hi": bhi}

    # 配对：优先按 id 交集，否则等长时按位置
    paired_a: List[float] = []
    paired_b: List[float] = []
    n_paired = 0
    if id_full and id_orc and all(x is not None for x in id_full + id_orc):
        map_full = {i: v for i, v in zip(id_full, f_full)}
        for i, v in zip(id_orc, f_orc):
            if i in map_full:
                paired_a.append(map_full[i])
                paired_b.append(v)
        n_paired = len(paired_a)
    elif len(f_full) == len(f_orc) and f_full:
        # 位置配对只在两边等长时成立；否则宁可给 None 也不给假的配对 CI
        paired_a, paired_b = f_full, f_orc
        n_paired = len(f_full)

    if n_paired >= 2:
        est, lo, hi = stats.paired_bootstrap(paired_b, paired_a, n_boot=n_boot, seed=int(seed))
        delta = sum(paired_b) / n_paired - sum(paired_a) / n_paired
        delta_ci: Optional[Tuple[float, float, float]] = (est, lo, hi)
    else:
        delta = (_block(f_orc)["acc"] - _block(f_full)["acc"])
        delta_ci = None

    return {
        "full": _block(f_full),
        "oracle_crop": _block(f_orc),
        "delta": delta,               # oracle − full，正数表示金标框带来增益
        "delta_paired_ci": delta_ci,  # (est, lo, hi)；无法配对时为 None
        "n_paired": n_paired,
        "n_unlabeled_condition": unk,
        "note": "delta 为 oracle_crop − full；无法配对时 delta_paired_ci 为 None（不编造区间）",
    }


# ============================================================ 便捷：定位 IoU


def mean_iou(records: Sequence[Dict[str, Any]], gold_key: str = "box",
             pred_key: str = "pred_box") -> float:
    """A 层定位质量：预测框与 gold 框的平均 IoU。缺框的记录跳过。"""
    vals: List[float] = []
    for r in records:
        g, p = r.get(gold_key), r.get(pred_key)
        if not g or not p:
            continue
        vals.append(iou(Box.from_json(g), Box.from_json(p)))
    return float(sum(vals) / len(vals)) if vals else float("nan")
