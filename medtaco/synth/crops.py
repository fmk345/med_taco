"""阶段 2.5：把样本的 gold 框渲染成裁剪图，喂给 teacher 模型。

为什么必须有这一步：``teacher.build_request`` 只从 ``sample.meta['crops']`` 取裁剪图，
没有裁剪图时它会退化成「原图 + 原图」，teacher 只能把三个 step 全填 ``none``——
合成出来的轨迹对训练毫无用处。裁剪图必须由**确定性代码**从像素级标注生成，
不能让 teacher 自己想象一个框（那就是 §7.2 的 answer-first trajectory）。

裁剪策略（与问题生成器对齐）：
  - G2 多尺度：沿 ``meta['scale_boxes']`` 取「最外层 + gold 框」两张，训练"先看大范围再放大"；
  - G1/G5 及默认：``pad_ratio`` 外扩的框 + gold 框两张，第二张是"裁对了之后的确认"；
  - G4 测量：只要 gold 框一张（测量不需要多尺度）。

输出写 ``<processed>/_crops/<sample_id>_<k>.png``，并把路径写回 ``meta['crops']``，
同时记 ``meta['crop_roles']`` 便于复盘。已存在的图不重画（幂等）。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from medtaco.tools.ops import crop_with_box, load_image
from medtaco.utils.schema import Box, Sample

__all__ = ["prepare_crops", "crop_boxes_for", "crops_dir"]

# 外扩比例：0.25 足够把病灶放进解剖上下文，又不至于让问题变成"看全图"
DEFAULT_PAD = 0.25


def crops_dir(cfg: Any) -> str:
    return os.path.join(str(cfg.paths.processed), "_crops")


def _box_from_json(d: Any) -> Optional[Box]:
    """Box.to_json() 是 [x0,y0,x1,y1,label]，也兼容 dict 与 4 元组。"""
    try:
        if isinstance(d, dict):
            return Box(float(d["x0"]), float(d["y0"]), float(d["x1"]), float(d["y1"]),
                       str(d.get("label", "")))
        if isinstance(d, (list, tuple)) and len(d) >= 4:
            label = str(d[4]) if len(d) >= 5 else ""
            return Box(float(d[0]), float(d[1]), float(d[2]), float(d[3]), label)
    except (KeyError, TypeError, ValueError):
        return None
    return None


def crop_boxes_for(sample: Sample, gen: Optional[str] = None) -> List[Tuple[str, Box]]:
    """返回 [(role, box), ...]，role 用于日志与 meta 复盘。

    空列表表示这条样本不该有裁剪（负样本 / 无区域标注）。
    """
    meta = sample.meta or {}
    if meta.get("need_crop") is False:
        return []
    gen = gen or str(meta.get("generator") or "")

    scale = [_box_from_json(d) for d in (meta.get("scale_boxes") or [])]
    scale = [b for b in scale if b is not None]
    if gen == "G2" and len(scale) >= 2:
        # 最外层（大范围）→ 最内层（gold）
        return [("outer", scale[0]), ("inner", scale[-1])]

    gold = _box_from_json(meta.get("anchor_box")) if meta.get("anchor_box") else None
    if gold is None and sample.boxes:
        gold = sample.boxes[0]
    if gold is None:
        return []
    if gen == "G4":
        return [("measure", gold)]
    return [("context", gold), ("tight", gold)]


def prepare_crops(samples: Sequence[Sample], cfg: Any,
                  pad: float = DEFAULT_PAD) -> Dict[str, int]:
    """就地给每个 Sample 写 ``meta['crops']``。返回统计。

    单条失败不抛：记进统计，该样本在 synth 阶段会被 build_request 退化成无裁剪请求，
    由 validate 阶段判定是否可用——比整批中止更可控。
    """
    out_dir = crops_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)
    stats = {"n": 0, "n_with_crops": 0, "n_no_crop": 0, "n_missing_image": 0,
             "n_error": 0, "n_png": 0}

    cache: Dict[str, np.ndarray] = {}
    for s in samples:
        stats["n"] += 1
        plan = crop_boxes_for(s)
        if not plan:
            stats["n_no_crop"] += 1
            s.meta["crops"] = []
            s.meta["crop_roles"] = []
            continue
        if not s.image or not os.path.exists(s.image):
            stats["n_missing_image"] += 1
            continue
        try:
            img = cache.get(s.image)
            if img is None:
                img = load_image(s.image)
                # 只缓存最近一张：同一张图连着的样本通常不多，缓存全部会吃光内存
                cache = {s.image: img}
            entries: List[Dict[str, Any]] = []
            roles: List[str] = []
            for k, (role, box) in enumerate(plan):
                # context/outer 外扩，tight/inner/measure 用原框
                pad_ratio = pad if role in ("context", "outer") else 0.0
                arr, eff = crop_with_box(img, box, pad_ratio=pad_ratio)
                path = os.path.join(out_dir, f"{_safe(s.id)}_{k}.png")
                if not os.path.exists(path):
                    _save_png(arr, path)
                    stats["n_png"] += 1
                entries.append({"path": path, "box": eff.to_json(),
                                "tool": "crop", "role": role})
                roles.append(role)
            s.meta["crops"] = entries
            s.meta["crop_roles"] = roles
            stats["n_with_crops"] += 1
        except Exception as e:  # noqa: BLE001 - 单条坏样本不拖垮整批
            stats["n_error"] += 1
            print(f"[crops] {s.id} 失败: {type(e).__name__}: {e}")
    return stats


def _safe(sid: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(sid))[:120]


def _save_png(arr: np.ndarray, path: str) -> None:
    from PIL import Image

    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype("uint8")
    Image.fromarray(np.ascontiguousarray(a)).save(path)
