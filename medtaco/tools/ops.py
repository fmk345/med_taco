"""图像工具原语：crop / zoom / window / measure / load_image，以及执行入口 execute。

设计要点（与 PROPOSAL §5.3 / §7.5 对齐）：
  - 所有工具只处理像素，不做任何"自述结果"——返回值就是真实产物；
  - execute 负责把返回图**落盘**，只把路径放进 ToolResult.image，避免内存里堆大数组；
  - 缺信息（CT 无 rescale、图无 PixelSpacing）一律 raise ToolUnavailable，
    由 execute 转成 ok=False，绝不静默编造读数。

依赖：numpy + PIL（不引入新依赖）。
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..utils.schema import Box, ToolCall, ToolResult

__all__ = [
    "ToolUnavailable",
    "ToolContext",
    "load_image",
    "crop",
    "zoom",
    "window",
    "measure",
    "execute",
    "parse_box",
    "crop_with_box",
    "zoom_with_scale",
]

# 单次 zoom 允许的最大上采样倍率。
# 超过 4x 纯属插值，训练出来的是"假细节"（PROPOSAL §3.2 分辨率审计同源考虑）。
MAX_UPSCALE: float = 4.0

# 退化框阈值：宽或高小于该值（像素）视为无效观测
MIN_BOX_SIDE: int = 8


class ToolUnavailable(Exception):
    """工具在当前数据上不可用（缺 spacing / 缺 HU 元信息等），不是代码 bug。"""


# ------------------------------------------------------------------ 基础工具


def _as_uint8(img: np.ndarray) -> np.ndarray:
    """把任意通道数/任意 dtype 的图规整成 HxWx3 uint8。

    uint16 按 /257 压到 8bit（仅用于展示，**不要**用它做 HU 还原——
    window() 走的是原始像素通道）。
    """
    a = np.asarray(img)
    if a.ndim == 2:
        a = a[:, :, None]
    if a.ndim != 3:
        raise ValueError(f"图像维度非法: shape={a.shape}")
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    elif a.shape[2] == 4:
        a = a[:, :, :3]
    elif a.shape[2] != 3:
        raise ValueError(f"图像通道数非法: shape={a.shape}")

    if a.dtype == np.uint8:
        return np.ascontiguousarray(a)
    if a.dtype == np.bool_:
        return (a.astype(np.uint8) * 255)
    if np.issubdtype(a.dtype, np.floating):
        f = np.nan_to_num(a.astype(np.float32))
        if f.size and float(f.max()) <= 1.0 + 1e-6:
            f = f * 255.0          # 约定 0-1 浮点
        return np.clip(f, 0, 255).astype(np.uint8)
    if a.dtype == np.uint16:
        return np.clip(a.astype(np.float32) / 257.0, 0, 255).astype(np.uint8)
    if a.dtype == np.int16:        # CT 原始像素常见 int16，先平移到非负仅用于展示
        return np.clip(a.astype(np.float32) + 32768.0, 0, 255).astype(np.uint8)
    return np.clip(a.astype(np.float32), 0, 255).astype(np.uint8)


def _resize(arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """PIL 双线性重采样，(w, h) 顺序。"""
    im = Image.fromarray(np.ascontiguousarray(arr))
    return np.asarray(im.resize((int(size[0]), int(size[1])), Image.BILINEAR))


def _cap_long_side(arr: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    """把长边压到 max_side 以内（观测上下文防爆），返回 (图, 施加的缩放比)。"""
    if not max_side or max_side <= 0:
        return arr, 1.0
    h, w = arr.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return arr, 1.0
    s = max_side / float(long_side)
    nw = max(1, int(round(w * s)))
    nh = max(1, int(round(h * s)))
    return _resize(arr, (nw, nh)), s


def _expand(box: Box, pad_ratio: float) -> Box:
    """按框自身宽高等比例外扩 pad_ratio（0.1 = 四边各扩 10%）。"""
    if not pad_ratio or pad_ratio <= 0:
        return box
    dx = box.w * float(pad_ratio)
    dy = box.h * float(pad_ratio)
    return Box(box.x0 - dx, box.y0 - dy, box.x1 + dx, box.y1 + dy, box.label)


def _to_int_box(box: Box, w: int, h: int) -> Tuple[int, int, int, int]:
    """浮点框 -> 整像素切片区间：向外取整，避免丢边，再 clip 到图内。"""
    x0 = int(math.floor(max(0.0, min(float(box.x0), float(w)))))
    y0 = int(math.floor(max(0.0, min(float(box.y0), float(h)))))
    x1 = int(math.ceil(max(0.0, min(float(box.x1), float(w)))))
    y1 = int(math.ceil(max(0.0, min(float(box.y1), float(h)))))
    return x0, y0, x1, y1


def parse_box(v: Any) -> Box:
    """把 [x0,y0,x1,y1(,label)] / dict / Box 统一成 Box。"""
    if isinstance(v, Box):
        return v
    if isinstance(v, dict):
        return Box(
            float(v["x0"]), float(v["y0"]), float(v["x1"]), float(v["y1"]), str(v.get("label", ""))
        )
    if isinstance(v, (list, tuple)) and len(v) >= 4:
        label = str(v[4]) if len(v) > 4 else ""
        return Box(float(v[0]), float(v[1]), float(v[2]), float(v[3]), label)
    raise ValueError(f"box 需要 [x0,y0,x1,y1]，收到: {v!r}")


# ------------------------------------------------------------------ 图像原语


def load_image(path: str) -> np.ndarray:
    """读图 -> HxWx3 uint8（灰度图转 RGB 三通道复制）。"""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def crop_with_box(img: np.ndarray, box: Box, pad_ratio: float = 0.0) -> Tuple[np.ndarray, Box]:
    """裁剪并返回 (子图, 实际生效的整像素框)。

    pad_ratio 先外扩，再 clip 到图像边界；完全落在图外时 raise ValueError。
    """
    arr = _as_uint8(img)
    h, w = arr.shape[:2]
    b = _expand(box, pad_ratio).clip(w, h)
    x0, y0, x1, y1 = _to_int_box(b, w, h)
    if x1 - x0 < 1 or y1 - y0 < 1:
        raise ValueError(f"裁剪区为空: box={box.as_tuple()}")
    return np.ascontiguousarray(arr[y0:y1, x0:x1]), Box(x0, y0, x1, y1, box.label)


def crop(img: np.ndarray, box: Box, pad_ratio: float = 0.0) -> np.ndarray:
    """按像素坐标裁剪（可外扩），越界自动 clip。"""
    return crop_with_box(img, box, pad_ratio)[0]


def zoom_with_scale(
    img: np.ndarray,
    box: Box,
    out_side: int = 1024,
    max_upscale: float = MAX_UPSCALE,
) -> Tuple[np.ndarray, Box, float]:
    """裁剪后重采样到 out_side 长边，返回 (图, 实际框, 实际倍率)。

    语义：
      - 裁剪区长边 >= out_side：只做缩放（scale < 1），绝不放大；
      - 裁剪区长边 <  out_side：放大，但倍率不超过 max_upscale
        （超出部分不会凭空产生信息，宁可留白不插值）；
      - 输出长边恒 <= out_side，倍率上限保证"放大的是真实像素而非幻觉"。
    """
    if out_side is None or out_side < 1:
        raise ValueError(f"out_side 非法: {out_side!r}")
    sub, eff_box = crop_with_box(img, box, 0.0)
    ch, cw = sub.shape[:2]
    long_side = max(cw, ch)
    if long_side < 1:
        raise ValueError("zoom 裁剪区为空")

    scale = float(out_side) / float(long_side)
    if max_upscale is not None and max_upscale > 0:
        scale = min(scale, float(max_upscale))

    nw = max(1, int(round(cw * scale)))
    nh = max(1, int(round(ch * scale)))
    # 双重保险：整数取整后长边也不许越过 out_side
    if max(nw, nh) > out_side:
        shrink = out_side / float(max(nw, nh))
        nw = max(1, int(round(nw * shrink)))
        nh = max(1, int(round(nh * shrink)))
    actual_scale = max(nw / float(cw), nh / float(ch))

    if (nw, nh) == (cw, ch):
        return sub, eff_box, actual_scale
    return _resize(sub, (nw, nh)), eff_box, float(actual_scale)


def zoom(img: np.ndarray, box: Box, out_side: int = 1024) -> np.ndarray:
    """裁剪 + 上采样到 out_side 长边（受 MAX_UPSCALE 限制，不超原图像素信息）。"""
    return zoom_with_scale(img, box, out_side)[0]


def _raw_channel(img: np.ndarray) -> np.ndarray:
    """取单通道**原始像素值**（不做 8bit 归一），供 HU 还原使用。"""
    a = np.asarray(img)
    if a.ndim == 3:
        a = a[:, :, 0]
    if a.ndim != 2:
        raise ValueError(f"window 需要 2D 或 3D 图，收到 shape={a.shape}")
    return a.astype(np.float64)


def window(img: np.ndarray, center: float, width: float, meta: dict) -> np.ndarray:
    """CT 窗宽窗位：先把像素还原成 HU，再线性映射到 0-255 uint8。

    HU = pixel * rescale_slope + rescale_intercept（DICOM 标准）；
    映射区间 [center - width/2, center + width/2]，线性截断。
    缺 rescale 元信息、或 photometric 不是 MONOCHROME2 时 raise ToolUnavailable。
    """
    meta = dict(meta or {})
    slope = meta.get("rescale_slope")
    intercept = meta.get("rescale_intercept")
    if slope is None or intercept is None:
        raise ToolUnavailable("缺少 rescale_slope/rescale_intercept，无法还原 HU（该数据不是 CT）")

    photometric = str(meta.get("photometric", "MONOCHROME2")).upper()
    if photometric and photometric != "MONOCHROME2":
        # MONOCHROME1 是反相显示，不能直接套窗；其它色彩空间更没有 HU
        raise ToolUnavailable(f"photometric={photometric} 不支持 HU 窗宽窗位（只支持 MONOCHROME2）")

    if width is None or float(width) <= 0:
        raise ToolUnavailable(f"窗宽非法: {width!r}")
    if center is None:
        raise ToolUnavailable("窗位为空")

    raw = _raw_channel(img)
    hu = raw * float(slope) + float(intercept)

    lo = float(center) - float(width) / 2.0
    hi = float(center) + float(width) / 2.0
    if hi <= lo:  # width 已在上面保证 > 0，这里只是数值兜底
        raise ToolUnavailable(f"窗区间非法: [{lo}, {hi}]")

    out = (hu - lo) / (hi - lo) * 255.0
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def measure(box: Box, meta: dict, mode: str = "long") -> float:
    """按 PixelSpacing 量框：long/short 返回 mm，area 返回 mm^2。

    meta['pixel_spacing'] = (sx, sy)，单位 mm/px（x 方向在前）。
    """
    meta = dict(meta or {})
    spacing = meta.get("pixel_spacing")
    if spacing is None:
        raise ToolUnavailable("缺少 pixel_spacing，无法换算物理尺寸（mm）")
    seq = list(spacing)
    if len(seq) != 2 or seq[0] is None or seq[1] is None:
        raise ToolUnavailable(f"pixel_spacing 非法: {spacing!r}")
    sx, sy = float(seq[0]), float(seq[1])
    if sx <= 0 or sy <= 0:
        raise ToolUnavailable(f"pixel_spacing 必须为正: {spacing!r}")

    mode = (mode or "long").lower()
    w_mm = box.w * sx
    h_mm = box.h * sy
    if mode == "long":
        return float(max(w_mm, h_mm))
    if mode == "short":
        return float(min(w_mm, h_mm))
    if mode == "area":
        return float(w_mm * h_mm)
    raise ValueError(f"未知 measure mode: {mode!r}（可用 long/short/area）")


# ------------------------------------------------------------------ 执行入口


@dataclass
class ToolContext:
    """一次样本的工具执行环境：原图 + 元信息 + 落盘目录。"""

    image: Optional[np.ndarray]
    meta: Dict[str, Any] = field(default_factory=dict)
    out_dir: str = "."
    obs_max_side: int = 1024
    jpeg_quality: int = 90
    counter: int = 0  # 文件名计数器，保证同一 ctx 内不重名


_TOOL_NAMES = ("crop", "zoom", "window", "measure", "load_image")


def _save(arr: np.ndarray, out_dir: str, name: str, counter: int, quality: int) -> str:
    """落盘并返回路径。quality <= 0 或 None 时存无损 PNG，否则 JPEG。"""
    os.makedirs(out_dir, exist_ok=True)
    lossless = quality is None or int(quality) <= 0 or arr.dtype != np.uint8
    ext = ".png" if lossless else ".jpg"
    path = os.path.join(out_dir, f"{name}_{int(counter):05d}{ext}")
    im = Image.fromarray(np.ascontiguousarray(arr))
    if lossless:
        im.save(path)
    else:
        im.save(path, quality=int(quality), subsampling=0)
    return path


def _require_image(ctx: ToolContext) -> np.ndarray:
    if ctx.image is None:
        raise ValueError("ToolContext.image 为空，请先 load_image")
    return np.asarray(ctx.image)


def execute(call: ToolCall, ctx: ToolContext) -> ToolResult:
    """按 call.name 分发，参数从 call.args 取，返回图写盘到 ctx.out_dir。

    约定：
      - 越界框一律 clip；clip 后宽或高 < MIN_BOX_SIDE 判退化，返回 ok=False；
      - 缺元信息（spacing / HU）走 ToolUnavailable -> ok=False + error；
      - ToolResult.meta['crop_box'] 记录实际生效的整像素框（crop / zoom 才有）。
    """
    t0 = time.perf_counter()
    name = (call.name or "").strip().lower()
    args: Dict[str, Any] = dict(call.args or {})

    try:
        if name not in _TOOL_NAMES:
            return ToolResult(ok=False, error=f"未知工具: {call.name!r}")

        # ---------------- load_image：换图，不作为观测返回给模型
        if name == "load_image":
            src = args.get("path") or args.get("image") or args.get("file")
            if not src:
                return ToolResult(ok=False, error="load_image 需要参数 path")
            arr = load_image(str(src))
            ctx.image = arr
            out = _save(arr, ctx.out_dir, name, ctx.counter, ctx.jpeg_quality)
            ctx.counter += 1
            return ToolResult(
                ok=True,
                image=out,
                meta={"src": str(src), "out_size": [int(arr.shape[1]), int(arr.shape[0])],
                      "elapsed_s": time.perf_counter() - t0},
            )

        img = _require_image(ctx)
        h, w = img.shape[:2]

        # ---------------- measure：纯数值，无图
        if name == "measure":
            if "box" not in args:
                return ToolResult(ok=False, error="measure 需要参数 box")
            box = parse_box(args["box"])
            mode = str(args.get("mode", "long")).lower()
            val = measure(box, ctx.meta, mode)
            unit = "mm^2" if mode == "area" else "mm"
            return ToolResult(
                ok=True,
                text=f"{val:.2f} {unit}",
                meta={
                    "value": val,
                    "unit": unit,
                    "mode": mode,
                    "spacing": list(ctx.meta.get("pixel_spacing")),  # type: ignore[arg-type]
                    "elapsed_s": time.perf_counter() - t0,
                },
            )

        # ---------------- window：整幅 CT 图
        if name == "window":
            center = args.get("center", args.get("window_center"))
            width = args.get("width", args.get("window_width"))
            if center is None or width is None:
                return ToolResult(ok=False, error="window 需要参数 center 与 width")
            out = window(img, float(center), float(width), ctx.meta)
            out, s = _cap_long_side(out, ctx.obs_max_side)
            path = _save(out, ctx.out_dir, name, ctx.counter, ctx.jpeg_quality)
            ctx.counter += 1
            return ToolResult(
                ok=True,
                image=path,
                meta={
                    "center": float(center),
                    "width": float(width),
                    "out_size": [int(out.shape[1]), int(out.shape[0])],
                    "obs_scale": float(s),
                    "elapsed_s": time.perf_counter() - t0,
                },
            )

        # ---------------- crop / zoom：都需要框
        if "box" not in args:
            return ToolResult(ok=False, error=f"{name} 需要参数 box")
        raw_box = parse_box(args["box"])
        clipped = raw_box.clip(w, h)
        if clipped.w < MIN_BOX_SIDE or clipped.h < MIN_BOX_SIDE:
            return ToolResult(
                ok=False,
                error=(f"退化框: clip 后 {clipped.w:.1f}x{clipped.h:.1f}px，"
                       f"宽高均需 >= {MIN_BOX_SIDE}px（原始框 {raw_box.as_tuple()}，图 {w}x{h}）"),
                meta={"crop_box": [clipped.x0, clipped.y0, clipped.x1, clipped.y1],
                      "reason": "degenerate_box",
                      "elapsed_s": time.perf_counter() - t0},
            )

        result_meta: Dict[str, Any] = {}
        if name == "crop":
            pad = float(args.get("pad_ratio", 0.0) or 0.0)
            out, eff = crop_with_box(img, clipped, pad)
            result_meta = {"pad_ratio": pad}
        else:  # zoom
            out_side = int(args.get("out_side", ctx.obs_max_side) or ctx.obs_max_side)
            max_up = args.get("max_upscale", MAX_UPSCALE)
            out, eff, scale = zoom_with_scale(
                img, clipped, out_side, float(max_up) if max_up is not None else MAX_UPSCALE
            )
            result_meta = {"out_side": out_side, "actual_scale": float(scale)}

        out, s = _cap_long_side(out, ctx.obs_max_side)
        path = _save(out, ctx.out_dir, name, ctx.counter, ctx.jpeg_quality)
        ctx.counter += 1
        result_meta.update(
            {
                "crop_box": [eff.x0, eff.y0, eff.x1, eff.y1],
                "src_box": [raw_box.x0, raw_box.y0, raw_box.x1, raw_box.y1],
                "out_size": [int(out.shape[1]), int(out.shape[0])],
                "obs_scale": float(s),
                "elapsed_s": time.perf_counter() - t0,
            }
        )
        return ToolResult(ok=True, image=path, meta=result_meta)

    except ToolUnavailable as e:
        return ToolResult(
            ok=False,
            error=str(e),
            meta={"reason": "tool_unavailable", "elapsed_s": time.perf_counter() - t0},
        )
    except (ValueError, TypeError, OSError) as e:
        return ToolResult(
            ok=False,
            error=f"{type(e).__name__}: {e}",
            meta={"reason": "bad_args", "elapsed_s": time.perf_counter() - t0},
        )
