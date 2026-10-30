"""图像预处理：统一格式 + sidecar 元信息 + 感知哈希去重 + Sample 构造。

设计要点（PROPOSAL §3.2）：
  1. **保留原始矩阵语义**：16-bit 灰度（CT/CXR DICOM 转出的 png）不做 percentile 拉伸，
     只做线性映射到 8-bit，并把 orig_size/缩放比/spacing/window 全部写进 sidecar——
     一旦丢了 spacing，`measure` 工具在该数据上就永久不可用。
  2. **长边封顶 4096**（cfg.data.resolution.max_side_cap），防止 4K+ 图在批量上传时爆内存；
     只缩不放，scale 一律记录，下游可精确还原坐标。
  3. **去重**用真 pHash（32x32 灰度 + 正交 DCT-II 取低频 8x8，去掉 DC 后按中位数二值化，
     63 bit 有效），汉明距离判重；`tile_hash=True` 时额外对 grid×grid 分块各算一次，
     因为 pHash 对「大图里嵌的小病灶裁剪」是盲的（整图低频几乎不变）。
  4. 随机性：本模块不用随机数；测试里的造图由调用方显式 seed。

不引入新依赖：只用 numpy / PIL；torch 不需要。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from medtaco.utils.config import Cfg
from medtaco.utils.io import atomic_write_json
from medtaco.utils.schema import Box, Sample

# pHash 常数
_PHASH_SIDE = 32          # 统一缩到 32x32 灰度再做 DCT
_PHASH_LOW = 8            # 取左上 8x8 低频，去掉 DC 后剩 63 bit
_HASH_BITS = _PHASH_LOW * _PHASH_LOW

# 分块哈希只对有纹理的块生效：std 低于此值的块（大片纯黑/纯白肺野）直接丢弃
_TILE_MIN_STD = 2.0

_RESAMPLE = Image.LANCZOS


# ---------------------------------------------------------------- 配置读取


def _get(cfg: Any, path: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Cfg):
        return cfg.get(path, default)
    cur: Any = cfg
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return default if cur is None else cur


def _log(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)


# ---------------------------------------------------------------- 预处理


def preprocess_image(src: str, dst: str, cfg: Any) -> dict:
    """把一张图统一成 RGB 8-bit 落盘，并写 sidecar json。

    参数：
        src: 源图路径（jpg/png/tif，DICOM 不在本函数职责内）
        dst: 目标路径，扩展名决定编码（.png 无损 / .jpg|.jpeg 质量取 cfg.data.jpeg_quality）
        cfg: 用于 max_side_cap 等；可为 None（用默认值）
    返回：sidecar 字典（同时以原子写落盘到 `<dst 去扩展名>.json`）
    """
    max_side = int(_get(cfg, "data.resolution.max_side_cap", 4096) or 4096)
    jpeg_q = int(_get(cfg, "data.jpeg_quality", 92) or 92)
    ext = os.path.splitext(dst)[1].lower()
    if ext not in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        raise ValueError(f"不支持的目标格式: {dst}")

    with Image.open(src) as im:
        mode_in = im.mode
        w0, h0 = im.size
        arr, mode_note = _to_rgb8(im)
    img = Image.fromarray(arr, mode="RGB")

    long_side = max(w0, h0)
    scale = 1.0
    if max_side > 0 and long_side > max_side:
        scale = max_side / float(long_side)
        nw = max(1, int(round(w0 * scale)))
        nh = max(1, int(round(h0 * scale)))
        img = img.resize((nw, nh), _RESAMPLE)

    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    if ext in (".jpg", ".jpeg"):
        img.save(dst, format="JPEG", quality=jpeg_q, subsampling=0)
    elif ext in (".tif", ".tiff"):
        img.save(dst, format="TIFF", compression="tiff_deflate")
    else:
        img.save(dst, format="PNG", optimize=False)

    src_meta = _load_src_meta(src)
    side = {
        "image": os.path.basename(dst),
        "src": os.path.abspath(src),
        "orig_size": [w0, h0],
        "size": [img.size[0], img.size[1]],
        "scale": round(scale, 8),
        "scaled": bool(scale != 1.0),
        "mode_in": mode_in,
        "mode_conversion": mode_note,
        "format": ext.lstrip("."),
    }
    side.update(src_meta)
    atomic_write_json(sidecar_path(dst), side)
    return side


def sidecar_path(image_path: str) -> str:
    """旁挂 json 路径：`a/b.png` -> `a/b.json`。"""
    return os.path.splitext(image_path)[0] + ".json"


def _to_rgb8(im: Image.Image) -> Tuple[np.ndarray, str]:
    """任意 PIL 模式 → HxWx3 uint8。16-bit 用线性映射（不做 percentile 拉伸）。"""
    mode = im.mode
    if mode in ("RGB",):
        return np.asarray(im, dtype=np.uint8), "none"
    if mode in ("L", "LA"):
        g = np.asarray(im.convert("L"), dtype=np.uint8)
        return np.stack([g, g, g], axis=-1), "gray->rgb"
    if mode in ("RGBA", "P", "CMYK", "YCbCr", "LAB", "HSV"):
        return np.asarray(im.convert("RGB"), dtype=np.uint8), f"{mode}->rgb"
    if mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        a = np.asarray(im).astype(np.float64)
        lo, hi = float(a.min()), float(a.max())
        if hi <= lo:
            g = np.zeros(a.shape, dtype=np.uint8)
        else:
            g = np.clip((a - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
        return np.stack([g, g, g], axis=-1), f"{mode}->rgb(linear)"
    # 兜底：交给 PIL
    return np.asarray(im.convert("RGB"), dtype=np.uint8), f"{mode}->rgb(fallback)"


def _load_src_meta(src: str) -> Dict[str, Any]:
    """找源图旁边的 sidecar / DICOM 导出 json，抽出 spacing / window / rescale 字段。

    兼容两种键名风格：DICOM 原生（PixelSpacing / WindowCenter）与项目 snake_case。
    一个都没有就返回空 dict（下游 `measure`/`window` 会按 ToolUnavailable 处理）。
    """
    out: Dict[str, Any] = {}
    cand = [sidecar_path(src)]
    stem = os.path.splitext(os.path.basename(src))[0]
    d = os.path.dirname(os.path.abspath(src))
    cand += [os.path.join(d, stem + ".dicom.json"), os.path.join(d, stem + "_meta.json")]
    data: Optional[dict] = None
    for p in cand:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    v = json.load(f)
                if isinstance(v, dict):
                    data = v
                    out["meta_src"] = p
                    break
            except (OSError, json.JSONDecodeError):
                continue
    if data is None:
        out["spacing"] = None
        out["window"] = None
        out["has_hu"] = False
        return out

    spacing = data.get("pixel_spacing") or data.get("PixelSpacing") or data.get("spacing")
    if isinstance(spacing, (list, tuple)) and len(spacing) == 2:
        out["spacing"] = [float(spacing[0]), float(spacing[1])]
    else:
        out["spacing"] = None

    # 记录 spacing 的方向：DICOM PixelSpacing 是 (row=y, col=x)，但很多导出直接写 (x,y)。
    out["spacing_order"] = str(data.get("spacing_order", "xy"))

    wc = data.get("window_center", data.get("WindowCenter"))
    ww = data.get("window_width", data.get("WindowWidth"))
    if isinstance(wc, (list, tuple)):
        wc = wc[0] if wc else None
    if isinstance(ww, (list, tuple)):
        ww = ww[0] if ww else None
    out["window"] = None if wc is None or ww is None else {"center": float(wc), "width": float(ww)}

    slope = data.get("rescale_slope", data.get("RescaleSlope"))
    inter = data.get("rescale_intercept", data.get("RescaleIntercept"))
    out["rescale"] = {
        "slope": float(slope) if slope is not None else None,
        "intercept": float(inter) if inter is not None else None,
    }
    out["photometric"] = str(data.get("photometric", data.get("PhotometricInterpretation", "")))
    out["has_hu"] = bool(
        data.get("photometric", data.get("PhotometricInterpretation")) == "MONOCHROME2"
        and slope is not None and inter is not None
    )
    # orig_size 以本函数实测的原图尺寸为准：源 sidecar 里的 orig_size 若是上一轮
    # 预处理产物（已缩过），照抄会把「原图尺寸」记错，直接污染 resolution 审计。
    out.pop("orig_size", None)
    return out


# ---------------------------------------------------------------- pHash


def _dct_matrix(n: int) -> np.ndarray:
    """正交 DCT-II 矩阵（n×n）：C[k,i] = a_k cos(pi k (2i+1)/2n)，a_0=sqrt(1/n)。"""
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    C = np.cos(np.pi * k * (2 * i + 1) / (2.0 * n))
    C *= math.sqrt(2.0 / n)
    C[0, :] = math.sqrt(1.0 / n)
    return C


_DCT32 = _dct_matrix(_PHASH_SIDE)


def _gray32(path: str) -> np.ndarray:
    """读图 → 32x32 float32 灰度。用 PIL 的 LANCZOS，避免 numpy 手写重采样。"""
    with Image.open(path) as im:
        g = im.convert("L").resize((_PHASH_SIDE, _PHASH_SIDE), _RESAMPLE)
        return np.asarray(g, dtype=np.float32)


def _bits_from_gray32(g: np.ndarray) -> int:
    """32x32 灰度 → 64-bit 整数哈希（低 63 位有效，最高位恒为 0）。"""
    D = _DCT32 @ g.astype(np.float64) @ _DCT32.T
    low = D[:_PHASH_LOW, :_PHASH_LOW].reshape(-1)      # 64 个系数，low[0] 是 DC
    coeffs = low[1:]                                   # 去掉 DC，剩 63
    med = float(np.median(coeffs))
    bits = 0
    for i, v in enumerate(coeffs):
        if v > med:
            bits |= 1 << i
    return bits


def phash(path: str) -> int:
    """单图 pHash。"""
    return _bits_from_gray32(_gray32(path))


def hamming_distance(a: int, b: int) -> int:
    """64-bit 汉明距离。"""
    return bin((a ^ b) & ((1 << 64) - 1)).count("1")


def _tile_hashes(path: str, grid: int, min_tile_side: int) -> List[Tuple[int, int]]:
    """分块 pHash：返回 [(flat_idx, bits), ...]，跳过边长不足与近乎无纹理的块。"""
    with Image.open(path) as im:
        g = im.convert("L")
        w, h = g.size
        tw, th = w // grid, h // grid
        if tw < min_tile_side or th < min_tile_side:
            return []
        arr = np.asarray(g, dtype=np.uint8)
    out: List[Tuple[int, int]] = []
    for r in range(grid):
        for c in range(grid):
            t = arr[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
            if float(t.std()) < _TILE_MIN_STD:
                continue
            small = np.asarray(Image.fromarray(t).resize((_PHASH_SIDE, _PHASH_SIDE), _RESAMPLE),
                               dtype=np.float32)
            out.append((r * grid + c, _bits_from_gray32(small)))
    return out


def _segment_widths(nbits: int, nseg: int) -> List[int]:
    base = nbits // nseg
    rem = nbits % nseg
    return [base + (1 if i < rem else 0) for i in range(nseg)]


def _seg_values(bits: int, widths: Sequence[int]) -> List[int]:
    """按段切片出整数（多索引哈希：汉明距离 ≤ nseg-1 时必有一段完全相同）。"""
    vals: List[int] = []
    off = 0
    for w in widths:
        vals.append((bits >> off) & ((1 << w) - 1))
        off += w
    return vals


def phash_dedup(paths: Iterable[str], hamming: int = 6, tile_hash: bool = False,
                grid: int = 3, min_tile_side: int = 256) -> Dict[str, str]:
    """感知哈希去重，返回 `{重复图: 保留图}`（保留图本身不出现在 key 里）。

    参数：
        hamming:  判重阈值（cfg.data.dedup.phash_hamming，默认 6）
        tile_hash: 是否额外做 grid×grid 分块哈希
        grid:     分块数（cfg.data.dedup.tile_grid）
        min_tile_side: 分块边长下限，小于此值直接放弃分块（cfg.data.dedup.min_tile_side）

    实现：多索引哈希——把 64 bit 切成 hamming+1 段，两图距离 ≤hamming 则至少一段相同，
    所以只需在同一段同值的桶里比对，避免 O(n²) 全比对；分块哈希对每个块位置各建一套桶。
    `hamming >= 63` 或 `paths` 很短时退化为暴力比对。
    """
    seq = [p for p in paths]
    kept: List[str] = []
    dup: Dict[str, str] = {}
    if not seq:
        return dup

    thr = max(0, int(hamming))
    nbits = _HASH_BITS  # 64 槽；最高位恒 0，不影响距离
    nseg = min(nbits, thr + 1)

    # 全图哈希
    full_bits: List[int] = []
    for p in seq:
        try:
            full_bits.append(phash(p))
        except Exception as e:  # noqa: BLE001 - 坏图不参与去重但要留痕
            _log("warn", f"pHash 失败 {p}: {type(e).__name__}: {e}")
            full_bits.append(-1)

    # 分块哈希
    tiles: List[List[Tuple[int, int]]] = []
    if tile_hash:
        for p, b in zip(seq, full_bits):
            if b < 0:
                tiles.append([])
                continue
            try:
                tiles.append(_tile_hashes(p, int(grid), int(min_tile_side)))
            except Exception as e:  # noqa: BLE001
                _log("warn", f"分块哈希失败 {p}: {type(e).__name__}: {e}")
                tiles.append([])
    else:
        tiles = [[] for _ in seq]

    widths = _segment_widths(nbits, nseg)
    brute = thr >= nbits - 1 or len(seq) <= 64

    full_buckets: Dict[Tuple[int, int], List[int]] = {}
    tile_buckets: Dict[Tuple[int, int, int], List[int]] = {}

    def _match_full(i: int) -> Optional[int]:
        """返回与之判重的已保留图下标。"""
        b = full_bits[i]
        if b < 0:
            return None
        if brute:
            for j in _kept_idx:
                if full_bits[j] >= 0 and hamming_distance(b, full_bits[j]) <= thr:
                    return j
            return None
        for s, v in enumerate(_seg_values(b, widths)):
            for j in full_buckets.get((s, v), ()):
                if hamming_distance(b, full_bits[j]) <= thr:
                    return j
        return None

    def _match_tile(i: int) -> Optional[int]:
        """任一分块命中即判重（大图嵌小病灶的场景）。"""
        if brute:
            for j in _kept_idx:
                for fi, bi in tiles[i]:
                    for fj, bj in tiles[j]:
                        if fi == fj and hamming_distance(bi, bj) <= thr:
                            return j
            return None
        for fi, bi in tiles[i]:
            for s, v in enumerate(_seg_values(bi, widths)):
                for j in tile_buckets.get((fi, s, v), ()):
                    for fj, bj in tiles[j]:
                        if fj == fi and hamming_distance(bi, bj) <= thr:
                            return j
        return None

    _kept_idx: List[int] = []
    for i, p in enumerate(seq):
        j = _match_full(i) if full_bits[i] >= 0 else None
        if j is None and tile_hash and tiles[i]:
            j = _match_tile(i)
        if j is not None:
            dup[p] = kept[j]
            continue
        kept.append(p)
        _kept_idx.append(i)
        b = full_bits[i]
        if b >= 0:
            for s, v in enumerate(_seg_values(b, widths)):
                full_buckets.setdefault((s, v), []).append(i)
        for fi, bi in tiles[i]:
            for s, v in enumerate(_seg_values(bi, widths)):
                tile_buckets.setdefault((fi, s, v), []).append(i)
    return dup


def phash_dedup_from_cfg(paths: Iterable[str], cfg: Any) -> Dict[str, str]:
    """按 cfg.data.dedup 调 phash_dedup（辅助函数）。"""
    return phash_dedup(
        paths,
        hamming=int(_get(cfg, "data.dedup.phash_hamming", 6) or 6),
        tile_hash=bool(_get(cfg, "data.dedup.tile_hash", True)),
        grid=int(_get(cfg, "data.dedup.tile_grid", 3) or 3),
        min_tile_side=int(_get(cfg, "data.dedup.min_tile_side", 256) or 256),
    )


# ---------------------------------------------------------------- Sample 构造


_ANSWER_TYPES = {"closed", "short", "long", "measure"}


def build_samples(records: Iterable[Any], cfg: Any = None) -> List[Sample]:
    """原始记录（dict 或 Sample）→ 补齐字段的 `Sample` 列表。

    补齐规则：
      - tier / license / commercial_ok 从 cfg.data.datasets.<dataset> 兜底；
      - answer_type 不在 {closed,short,long,measure} 时，有框 → short，无框 → short；
      - boxes 接受 [x0,y0,x1,y1,(label)] 或 dict；
      - id 缺失时用 (dataset, image, question) 的稳定哈希；
      - `cfg.data.commercial_only=true` 时丢掉 non-commercial（商用轨）；
      - 同一 id 只保留第一条（稳定去重）；
      - `meta.zoomable` 若 meta.orig_size 可用则一并算好，供下游筛 zoom 任务。
    """
    zoom_min = int(_get(cfg, "data.resolution.zoom_min_side", 1024) or 1024)
    commercial_only = bool(_get(cfg, "data.commercial_only", False))
    ds_cfg = _get(cfg, "data.datasets", {}) or {}

    out: List[Sample] = []
    seen: set = set()
    n_drop_commercial = 0
    for r in records:
        if r is None:
            continue
        s = _coerce_sample(r, ds_cfg, zoom_min)
        if s is None:
            continue
        if commercial_only and not s.commercial_ok:
            n_drop_commercial += 1
            continue
        if s.id in seen:
            continue
        seen.add(s.id)
        out.append(s)
    if n_drop_commercial:
        _log("info", f"commercial_only 过滤掉 {n_drop_commercial} 条")
    return out


def _coerce_sample(r: Any, ds_cfg: dict, zoom_min: int) -> Optional[Sample]:
    if isinstance(r, Sample):
        d = r.to_json()
    elif isinstance(r, dict):
        d = dict(r)
    else:
        return None

    dataset = str(d.get("dataset") or d.get("source") or "")
    dm = ds_cfg.get(dataset, {}) if isinstance(ds_cfg, dict) else {}
    boxes = [_box_from_any(b) for b in (d.get("boxes") or [])]
    boxes = [b for b in boxes if b is not None]
    meta = dict(d.get("meta") or {})

    if "orig_size" in meta:
        _fill_zoomable(meta, zoom_min)

    at = str(d.get("answer_type") or "short").lower()
    if at not in _ANSWER_TYPES:
        at = "short"

    sid = str(d.get("id") or "")
    if not sid:
        raw = f"{dataset}|{d.get('image','')}|{d.get('question','')}"
        sid = hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]

    image = str(d.get("image") or "")
    if not image:
        return None

    tier = str(d.get("tier") or dm.get("tier") or ("A" if boxes else "B"))
    # commercial_ok：记录没填 license 时视为「未标注」，用配置兜底；填了就尊重记录本身，
    # 因为下游可能已经按子集（例如 ISIC 某些条目不可商用）改过。
    lic = str(d.get("license") or dm.get("license") or "unknown")
    if d.get("license"):
        commercial = bool(d.get("commercial_ok", dm.get("commercial", False)))
    else:
        commercial = bool(dm.get("commercial", d.get("commercial_ok", False)))
    return Sample(
        id=sid,
        image=image,
        question=str(d.get("question") or ""),
        answer=str(d.get("answer") or ""),
        dataset=dataset,
        tier=tier,
        answer_type=at,
        boxes=boxes,
        masks=list(d.get("masks") or []),
        license=lic,
        commercial_ok=commercial,
        meta=meta,
    )


def _box_from_any(b: Any) -> Optional[Box]:
    try:
        if isinstance(b, Box):
            return b
        if isinstance(b, dict):
            return Box(float(b["x0"]), float(b["y0"]), float(b["x1"]), float(b["y1"]), str(b.get("label", "")))
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            return Box(float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                       str(b[4]) if len(b) > 4 else "")
    except (TypeError, ValueError, KeyError):
        return None
    return None


def _fill_zoomable(meta: Dict[str, Any], zoom_min: int) -> None:
    v = meta.get("orig_size")
    if isinstance(v, (list, tuple)) and len(v) == 2:
        try:
            meta["zoomable"] = bool(max(int(v[0]), int(v[1])) >= zoom_min)
        except (TypeError, ValueError):
            pass
