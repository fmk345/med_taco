"""preprocess.py 单测：sidecar / 缩放 / 16-bit 转换 / pHash 去重 / 分块哈希 / build_samples。

不联网、不读大数据文件：所有数据用 tmp_path 现造（PIL 小图 + 小 json）。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
from PIL import Image

from medtaco.data.preprocess import (
    build_samples,
    hamming_distance,
    phash,
    phash_dedup,
    phash_dedup_from_cfg,
    preprocess_image,
    sidecar_path,
)
from medtaco.utils.config import Cfg
from medtaco.utils.schema import Sample


# ---------------------------------------------------------------- 造数工具


def _cfg(tmp_path, **data_over) -> Cfg:
    raw = {
        "seed": 42,
        "paths": {"raw": str(tmp_path / "raw"), "processed": str(tmp_path / "processed")},
        "data": {
            "dedup": {"phash_hamming": 6, "tile_hash": True, "tile_grid": 3, "min_tile_side": 256},
            "resolution": {"zoom_min_side": 1024, "max_side_cap": 4096},
            "commercial_only": False,
            "datasets": {
                "MS-CXR": {"tier": "A", "license": "PhysioNet-DUA", "commercial": False},
                "SLAKE": {"tier": "A", "license": "research-only", "commercial": False},
                "ChestX-ray14": {"tier": "A", "license": "public-domain", "commercial": True},
                "LIDC-IDRI": {"tier": "A", "license": "CC-BY-3.0", "commercial": True},
            },
        },
    }
    for k, v in data_over.items():
        raw["data"][k] = v
    return Cfg(raw)


def _noise_png(path, seed: int, size=(64, 64)) -> str:
    """确定性随机噪声图（np.random.default_rng 显式 seed）。"""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(arr).save(path)
    return path


def _blocky_png(path, blocks: list, cell: int = 32) -> str:
    """按 3x3 拼块：blocks 是 9 个 (seed) 或 None，每块 cell×cell 随机噪声。"""
    grid = 3
    out = np.zeros((grid * cell, grid * cell, 3), dtype=np.uint8)
    for idx, seed in enumerate(blocks):
        r, c = divmod(idx, grid)
        if seed is None:
            out[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = 128
        else:
            rng = np.random.default_rng(seed)
            out[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = rng.integers(
                0, 256, size=(cell, cell, 3), dtype=np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(out).save(path)
    return path


def _noisy_copy(src: str, dst: str, amp: int = 5, seed: int = 7) -> str:
    """加小幅均匀噪声的副本（模拟重压缩 / 轻微扰动）。"""
    rng = np.random.default_rng(seed)
    base = np.asarray(Image.open(src).convert("RGB")).astype(np.int16)
    noisy = np.clip(base + rng.integers(-amp, amp + 1, size=base.shape), 0, 255).astype(np.uint8)
    Image.fromarray(noisy).save(dst)
    return dst


# ---------------------------------------------------------------- preprocess_image


def test_preprocess_resizes_and_writes_sidecar(tmp_path):
    src = _noise_png(str(tmp_path / "src.png"), seed=21, size=(200, 100))  # W=200,H=100
    dst = str(tmp_path / "out" / "a.png")
    cfg = _cfg(tmp_path, resolution={"zoom_min_side": 1024, "max_side_cap": 50})
    side = preprocess_image(src, dst, cfg)
    assert os.path.exists(dst) and os.path.exists(sidecar_path(dst))
    with Image.open(dst) as im:
        assert im.mode == "RGB" and im.size == (50, 25)
    on_disk = json.loads(open(sidecar_path(dst), encoding="utf-8").read())
    assert on_disk["orig_size"] == [200, 100]
    assert on_disk["size"] == [50, 25]
    assert on_disk["scale"] == pytest.approx(0.25) and on_disk["scaled"] is True
    assert on_disk["spacing"] is None and on_disk["window"] is None
    assert on_disk["has_hu"] is False
    assert on_disk == side


def test_preprocess_no_upscale_and_jpeg(tmp_path):
    src = _noise_png(str(tmp_path / "src.png"), seed=22, size=(30, 20))
    dst = str(tmp_path / "b.jpg")
    side = preprocess_image(src, dst, _cfg(tmp_path))
    assert side["scale"] == 1.0 and side["scaled"] is False
    assert side["size"] == [30, 20]
    with Image.open(dst) as im:
        assert im.format == "JPEG" and im.size == (30, 20)


def test_preprocess_reads_source_dicom_sidecar(tmp_path):
    src = _noise_png(str(tmp_path / "ct.png"), seed=23, size=(64, 64))
    meta = {
        "PixelSpacing": [0.7, 0.7],
        "WindowCenter": [40, 80],
        "WindowWidth": [400, 800],
        "RescaleSlope": 1.0,
        "RescaleIntercept": -1024,
        "PhotometricInterpretation": "MONOCHROME2",
    }
    with open(sidecar_path(src), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    side = preprocess_image(src, str(tmp_path / "ct_out.png"), _cfg(tmp_path))
    assert side["spacing"] == [0.7, 0.7]
    assert side["window"] == {"center": 40.0, "width": 400.0}
    assert side["has_hu"] is True
    assert side["rescale"]["intercept"] == -1024.0


def test_preprocess_16bit_gray_to_8bit(tmp_path):
    arr = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) * 10)
    src = str(tmp_path / "g16.png")
    Image.fromarray(arr).save(src)
    side = preprocess_image(src, str(tmp_path / "g8.png"), _cfg(tmp_path))
    assert side["mode_in"] == "I;16" and side["mode_conversion"].startswith("I;16")
    with Image.open(tmp_path / "g8.png") as im:
        assert im.mode == "RGB"
        a = np.asarray(im)
        assert a[..., 0].min() == 0 and a[..., 0].max() == 255


def test_preprocess_bad_extension(tmp_path):
    src = _noise_png(str(tmp_path / "s.png"), seed=24)
    with pytest.raises(ValueError):
        preprocess_image(src, str(tmp_path / "s.bmp"), _cfg(tmp_path))


# ---------------------------------------------------------------- pHash


def test_phash_similar_and_dissimilar(tmp_path):
    a = _noise_png(str(tmp_path / "a.png"), seed=31, size=(64, 64))
    b = _noise_png(str(tmp_path / "b.png"), seed=32, size=(64, 64))
    ha, hb = phash(a), phash(b)
    assert hamming_distance(ha, ha) == 0
    assert hamming_distance(ha, hb) > 6   # 两张无关噪声图
    assert hamming_distance(ha, phash(_noisy_copy(a, str(tmp_path / "a_noisy.png")))) <= 6


def test_phash_dedup_basic(tmp_path):
    a = _noise_png(str(tmp_path / "a.png"), seed=41)
    b = _noise_png(str(tmp_path / "b.png"), seed=42)
    a2 = _noisy_copy(a, str(tmp_path / "a2.png"), amp=5, seed=7)

    dup = phash_dedup([a, a2, b], hamming=6, tile_hash=False)
    assert dup == {a2: a}          # a2 与 a 判重，b 独立


def test_phash_dedup_identical_file(tmp_path):
    a = _noise_png(str(tmp_path / "a.png"), seed=43)
    a2 = str(tmp_path / "a_copy.png")
    Image.open(a).save(a2)
    assert phash_dedup([a, a2], hamming=6, tile_hash=False) == {a2: a}


def test_phash_dedup_reports_bad_file(tmp_path, capsys):
    good = _noise_png(str(tmp_path / "g.png"), seed=44)
    bad = str(tmp_path / "bad.png")
    with open(bad, "wb") as f:
        f.write(b"not an image")
    dup = phash_dedup([bad, good], hamming=6, tile_hash=False)
    assert good not in dup and bad not in dup
    assert "[warn]" in capsys.readouterr().out


def test_phash_dedup_empty_and_brute_force_path(tmp_path):
    assert phash_dedup([]) == {}
    a = _noise_png(str(tmp_path / "a.png"), seed=45)
    b = _noise_png(str(tmp_path / "b.png"), seed=46)
    # 阈值 >= 63 时走暴力分支，不应报错
    dup = phash_dedup([a, b], hamming=63, tile_hash=False)
    assert isinstance(dup, dict)


def test_tile_hash_catches_shared_region(tmp_path):
    """整图不同、但有一个分块完全相同：全图 pHash 抓不到，分块哈希必须抓到。"""
    shared = 777
    blocks_a = [1, 2, 3, 4, 5, 6, 7, 8, shared]     # shared 在右下角
    blocks_b = [11, 12, 13, 14, 15, 16, 17, 18, shared]
    a = _blocky_png(str(tmp_path / "ta.png"), blocks_a, cell=32)
    b = _blocky_png(str(tmp_path / "tb.png"), blocks_b, cell=32)

    # 前提校验：全图哈希确实不同（否则本测试不成立）
    assert hamming_distance(phash(a), phash(b)) > 6

    # 分块边长 32 < 默认 min_tile_side=256，所以必须显式放低阈值
    assert phash_dedup([a, b], hamming=6, tile_hash=False) == {}
    assert phash_dedup([a, b], hamming=6, tile_hash=True, grid=3, min_tile_side=16) == {b: a}


def test_tile_hash_respects_min_tile_side(tmp_path):
    shared = 888
    a = _blocky_png(str(tmp_path / "ta.png"), [1, 2, 3, 4, 5, 6, 7, 8, shared], cell=32)
    b = _blocky_png(str(tmp_path / "tb.png"), [11, 12, 13, 14, 15, 16, 17, 18, shared], cell=32)
    # 块 32px < min_tile_side 64 → 分块直接放弃，判不出重复
    assert phash_dedup([a, b], hamming=6, tile_hash=True, grid=3, min_tile_side=64) == {}


def test_tile_hash_not_triggered_by_flat_tiles(tmp_path):
    """纯色块（无纹理）不参与分块比对，避免大片均匀肺野互相误判。

    两图只有左上角一块有纹理且内容不同，其余 8 块是完全相同的纯色——若把纯色块也纳入
    分块比对，这 8 块会平凡地「相同」而误判为重复；滤波后必须判为独立。
    """
    a = _blocky_png(str(tmp_path / "ta.png"), [5, None, None, None, None, None, None, None, None], cell=32)
    b = _blocky_png(str(tmp_path / "tb.png"), [6, None, None, None, None, None, None, None, None], cell=32)
    assert hamming_distance(phash(a), phash(b)) > 6   # 前提：全图不相同
    assert phash_dedup([a, b], hamming=6, tile_hash=True, grid=3, min_tile_side=16) == {}


def test_phash_dedup_from_cfg(tmp_path):
    a = _noise_png(str(tmp_path / "a.png"), seed=47)
    a2 = _noisy_copy(a, str(tmp_path / "a2.png"))
    cfg = _cfg(tmp_path, dedup={"phash_hamming": 6, "tile_hash": False,
                                "tile_grid": 3, "min_tile_side": 256})
    assert phash_dedup_from_cfg([a, a2], cfg) == {a2: a}


def test_hamming_distance_symmetry():
    assert hamming_distance(0, 0b1011) == 3
    assert hamming_distance(0b1011, 0) == 3
    assert hamming_distance((1 << 63), 0) == 1


# ---------------------------------------------------------------- build_samples


def test_build_samples_from_records(tmp_path):
    cfg = _cfg(tmp_path)
    recs = [
        {"id": "s1", "image": "a.png", "question": "q", "answer": "a",
         "dataset": "ChestX-ray14", "boxes": [[0, 0, 10, 10, "Nodule"]]},
        {"image": "b.png", "question": "q2", "answer": "a2", "dataset": "SLAKE",
         "answer_type": "weird_type", "meta": {"orig_size": [2048, 1024]}},
        {"id": "s1", "image": "dup.png", "question": "", "answer": "", "dataset": "SLAKE"},
        {"id": "bad", "image": "", "question": "", "answer": ""},
    ]
    out = build_samples(recs, cfg)
    assert len(out) == 2                      # 重复 id 与缺图都丢掉
    s1, s2 = out
    assert s1.tier == "A" and s1.license == "public-domain" and s1.commercial_ok is True
    assert s1.boxes[0].label == "Nodule"
    assert s2.tier == "A" and s2.answer_type == "short"   # 非法 answer_type 归一到 short
    assert s2.meta["zoomable"] is True                     # 由原尺寸算出
    # 自动 id 稳定
    again = build_samples([recs[1]], cfg)
    assert again[0].id == s2.id


def test_build_samples_commercial_only(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.set("data.commercial_only", True)
    recs = [
        {"id": "a", "image": "a.png", "question": "", "answer": "", "dataset": "ChestX-ray14"},
        {"id": "b", "image": "b.png", "question": "", "answer": "", "dataset": "SLAKE"},
    ]
    out = build_samples(recs, cfg)
    assert [s.id for s in out] == ["a"]


def test_build_samples_accepts_sample_objects(tmp_path):
    cfg = _cfg(tmp_path)
    s = Sample(id="x", image="a.png", question="q", answer="a", dataset="LIDC-IDRI")
    out = build_samples([s], cfg)
    # 记录没填 license → 用配置兜底
    assert out[0].license == "CC-BY-3.0" and out[0].commercial_ok is True


def test_build_samples_respects_record_commercial_flag(tmp_path):
    """记录显式给了 license（子集已单独判定）时，尊重记录而不是配置。"""
    cfg = _cfg(tmp_path)
    rec = {"id": "y", "image": "a.png", "question": "", "answer": "",
           "dataset": "ChestX-ray14", "license": "public-domain", "commercial_ok": False}
    assert build_samples([rec], cfg)[0].commercial_ok is False


def test_build_samples_default_cfg_none(tmp_path):
    out = build_samples([{"id": "z", "image": "a.png", "question": "q", "answer": "a"}], None)
    assert len(out) == 1 and out[0].license == "unknown"


def test_build_samples_drops_bad_boxes(tmp_path):
    cfg = _cfg(tmp_path)
    rec = {"id": "w", "image": "a.png", "question": "", "answer": "",
           "boxes": [[0, 0, 10, 10], "junk", [3, 3, 1, 1]]}
    out = build_samples([rec], cfg)
    assert len(out[0].boxes) == 1   # 字符串与反向框都被丢弃
