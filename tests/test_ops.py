"""medtaco/tools/ops.py 的单测：不联网、不读大文件，全部用 tmp_path 造小图。"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medtaco.tools.ops import (  # noqa: E402
    MAX_UPSCALE,
    MIN_BOX_SIDE,
    ToolContext,
    ToolUnavailable,
    crop,
    execute,
    load_image,
    measure,
    parse_box,
    window,
    zoom,
    zoom_with_scale,
)
from medtaco.utils.schema import Box, ToolCall  # noqa: E402


# ---------------------------------------------------------------- 造图工具


def make_rgb(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    """固定 seed 的随机 RGB 图（HxWx3 uint8）。"""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8)


def write_png(path: str, arr: np.ndarray) -> str:
    Image.fromarray(arr).save(path)
    return path


# ---------------------------------------------------------------- load_image


def test_load_image_gray_to_rgb(tmp_path):
    gray = np.arange(64 * 32, dtype=np.uint16).reshape(32, 64)  # 16bit 灰度
    p = str(tmp_path / "gray.png")
    Image.fromarray(gray).save(p)

    img = load_image(p)
    assert img.shape == (32, 64, 3)
    assert img.dtype == np.uint8


def test_load_image_rgb_roundtrip(tmp_path):
    arr = make_rgb(20, 30, seed=1)
    p = write_png(str(tmp_path / "rgb.png"), arr)
    out = load_image(p)
    assert out.shape == (20, 30, 3)
    assert np.array_equal(out, arr)


# ---------------------------------------------------------------- crop


def test_crop_out_of_bounds_is_clipped():
    img = make_rgb(64, 64)
    sub = crop(img, Box(-10, -10, 40, 50))
    # clip 到 [0,64]x[0,64] -> 40 宽 50 高
    assert sub.shape == (50, 40, 3)
    assert np.array_equal(sub, img[0:50, 0:40])


def test_crop_partially_outside_right_bottom():
    img = make_rgb(64, 64)
    sub = crop(img, Box(50, 50, 200, 200))
    assert sub.shape == (14, 14, 3)
    assert np.array_equal(sub, img[50:64, 50:64])


def test_crop_fully_outside_raises():
    img = make_rgb(64, 64)
    with pytest.raises(ValueError):
        crop(img, Box(100, 100, 200, 200))


def test_crop_pad_ratio_expands():
    img = make_rgb(64, 64)
    sub = crop(img, Box(20, 20, 30, 30), pad_ratio=0.5)
    # dx = dy = 10 * 0.5 = 5 -> (15,15,35,35)
    assert sub.shape == (20, 20, 3)
    assert np.array_equal(sub, img[15:35, 15:35])


def test_crop_pad_clipped_at_border():
    img = make_rgb(64, 64)
    sub = crop(img, Box(2, 2, 12, 12), pad_ratio=1.0)
    # (2-10, 2-10, 12+10, 12+10) -> clip 到 (0,0,22,22)
    assert sub.shape == (22, 22, 3)


# ---------------------------------------------------------------- zoom


def test_zoom_downscales_to_out_side():
    img = make_rgb(64, 64)
    out = zoom(img, Box(0, 0, 64, 64), out_side=32)
    assert out.shape == (32, 32, 3)
    assert max(out.shape[:2]) <= 32


def test_zoom_upscale_is_capped():
    """小裁剪区不能无限放大：倍率不得超过 MAX_UPSCALE。"""
    img = make_rgb(64, 64)
    sub, eff, scale = zoom_with_scale(img, Box(0, 0, 16, 16), out_side=1024)
    assert scale == pytest.approx(MAX_UPSCALE)
    # 16 * 4 = 64，而不是 1024
    assert max(sub.shape[:2]) == int(round(16 * MAX_UPSCALE))
    assert eff.as_tuple() == (0, 0, 16, 16)


def test_zoom_respects_out_side_when_within_cap():
    img = make_rgb(128, 128)
    # 裁剪 64x64，out_side=128 -> scale 2.0（<= 4）
    sub, _, scale = zoom_with_scale(img, Box(0, 0, 64, 64), out_side=128)
    assert scale == pytest.approx(2.0)
    assert max(sub.shape[:2]) == 128


def test_zoom_long_side_never_exceeds_out_side():
    img = make_rgb(100, 60)  # 高 100 宽 60
    for out_side in (8, 16, 33, 64, 128):
        sub, _, _ = zoom_with_scale(img, Box(0, 0, 60, 100), out_side=out_side)
        assert max(sub.shape[:2]) <= out_side, out_side


def test_zoom_preserves_aspect_ratio():
    img = make_rgb(64, 64)
    sub, _, _ = zoom_with_scale(img, Box(0, 0, 40, 20), out_side=80)
    h, w = sub.shape[:2]
    assert abs(w / h - 40 / 20) < 0.1


def test_zoom_custom_upscale_cap():
    img = make_rgb(64, 64)
    sub, _, scale = zoom_with_scale(img, Box(0, 0, 8, 8), out_side=1024, max_upscale=2.0)
    assert scale == pytest.approx(2.0)
    assert max(sub.shape[:2]) == 16


# ---------------------------------------------------------------- window


def _ct_meta(slope: float = 2.0, intercept: float = -1024.0, **kw) -> dict:
    meta = {"rescale_slope": slope, "rescale_intercept": intercept, "photometric": "MONOCHROME2"}
    meta.update(kw)
    return meta


def test_window_hounsfield_restore_is_exact():
    """构造已知 slope/intercept，验证输出确实是 HU 线性映射而不是 min-max 拉伸。"""
    pixels = np.array([[0, 256], [512, 768]], dtype=np.int16)
    # HU = pixel*2 - 1024 -> [-1024, -512, 0, 512]
    out = window(pixels, center=0, width=1024, meta=_ct_meta())
    # 窗区间 [-512, 512]：-1024 和 -512 clip 到 0，0 -> 127.5->128，512 -> 255
    assert out.dtype == np.uint8
    assert out.tolist() == [[0, 0], [128, 255]]


def test_window_differs_from_minmax_stretch():
    """若只做 min-max 拉伸，整幅图会被铺满 0-255；HU 还原则应按窗截断。"""
    pixels = np.array([[1000, 1100], [1200, 1300]], dtype=np.int16)  # HU: 976..1576
    out = window(pixels, center=1000, width=100, meta=_ct_meta())
    # 窗区间 [950,1050]：1000->127，1100(1576HU)->255，其余同理
    expected_hu = pixels.astype(np.float64) * 2.0 - 1024.0
    expected = np.clip(np.round((expected_hu - 950.0) / 100.0 * 255.0), 0, 255).astype(np.uint8)
    assert np.array_equal(out, expected)
    assert out.min() > 0 and out.max() == 255


def test_window_uses_slope_not_identity():
    pixels = np.array([[0, 100], [200, 300]], dtype=np.int16)
    a = window(pixels, center=0, width=600, meta=_ct_meta(slope=1.0, intercept=0.0))
    b = window(pixels, center=0, width=600, meta=_ct_meta(slope=2.0, intercept=0.0))
    assert not np.array_equal(a, b)


def test_window_accepts_rgb_uses_first_channel():
    pixels = np.zeros((2, 2, 3), dtype=np.int16)
    pixels[..., 0] = 512
    pixels[..., 1] = 0
    pixels[..., 2] = 0
    out = window(pixels, center=0, width=1024, meta=_ct_meta())
    assert out.shape == (2, 2)
    assert np.all(out == 128)


def test_window_missing_rescale_raises():
    pixels = np.zeros((4, 4), dtype=np.int16)
    with pytest.raises(ToolUnavailable):
        window(pixels, center=40, width=400, meta={})
    with pytest.raises(ToolUnavailable):
        window(pixels, center=40, width=400, meta={"rescale_slope": 1.0})  # 缺 intercept


def test_window_rejects_non_monochrome2():
    pixels = np.zeros((4, 4), dtype=np.int16)
    meta = _ct_meta(photometric="MONOCHROME1")
    with pytest.raises(ToolUnavailable):
        window(pixels, center=40, width=400, meta=meta)


def test_window_invalid_width_raises():
    pixels = np.zeros((4, 4), dtype=np.int16)
    with pytest.raises(ToolUnavailable):
        window(pixels, center=40, width=0, meta=_ct_meta())


# ---------------------------------------------------------------- measure


def test_measure_long_short_area():
    box = Box(0, 0, 10, 20)  # w=10, h=20 px
    meta = {"pixel_spacing": (0.5, 0.25)}  # sx=0.5, sy=0.25 mm/px
    assert measure(box, meta, "long") == pytest.approx(20 * 0.25)   # 5.0 mm
    assert measure(box, meta, "short") == pytest.approx(10 * 0.5)   # 5.0 mm
    assert measure(box, meta, "area") == pytest.approx(10 * 0.5 * 20 * 0.25)  # 25 mm^2


def test_measure_anisotropic_spacing_long_picks_image_axis():
    """spacing 各向异性时，long 取 mm 后的最大值，而不是像素上的最大边。"""
    box = Box(0, 0, 10, 20)  # 像素上高更长
    meta = {"pixel_spacing": (2.0, 1.0)}  # 但 x 方向更"胖"
    assert measure(box, meta, "long") == pytest.approx(20.0)   # max(20, 20) -> 20
    meta2 = {"pixel_spacing": (3.0, 1.0)}
    assert measure(box, meta2, "long") == pytest.approx(30.0)  # max(30, 20)
    assert measure(box, meta2, "short") == pytest.approx(20.0)


def test_measure_missing_spacing_raises():
    box = Box(0, 0, 10, 20)
    with pytest.raises(ToolUnavailable):
        measure(box, {}, "long")
    with pytest.raises(ToolUnavailable):
        measure(box, {"pixel_spacing": None}, "long")
    with pytest.raises(ToolUnavailable):
        measure(box, {"pixel_spacing": (0.5,)}, "long")


def test_measure_bad_mode_raises_value_error():
    with pytest.raises(ValueError):
        measure(Box(0, 0, 10, 20), {"pixel_spacing": (1.0, 1.0)}, "diagonal")


# ---------------------------------------------------------------- execute


def _ctx(tmp_path, **kw) -> ToolContext:
    defaults = dict(
        image=make_rgb(64, 64),
        meta={"pixel_spacing": (0.5, 0.5), "rescale_slope": 2.0,
              "rescale_intercept": -1024.0, "photometric": "MONOCHROME2"},
        out_dir=str(tmp_path / "obs"),
        obs_max_side=256,
        jpeg_quality=0,  # 存 PNG，避免 JPEG 损失影响断言
    )
    defaults.update(kw)
    return ToolContext(**defaults)


def test_execute_crop_writes_file_and_meta(tmp_path):
    ctx = _ctx(tmp_path)
    res = execute(ToolCall("crop", {"box": [10, 10, 40, 42]}), ctx)
    assert res.ok
    assert res.image and os.path.exists(res.image)
    assert res.meta["crop_box"] == [10, 10, 40, 42]
    assert res.meta["out_size"] == [30, 32]
    img = load_image(res.image)
    assert img.shape == (32, 30, 3)


def test_execute_crop_clips_out_of_bounds_box(tmp_path):
    ctx = _ctx(tmp_path)
    res = execute(ToolCall("crop", {"box": [-20, -20, 500, 500]}), ctx)
    assert res.ok
    assert res.meta["crop_box"] == [0, 0, 64, 64]


def test_execute_rejects_degenerate_box(tmp_path):
    ctx = _ctx(tmp_path)
    res = execute(ToolCall("crop", {"box": [0, 0, 5, 50]}), ctx)  # 宽 5 < 8
    assert not res.ok
    assert "退化框" in res.error
    assert res.meta["reason"] == "degenerate_box"

    res2 = execute(ToolCall("zoom", {"box": [10, 10, 200, 15]}), ctx)  # clip 后高 5
    assert not res2.ok
    assert res2.meta["reason"] == "degenerate_box"


def test_execute_rejects_fully_outside_box(tmp_path):
    ctx = _ctx(tmp_path)
    res = execute(ToolCall("crop", {"box": [100, 100, 200, 200]}), ctx)
    assert not res.ok


def test_execute_zoom_reports_actual_scale(tmp_path):
    ctx = _ctx(tmp_path)
    res = execute(ToolCall("zoom", {"box": [0, 0, 16, 16], "out_side": 1024}), ctx)
    assert res.ok
    assert res.meta["actual_scale"] == pytest.approx(MAX_UPSCALE)
    assert max(res.meta["out_size"]) == 64


def test_execute_zoom_observation_capped_by_obs_max_side(tmp_path):
    ctx = _ctx(tmp_path, obs_max_side=32)
    res = execute(ToolCall("zoom", {"box": [0, 0, 64, 64], "out_side": 1024}), ctx)
    assert res.ok
    assert max(load_image(res.image).shape[:2]) <= 32


def test_execute_window_ok_and_missing_hu(tmp_path):
    pix = np.full((32, 32), 512, dtype=np.int16)
    ctx = _ctx(tmp_path, image=pix)
    res = execute(ToolCall("window", {"center": 0, "width": 1024}), ctx)
    assert res.ok and os.path.exists(res.image)
    assert load_image(res.image).max() in (127, 128)

    ctx2 = _ctx(tmp_path, image=pix, meta={})
    res2 = execute(ToolCall("window", {"center": 0, "width": 1024}), ctx2)
    assert not res2.ok
    assert res2.meta["reason"] == "tool_unavailable"


def test_execute_measure_ok_and_missing_spacing(tmp_path):
    ctx = _ctx(tmp_path)
    res = execute(ToolCall("measure", {"box": [0, 0, 10, 20], "mode": "long"}), ctx)
    assert res.ok and res.image is None
    assert "mm" in res.text
    assert res.meta["value"] == pytest.approx(10.0)

    ctx2 = _ctx(tmp_path, meta={})
    res2 = execute(ToolCall("measure", {"box": [0, 0, 10, 20]}), ctx2)
    assert not res2.ok
    assert res2.meta["reason"] == "tool_unavailable"


def test_execute_unknown_tool_and_bad_args(tmp_path):
    ctx = _ctx(tmp_path)
    assert not execute(ToolCall("rotate", {}), ctx).ok
    assert not execute(ToolCall("crop", {}), ctx).ok
    assert not execute(ToolCall("crop", {"box": [0, 0]}), ctx).ok


def test_execute_counter_makes_unique_files(tmp_path):
    ctx = _ctx(tmp_path)
    names = set()
    for _ in range(3):
        res = execute(ToolCall("crop", {"box": [0, 0, 32, 32]}), ctx)
        assert res.ok
        names.add(res.image)
    assert len(names) == 3


def test_execute_load_image_sets_context(tmp_path):
    p = write_png(str(tmp_path / "in.png"), make_rgb(16, 24, seed=4))
    ctx = ToolContext(image=None, meta={}, out_dir=str(tmp_path / "obs"))
    res = execute(ToolCall("load_image", {"path": p}), ctx)
    assert res.ok
    assert ctx.image is not None and ctx.image.shape == (16, 24, 3)


def test_execute_window_capped_by_obs_max_side(tmp_path):
    pix = np.full((128, 128), 512, dtype=np.int16)
    ctx = _ctx(tmp_path, image=pix, obs_max_side=64)
    res = execute(ToolCall("window", {"center": 0, "width": 1000}), ctx)
    assert res.ok
    assert max(load_image(res.image).shape[:2]) <= 64


def test_parse_box_formats():
    assert parse_box([1, 2, 3, 4]).as_tuple() == (1.0, 2.0, 3.0, 4.0)
    assert parse_box([1, 2, 3, 4, "lesion"]).label == "lesion"
    assert parse_box({"x0": 1, "y0": 2, "x1": 3, "y1": 4}).as_tuple() == (1, 2, 3, 4)
    assert parse_box(Box(1, 2, 3, 4)).as_tuple() == (1, 2, 3, 4)


def test_min_box_side_constant():
    assert MIN_BOX_SIDE == 8
