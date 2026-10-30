"""medtaco/eval/metrics.py 的单测：不联网、不读大文件、不依赖本地没装的东西。

覆盖点（对应任务书要求）：
  - IoU 边界（相同/相离/仅共边/退化框/包含）
  - VEG 的正负号语义（破坏后变低 → VEG>0）
  - 三种破坏的可复现性（同 seed 同结果）与基本性质（形状不变、内容变化）
  - paired_report 在构造的 b/c 计数上给出正确 p、有效样本量
  - per_source_report 不混合 OmniMedVQA
  - 效率三指标 + oracle_crop_ceiling
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pytest

from medtaco.eval import metrics as M
from medtaco.utils.schema import Box
from medtaco.utils import stats


# ============================================================ IoU


def test_iou_identical_is_one() -> None:
    b = Box(10, 10, 30, 40)
    assert M.iou(b, Box(10, 10, 30, 40)) == pytest.approx(1.0)


def test_iou_half_overlap() -> None:
    # A=20x20=400，B=20x20=400，交=10x20=200，并=600 → 1/3
    a = Box(0, 0, 20, 20)
    b = Box(10, 0, 30, 20)
    assert M.iou(a, b) == pytest.approx(200 / 600)
    # IoU 对称
    assert M.iou(b, a) == pytest.approx(M.iou(a, b))


def test_iou_disjoint_is_zero() -> None:
    assert M.iou(Box(0, 0, 10, 10), Box(50, 50, 60, 60)) == 0.0


def test_iou_edge_touching_is_zero() -> None:
    # 只共享一条边，交集面积为 0
    assert M.iou(Box(0, 0, 10, 10), Box(10, 0, 20, 10)) == 0.0


def test_iou_degenerate_box_returns_zero() -> None:
    """宽或高为 0 的退化框：schema 允许构造，但 IoU 必须归零（不能除零）。"""
    zero_w = Box(5, 0, 5, 10)
    assert zero_w.w == 0
    assert M.iou(zero_w, Box(0, 0, 10, 10)) == 0.0
    assert M.iou(Box(0, 0, 10, 10), zero_w) == 0.0
    assert M.iou(zero_w, zero_w) == 0.0
    zero_h = Box(0, 7, 10, 7)
    assert M.iou(zero_h, Box(0, 0, 10, 10)) == 0.0


def test_iou_containment() -> None:
    # 小框完全在大框里：IoU = 小/大 = 100/400
    big = Box(0, 0, 20, 20)
    small = Box(5, 5, 15, 15)
    assert M.iou(big, small) == pytest.approx(100 / 400)


def test_mean_iou_skips_missing_boxes() -> None:
    recs = [
        {"box": [0, 0, 10, 10], "pred_box": [0, 0, 10, 10]},
        {"box": [0, 0, 10, 10]},                 # 缺预测框 → 跳过
        {"box": [0, 0, 10, 10], "pred_box": [20, 20, 30, 30]},
    ]
    assert M.mean_iou(recs) == pytest.approx(0.5)
    assert math.isnan(M.mean_iou([{"id": "x"}]))


# ============================================================ VEG


def test_veg_sign_semantics() -> None:
    """破坏后准确率下降 → VEG > 0（这正是"工具承重"的证据）。"""
    assert M.veg(0.8, 0.5) == pytest.approx(0.3)
    assert M.veg(0.5, 0.8) < 0
    assert M.veg(0.7, 0.7) == 0.0
    assert math.isnan(M.veg(float("nan"), 0.5))
    assert math.isnan(M.veg(0.5, None))  # type: ignore[arg-type]


# ============================================================ 破坏


def _img(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("mode", M.CORRUPT_MODES)
def test_corrupt_preserves_shape_and_dtype(mode: str) -> None:
    img = _img()
    out = M.corrupt_crop(img, mode, rng=1)
    assert isinstance(out, np.ndarray)
    assert out.shape == img.shape
    assert out.dtype == img.dtype
    # 返回新数组，不得就地改原图
    assert out is not img
    assert not np.array_equal(out, img)


@pytest.mark.parametrize("mode", M.CORRUPT_MODES)
def test_corrupt_reproducible_same_seed(mode: str) -> None:
    """同 seed 两次结果逐字节相同；不同 seed 应当不同（黑图除外）。"""
    img = _img()
    a = M.corrupt_crop(img, mode, rng=123)
    b = M.corrupt_crop(img, mode, rng=123)
    assert np.array_equal(a, b)
    c = M.corrupt_crop(img, mode, rng=124)
    if mode == "blackout":
        assert np.array_equal(a, c)  # 黑图与 seed 无关
    else:
        assert not np.array_equal(a, c)


def test_corrupt_reproducible_with_generator_object() -> None:
    img = _img()
    a = M.corrupt_crop(img, "shuffle_pixels", rng=np.random.default_rng(7))
    b = M.corrupt_crop(img, "shuffle_pixels", rng=np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_blackout_is_all_zero() -> None:
    out = M.corrupt_crop(_img(), "blackout", rng=0)
    assert out.max() == 0 and out.min() == 0


def test_shuffle_pixels_keeps_pixel_multiset() -> None:
    """置换只打乱位置，不改变像素值的多重集合（强度直方图不变）。"""
    img = _img()
    out = M.corrupt_crop(img, "shuffle_pixels", rng=5)
    assert np.array_equal(np.sort(img.reshape(-1, 3), axis=0),
                          np.sort(out.reshape(-1, 3), axis=0))


def test_random_crop_area_matched_uses_source() -> None:
    """给了 source 时从全图随机取同面积块再缩回来：内容应来自 source。"""
    src = _img(128, 128, seed=11)
    img = src[:64, :64]  # 一块 64x64 的裁剪
    out = M.corrupt_crop(img, "random_crop_area_matched", rng=3, source=src)
    assert out.shape == img.shape
    assert not np.array_equal(out, img)


def test_random_crop_area_matched_without_source_still_changes() -> None:
    img = _img()
    out = M.corrupt_crop(img, "random_crop_area_matched", rng=0)
    assert out.shape == img.shape
    assert not np.array_equal(out, img)


def test_corrupt_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        M.corrupt_crop(_img(), "gaussian_blur", rng=0)


def test_corrupt_grayscale_2d() -> None:
    g = np.random.default_rng(0).integers(0, 256, size=(32, 48), dtype=np.uint8)
    for mode in M.CORRUPT_MODES:
        out = M.corrupt_crop(g, mode, rng=2)
        assert out.shape == g.shape


# ============================================================ run_veg


def test_run_veg_positive_when_corruption_hurts() -> None:
    """构造：正常图亮 → 判对；黑图 → 判错。VEG 必须为正。"""
    recs = [{"id": f"r{i}", "image": np.full((16, 16, 3), 200, dtype=np.uint8),
             "correct": True} for i in range(20)]

    def evaluate(rec: Dict, img: np.ndarray) -> bool:
        return float(img.mean()) > 100.0

    out = M.run_veg(recs, M.corrupt_crop, evaluate, modes=["blackout"], n_boot=200, seed=0)
    assert out["n"] == 20
    assert out["acc_normal"] == pytest.approx(1.0)
    pm = out["per_mode"]["blackout"]
    assert pm["acc_corrupted"] == pytest.approx(0.0)
    assert pm["veg"] == pytest.approx(1.0)
    assert pm["veg"] > 0  # 语义：破坏后变低 → VEG>0
    assert pm["flips_down"] == 20 and pm["flips_up"] == 0
    # 配对 delta 的点估计应等于 acc_normal - acc_corrupted
    assert pm["veg_ci"][0] == pytest.approx(1.0)
    assert out["veg_mean"] == pytest.approx(1.0)


def test_run_veg_zero_when_corruption_ignored() -> None:
    """模型不看图（恒答对）→ 三种 mode 的 VEG 都是 0：结构不生效。"""
    recs = [{"id": f"r{i}", "image": _img(16, 16, seed=i), "correct": True} for i in range(5)]
    out = M.run_veg(recs, M.corrupt_crop, lambda r, im: True, n_boot=100, seed=0)
    for mode in M.CORRUPT_MODES:
        assert out["per_mode"][mode]["veg"] == pytest.approx(0.0)
    assert out["veg_mean"] == pytest.approx(0.0)


def test_run_veg_skips_records_without_image() -> None:
    recs = [{"id": "a", "image": _img(8, 8), "correct": True},
            {"id": "b", "correct": True}]  # 无图 → 跳过
    out = M.run_veg(recs, M.corrupt_crop, lambda r, im: True,
                    modes=["blackout"], n_boot=50, seed=0)
    assert out["n"] == 1
    assert out["n_skipped"] == 1


def test_run_veg_reproducible_across_runs() -> None:
    recs = [{"id": str(i), "image": _img(32, 32, seed=i)} for i in range(6)]
    calls: List[np.ndarray] = []

    def evaluate(rec: Dict, img: np.ndarray) -> bool:
        calls.append(img.copy())
        return bool(img.mean() > 128)

    r1 = M.run_veg(recs, M.corrupt_crop, evaluate, n_boot=100, seed=7)
    first = [c.copy() for c in calls]
    calls.clear()
    r2 = M.run_veg(recs, M.corrupt_crop, evaluate, n_boot=100, seed=7)
    second = calls
    assert r1["acc_normal"] == r2["acc_normal"]
    assert r1["veg"] == r2["veg"]
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_run_veg_empty_records() -> None:
    out = M.run_veg([], M.corrupt_crop, lambda r, im: True)
    assert out["n"] == 0
    assert math.isnan(out["acc_normal"])


# ============================================================ paired_report


def test_paired_report_mcnemar_counts_and_p() -> None:
    """构造 b=10 / c=0 的不一致：精确双尾 p = 2 * 2^-10 = 0.001953125。"""
    a = [True] * 10 + [True] * 40 + [False] * 50
    b = [False] * 10 + [True] * 40 + [False] * 50
    rep = M.paired_report(a, b, n_boot=300, icc=0.0, m=1.0, seed=0)
    assert rep["n"] == 100
    assert rep["mcnemar"]["b"] == 10
    assert rep["mcnemar"]["c"] == 0
    assert rep["mcnemar"]["p"] == pytest.approx(2 * 0.5 ** 10, rel=1e-9)
    # 与 utils.stats 的实现一致
    _, p_ref = stats.mcnemar(10, 0)
    assert rep["mcnemar"]["p"] == pytest.approx(p_ref)
    # acc：A 50/100，B 40/100，差值 0.10
    assert rep["acc_a"] == pytest.approx(0.5)
    assert rep["acc_b"] == pytest.approx(0.4)
    assert rep["diff"] == pytest.approx(0.1)
    est, lo, hi = rep["diff_boot_ci"]
    assert est == pytest.approx(0.1)
    assert lo <= est <= hi


def test_paired_report_symmetric_b_c_swap() -> None:
    a = [True] * 6 + [False] * 7
    b = [False] * 6 + [True] * 7
    r_ab = M.paired_report(a, b, n_boot=200, seed=1)
    r_ba = M.paired_report(b, a, n_boot=200, seed=1)
    assert r_ab["mcnemar"]["b"] == r_ba["mcnemar"]["c"]
    assert r_ab["mcnemar"]["p"] == pytest.approx(r_ba["mcnemar"]["p"])
    assert r_ab["diff"] == pytest.approx(-r_ba["diff"])


def test_paired_report_no_discordance_p_is_one() -> None:
    a = [True, False, True, True]
    r = M.paired_report(a, list(a), n_boot=100, seed=0)
    assert r["mcnemar"]["discordant"] == 0
    assert r["mcnemar"]["p"] == 1.0
    assert r["diff"] == 0.0


def test_paired_report_effective_n() -> None:
    """ICC>0 时有效样本量按 DEFF 缩小：n_eff = n/(1+(m-1)·ICC)。"""
    a = [True, False] * 50
    b = [True, True] * 50
    rep = M.paired_report(a, b, n_boot=100, icc=0.2, m=5, seed=0)
    deff = 1 + (5 - 1) * 0.2
    assert rep["deff"] == pytest.approx(deff)
    assert rep["effective_n"] == pytest.approx(100 / deff)
    assert rep["effective_n"] < rep["n"]
    assert rep["halfwidth_effective"] > rep["halfwidth_nominal"]
    # ICC=0 时有效样本量 = n
    rep0 = M.paired_report(a, b, n_boot=100, icc=0.0, m=5, seed=0)
    assert rep0["effective_n"] == pytest.approx(100.0)


def test_paired_report_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        M.paired_report([True, False], [True])


def test_paired_report_empty() -> None:
    rep = M.paired_report([], [], n_boot=10)
    assert rep["n"] == 0
    assert math.isnan(rep["diff"])


# ============================================================ per_source_report


def _recs(dataset: str, n: int, k: int, source: str | None = None) -> List[Dict]:
    out = []
    for i in range(n):
        r: Dict = {"id": f"{dataset}-{i}", "dataset": dataset, "correct": i < k}
        if source:
            r["source"] = source
        out.append(r)
    return out


def test_per_source_report_splits_omnimedvqa_subsources() -> None:
    recs = (
        _recs("OmniMedVQA", 10, 5, source="Kvasir")
        + _recs("OmniMedVQA", 10, 9, source="Fundus")
        + _recs("SLAKE", 10, 8)
    )
    rep = M.per_source_report(recs)
    labels = set(rep["per_source"])
    assert "OmniMedVQA/Kvasir" in labels
    assert "OmniMedVQA/Fundus" in labels
    assert "OmniMedVQA" not in labels          # 不得出现聚合键
    assert rep["per_source"]["OmniMedVQA/Kvasir"]["acc"] == pytest.approx(0.5)
    assert rep["per_source"]["OmniMedVQA/Fundus"]["acc"] == pytest.approx(0.9)
    # overall 是 30 题的正确数 5+9+8，不是"{Omni 平均=.7} 与 D 的平均"
    assert rep["overall"]["n"] == 30
    assert rep["overall"]["acc"] == pytest.approx((5 + 9 + 8) / 30)
    assert rep["mixed_excluded"] == []


def test_per_source_report_omnimedvqa_without_subsource_is_excluded() -> None:
    """缺子源的 OmniMedVQA 记录 → 单独桶、标记 mixed、从 overall 剔除。"""
    recs = _recs("OmniMedVQA", 10, 10) + _recs("SLAKE", 10, 5)
    rep = M.per_source_report(recs)
    assert "OmniMedVQA/unknown" in rep["per_source"]
    assert rep["per_source"]["OmniMedVQA/unknown"]["mixed"] is True
    assert rep["mixed_excluded"] == ["OmniMedVQA/unknown"]
    # overall 只剩 SLAKE：0.5，绝不会被 100% 的 Omni 桶拉高
    assert rep["overall"]["n"] == 10
    assert rep["overall"]["acc"] == pytest.approx(0.5)
    assert any("混合源" in w for w in rep["warnings"])


def test_per_source_report_custom_source_key() -> None:
    recs = [{"id": "1", "bench": "VQA-RAD", "correct": True},
            {"id": "2", "bench": "PathVQA", "correct": False}]
    rep = M.per_source_report(recs, source_key="bench")
    assert set(rep["per_source"]) == {"VQA-RAD", "PathVQA"}


def test_per_source_report_skips_records_without_correct() -> None:
    recs = [{"id": "1", "dataset": "SLAKE", "correct": True},
            {"id": "2", "dataset": "SLAKE"}]  # 无 correct
    rep = M.per_source_report(recs)
    assert rep["per_source"]["SLAKE"]["n"] == 1
    assert any("缺少 correct" in w for w in rep["warnings"])


# ============================================================ 效率指标


def test_call_rate_and_avg_tokens() -> None:
    recs = [
        {"id": "1", "n_calls": 2, "tokens": 1000},
        {"id": "2", "calls": [{"name": "crop"}], "tokens": 3000},
        {"id": "3", "n_calls": 0, "tokens": 2000},
        {"id": "4", "steps": [{"call": {"name": "zoom"}}, {"call": None}], "tokens": 4000},
    ]
    assert M.call_rate(recs) == pytest.approx(0.75)
    assert M.avg_tokens(recs) == pytest.approx((1000 + 3000 + 2000 + 4000) / 4)
    assert math.isnan(M.call_rate([]))
    assert math.isnan(M.avg_tokens([{"id": "x"}]))


def test_tool_success_rate_per_call() -> None:
    recs = [
        {"id": "1", "n_calls": 2, "tool_oks": [True, False]},   # 1/2
        {"id": "2", "n_calls": 1, "tool_ok": True},             # 1/1
        {"id": "3", "n_calls": 0},                              # 不参与
    ]
    assert M.tool_success_rate(recs) == pytest.approx(2 / 3)
    # 没有任何调用时返回 nan，而不是 0（避免把"不调用"记成"全失败"）
    assert math.isnan(M.tool_success_rate([{"id": "x", "n_calls": 0}]))


def test_efficiency_report_bundle() -> None:
    recs = [{"id": "1", "n_calls": 2, "tokens": 100, "tool_ok": True},
            {"id": "2", "n_calls": 0, "tokens": 300}]
    rep = M.efficiency_report(recs)
    assert rep["n"] == 2
    assert rep["call_rate"] == pytest.approx(0.5)
    assert rep["avg_tokens"] == pytest.approx(200)
    assert rep["avg_calls"] == pytest.approx(1.0)
    assert rep["tool_success_rate"] == pytest.approx(1.0)


# ============================================================ oracle 上界


def test_oracle_crop_ceiling_by_condition_field() -> None:
    recs: List[Dict] = []
    for i in range(20):
        recs.append({"id": f"q{i}", "condition": "full", "correct": i < 10})        # 50%
        recs.append({"id": f"q{i}", "condition": "oracle_crop", "correct": i < 16})  # 80%
    rep = M.oracle_crop_ceiling(recs, n_boot=200, seed=0)
    assert rep["full"]["n"] == 20
    assert rep["oracle_crop"]["n"] == 20
    assert rep["full"]["acc"] == pytest.approx(0.5)
    assert rep["oracle_crop"]["acc"] == pytest.approx(0.8)
    assert rep["delta"] == pytest.approx(0.3)     # 方向：oracle − full
    assert rep["n_paired"] == 20
    est, lo, hi = rep["delta_paired_ci"]
    assert est == pytest.approx(0.3)
    assert lo <= est <= hi


def test_oracle_crop_ceiling_two_lists() -> None:
    full = [{"id": str(i), "correct": i < 4} for i in range(10)]
    orc = [{"id": str(i), "correct": i < 7} for i in range(10)]
    rep = M.oracle_crop_ceiling(full, oracle_records=orc, n_boot=100, seed=0)
    assert rep["delta"] == pytest.approx(0.3)


def test_oracle_crop_ceiling_unpaired_gives_no_fake_ci() -> None:
    """长度不等且无 id → 不编造配对区间。"""
    full = [{"correct": False}] * 5
    orc = [{"correct": True}] * 8
    rep = M.oracle_crop_ceiling(full, oracle_records=orc, n_boot=50, seed=0)
    assert rep["delta"] == pytest.approx(1.0)  # 方向固定为 oracle − full
    assert rep["delta_paired_ci"] is None


def test_oracle_crop_ceiling_empty() -> None:
    rep = M.oracle_crop_ceiling([], n_boot=10)
    assert rep["full"]["n"] == 0
    assert math.isnan(rep["full"]["acc"])


# ============================================================ 落盘路径（tmp_path + PIL）

def test_run_veg_from_image_files(tmp_path) -> None:
    """记录里只给图片**路径**（真实评测的形态）：要能读回、能破坏、能配对。"""
    from PIL import Image

    rng = np.random.default_rng(2026)
    paths = []
    for i in range(4):
        arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        p = tmp_path / f"img{i}.png"
        Image.fromarray(arr).save(p)
        paths.append(str(p))

    recs = [{"id": f"q{i}", "image": p, "correct": i % 2 == 0} for i, p in enumerate(paths)]
    # 再加一条路径不存在的记录：必须走跳过分支而不是崩
    recs.append({"id": "missing", "image": str(tmp_path / "nope.png"), "correct": True})

    seen_shapes = []

    def evaluate(rec: Dict, img: np.ndarray) -> bool:
        seen_shapes.append(img.shape)
        return float(img.mean()) > 127.0

    out = M.run_veg(recs, M.corrupt_crop, evaluate, n_boot=100, seed=3)
    assert out["n"] == 4
    assert out["n_skipped"] == 1
    assert all(s == (64, 64, 3) for s in seen_shapes)
    assert set(out["per_mode"]) == set(M.CORRUPT_MODES)
    # 记录自带的 correct 优先，不再跑正常一路 → 只评估 4 条 × 3 mode 次
    assert len(seen_shapes) == 4 * len(M.CORRUPT_MODES)
    for mode in M.CORRUPT_MODES:
        assert out["per_mode"][mode]["n"] == 4
        assert out["per_mode"][mode]["flips_down"] + out["per_mode"][mode]["flips_up"] >= 0


def test_per_source_report_real_shaped_records(tmp_path) -> None:
    """混合源 + 多数据集：每个源各自出数，overall 不掺入混合桶。"""
    from PIL import Image

    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    p = tmp_path / "x.png"
    Image.fromarray(arr).save(p)

    recs = [
        {"id": "1", "dataset": "SLAKE", "image": str(p), "correct": True},
        {"id": "2", "dataset": "SLAKE", "image": str(p), "correct": False},
        {"id": "3", "dataset": "VQA-RAD", "image": str(p), "correct": True},
        {"id": "4", "dataset": "OmniMedVQA", "source": "Fundus", "image": str(p), "correct": True},
        {"id": "5", "dataset": "OmniMedVQA", "image": str(p), "correct": False},
    ]
    rep = M.per_source_report(recs, n_boot=100)
    assert set(rep["per_source"]) == {"SLAKE", "VQA-RAD", "OmniMedVQA/Fundus", "OmniMedVQA/unknown"}
    assert rep["per_source"]["SLAKE"]["acc"] == pytest.approx(0.5)
    assert rep["per_source"]["OmniMedVQA/Fundus"]["acc"] == pytest.approx(1.0)
    assert rep["mixed_excluded"] == ["OmniMedVQA/unknown"]
    # overall 只含 SLAKE(2) + VQA-RAD(1) + Omni/Fundus(1)
    assert rep["overall"]["n"] == 4
    assert rep["overall"]["acc"] == pytest.approx(3 / 4)
