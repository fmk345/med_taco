"""medtaco/rl/advantage.py 的单测。

轨迹布局（手算期望值全部写死成具体 token 下标）：
  单调用 n=14：T1=[0,5) 即 0..4，obs=[5,9) 即 5..8，t2=[9,14) 即 9..13
  双调用 n=24：T1=[0,5)，call0 obs=[5,9) t2=[9,14)，call1 obs=[14,18) t2=[18,24)
"""
from __future__ import annotations

import pytest

from medtaco.rl.advantage import (
    Segment,
    TurnMasks,
    build_loss_masks,
    group_advantage,
    ogar_masks,
    staged_deltas,
    token_length,
    validate_probes,
)

# ---------------------------------------------------------------- 夹具


def _one_call() -> Segment:
    return Segment(t1=(0, 5), calls=[(5, 9, 9, 14)])


def _two_calls() -> Segment:
    return Segment(t1=(0, 5), calls=[(5, 9, 9, 14), (14, 18, 18, 24)])


# 单调用四种 regime 的期望 mask（14 个 token）
EXP_HELPFUL = [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
EXP_REDUNDANT = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
EXP_MISLEADING = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
EXP_NECESSARY_FAIL = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]


# ---------------------------------------------------------------- Δ_k


def test_staged_deltas_basic() -> None:
    # a0=0,a1=1,a2=0 -> Δ1=+1, Δ2=-1
    assert staged_deltas([0, 1, 0]) == [1, -1]
    assert staged_deltas([-1, 0, 0, 1]) == [1, 0, 1]
    # 单次探针（K=0）没有 Δ
    assert staged_deltas([1]) == []


def test_staged_deltas_sum_identity() -> None:
    """恒等式：sum(Δ_k) == a_K - a_0（任意分数序列）。"""
    import random

    rng = random.Random(20261028)
    for _ in range(200):
        k = rng.randint(0, 6)
        scores = [rng.choice([-1, 0, 1]) for _ in range(k + 1)]
        deltas = staged_deltas(scores)
        assert len(deltas) == k
        assert sum(deltas) == scores[-1] - scores[0]


def test_staged_deltas_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        staged_deltas([0, 2])
    with pytest.raises(ValueError):
        staged_deltas([0.0, 1.0])
    with pytest.raises(ValueError):
        staged_deltas([])


# ---------------------------------------------------------------- 单调用 4 regime


def test_regime_helpful() -> None:
    """有用：a0=0 -> a1=1。T1 全 1；obs 被强制 0；t2 全 1。"""
    m = ogar_masks([0, 1], _one_call())
    assert isinstance(m, TurnMasks)
    assert m.token_mask == EXP_HELPFUL
    assert m.call_indices == [5]


def test_regime_redundant() -> None:
    """冗余：a0=1 -> a1=1，Δ_1=0 所以工具分支全 0，T1 仍拿信用（Δ_total=0>=0）。"""
    m = ogar_masks([1, 1], _one_call())
    assert m.token_mask == EXP_REDUNDANT


def test_regime_misleading() -> None:
    """误导：a0=1 -> a1=0，Δ_total<0 关掉 T1；惩罚落到工具分支，但 obs 仍为 0。"""
    m = ogar_masks([1, 0], _one_call())
    assert m.token_mask == EXP_MISLEADING


def test_regime_necessary_but_failed() -> None:
    """必需但失败：a0=-1 -> a1=-1。Δ_total=0>=0，T1 保留；Δ_1=0，工具分支无信用。"""
    m = ogar_masks([-1, -1], _one_call())
    assert m.token_mask == EXP_NECESSARY_FAIL


def test_mode_always_only_t1() -> None:
    """消融模式：只走 T1 逻辑，且无条件置 1，不看 Δ（即便 Δ_total<0）。"""
    m_pos = ogar_masks([0, 1], _one_call(), mode="always")
    m_neg = ogar_masks([1, 0], _one_call(), mode="always")
    exp = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert m_pos.token_mask == exp
    assert m_neg.token_mask == exp


def test_invalid_mode_and_mismatch() -> None:
    with pytest.raises(ValueError):
        ogar_masks([0, 1], _one_call(), mode="nope")
    with pytest.raises(ValueError):  # 探针数 != n_calls+1
        ogar_masks([0, 1, 1], _one_call())


# ---------------------------------------------------------------- 多轮分段


def test_multi_turn_d1_positive_d2_negative() -> None:
    """Δ_1>0, Δ_2<0：call0 的 t2 拿信用，call1 的 t2 也拿信用（Δ_2≠0），T1 因 Δ_total=0 保留。"""
    seg = _two_calls()
    assert token_length(seg) == 24
    m = ogar_masks([0, 1, 0], seg)
    exp = (
        [1, 1, 1, 1, 1]      # T1 0..4
        + [0, 0, 0, 0]        # obs0 5..8 强制 0
        + [1, 1, 1, 1, 1]     # t2_0 9..13
        + [0, 0, 0, 0]        # obs1 14..17 强制 0
        + [1, 1, 1, 1, 1, 1]  # t2_1 18..23
    )
    assert m.token_mask == exp
    assert m.call_indices == [5, 14]


def test_multi_turn_d2_zero_is_excluded() -> None:
    """Δ_2=0：第二次调用的 obs∪t2 全 0，只保留 call0 的 t2。"""
    m = ogar_masks([0, 1, 1], _two_calls())
    exp = (
        [1, 1, 1, 1, 1]
        + [0, 0, 0, 0]
        + [1, 1, 1, 1, 1]
        + [0] * 10            # obs1 + t2_1 全 0
    )
    assert m.token_mask == exp


def test_multi_turn_t1_gated_off() -> None:
    """Δ_total<0：T1 关；Δ_1=0 的 call0 无信用，但 Δ_2=-1 的 call1 仍拿信用。

    （分段路由与轨迹门相互独立：Δ_k≠0 就看 Δ_k 的符号，不看 Δ_total。）
    """
    # a=[1,1,0] -> Δ_1=0, Δ_2=-1, Δ_total=-1
    m = ogar_masks([1, 1, 0], _two_calls())
    exp = (
        [0] * 5              # T1 0..4 因 Δ_total<0 关闭
        + [0, 0, 0, 0]        # obs0 恒 0
        + [0] * 5             # t2_0 因 Δ_1=0 关闭
        + [0, 0, 0, 0]        # obs1 恒 0
        + [1, 1, 1, 1, 1, 1]  # t2_1 因 Δ_2≠0 打开
    )
    assert m.token_mask == exp


def test_multi_turn_d1_zero_d2_positive() -> None:
    """Δ_1=0, Δ_2=+1：只有第二次调用的 t2 拿信用，T1 因 Δ_total=1 保留。"""
    m = ogar_masks([0, 0, 1], _two_calls())
    exp = (
        [1, 1, 1, 1, 1]
        + [0] * 9             # obs0 + t2_0
        + [0, 0, 0, 0]        # obs1
        + [1, 1, 1, 1, 1, 1]  # t2_1
    )
    assert m.token_mask == exp


def test_multiple_segments_flattened() -> None:
    """多个 Segment 的 calls 按顺序拍平，编号连续。"""
    s1 = Segment(t1=(0, 3), calls=[(3, 6, 6, 10)])
    s2 = Segment(t1=(-1, -1), calls=[(10, 13, 13, 16)])
    # a=[0,1,0] -> Δ_1=+1, Δ_2=-1，两段 t2 都命中
    m = ogar_masks([0, 1, 0], [s1, s2])
    assert token_length([s1, s2]) == 16
    assert m.call_indices == [3, 10]
    # T1 只在第一段；第二段 t1 是占位符
    assert m.token_mask == [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]


def test_missing_segment_placeholders() -> None:
    """(-1,-1) 占位符不产生 mask，也不报错。"""
    seg = Segment(t1=(-1, -1), calls=[(-1, -1, 0, 4)])
    m = ogar_masks([0, 1], seg)
    assert m.token_mask == [1, 1, 1, 1]
    assert m.call_indices == [0]


# ---------------------------------------------------------------- 观测强制 mask


@pytest.mark.parametrize("scores", [[0, 1], [1, 0], [1, 1], [-1, 1], [0, -1]])
def test_observation_always_masked(scores: list[int]) -> None:
    """无论 Δ 怎么取值、无论哪种模式，obs 区间 [5,9) 必须是 0。"""
    for mode in ("paper", "always"):
        m = ogar_masks(scores, _one_call(), mode=mode)
        assert m.token_mask[5:9] == [0, 0, 0, 0], f"{scores} mode={mode}"


def test_observation_masked_even_inside_t1_rule() -> None:
    """段重叠的极端情形：obs 与 T1 重叠时，obs 的强制 0 仍然赢。"""
    seg = Segment(t1=(0, 10), calls=[(4, 8, 10, 12)])
    m = ogar_masks([0, 1], seg)
    # T1 置 1 覆盖 0..9，但 obs 4..7 被压回 0
    assert m.token_mask == [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]


# ---------------------------------------------------------------- group 归一化


def test_group_advantage_basic() -> None:
    # 均值 2.5, 总体 std sqrt((0.5^2*4)/4)? 值 [1,2,3,4] -> mean 2.5, std ~1.1180
    out = group_advantage([1.0, 2.0, 3.0, 4.0], 1e-6, "skip_group")
    assert out is not None
    assert out[0] == pytest.approx(-1.3416, abs=1e-3)
    assert out[3] == pytest.approx(1.3416, abs=1e-3)
    assert sum(out) == pytest.approx(0.0, abs=1e-9)


def test_zero_variance_skip_group() -> None:
    """组内全同 -> None（std 恰为 0）。"""
    assert group_advantage([1, 1, 1], 1e-6, "skip_group") is None
    assert group_advantage([0.0] * 8, 1e-6, "skip_group") is None
    assert group_advantage([-2], 1e-6, "skip_group") is None  # 单样本也按零方差


def test_zero_variance_eps_floor() -> None:
    """eps_floor 不跳过，零方差时输出全 0。"""
    out = group_advantage([1, 1, 1], 1e-6, "eps_floor")
    assert out == [0.0, 0.0, 0.0]


def test_nonzero_variance_not_skipped_with_tiny_eps() -> None:
    out = group_advantage([0, 1], 1e-6, "skip_group")
    assert out is not None
    assert out == pytest.approx([-1.0, 1.0])


def test_group_advantage_bad_policy() -> None:
    with pytest.raises(ValueError):
        group_advantage([1, 2], 1e-6, "whatever")


# ---------------------------------------------------------------- build_loss_masks


def test_build_loss_masks_two_channels() -> None:
    """4 条 rollout 覆盖 4 regime，两条通道各自独立组装。"""
    seg = _one_call()
    scores = [[0, 1], [1, 1], [1, 0], [-1, -1]]
    out = build_loss_masks(
        segments_list=[seg] * 4,
        probe_scores_list=scores,
        std_eps=1e-6,
        zero_var_policy="skip_group",
    )
    assert out["ok"] is True
    assert out["gate"] == [1, 1, 0, 1]
    assert out["delta_total"] == [1, 0, -1, 0]

    # 通道 1（outcome）用 OGAR 掩码，逐条对齐四种 regime
    assert out["loss_mask_1"] == [
        EXP_HELPFUL,
        EXP_REDUNDANT,
        EXP_MISLEADING,
        EXP_NECESSARY_FAIL,
    ]
    # 通道 2（process）= g * covered，covered = T1 ∪ t2（obs 恒 0）
    covered = [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    assert out["loss_mask_2"] == [
        covered,
        covered,
        [0] * 14,       # gate=0
        covered,
    ]
    # 通道 1 的 advantage 由最终分归一化：末项 [1,1,0,-1]
    assert out["adv_1"] is not None and len(out["adv_1"]) == 4
    # 通道 2 的 advantage 由 Δ_total 归一化：[1,0,-1,0]
    assert out["adv_2"] is not None and len(out["adv_2"]) == 4
    assert sum(out["adv_2"]) == pytest.approx(0.0, abs=1e-9)


def test_build_loss_masks_zero_variance_channel() -> None:
    """全部 Δ_total=0 -> 通道 2 的 advantage 为 None、mask 也为 None，但通道 1 仍在。"""
    scores = [[0, 0], [1, 1], [1, 1]]
    out = build_loss_masks(
        segments_list=[_one_call()] * 3,
        probe_scores_list=scores,
        std_eps=1e-6,
        zero_var_policy="skip_group",
    )
    assert out["ok"] is False
    assert out["adv_2"] is None
    assert out["loss_mask_2"] is None
    assert out["adv_1"] is not None
    assert out["loss_mask_1"] is not None


def test_build_loss_masks_reads_cfg() -> None:
    """cfg.rl.advantage 缺省值生效。"""
    import os

    import medtaco

    from medtaco.utils.config import load_config

    root = os.path.dirname(os.path.dirname(os.path.abspath(medtaco.__file__)))
    cfg_path = os.path.join(root, "configs", "default.yaml")
    assert os.path.exists(cfg_path), cfg_path
    cfg = load_config(cfg_path)
    out = build_loss_masks(
        segments_list=[_one_call()] * 2,
        probe_scores_list=[[0, 0], [0, 0]],
        cfg=cfg,
    )
    assert out["adv_1"] is None  # config 默认 skip_group，零方差跳过
    assert out["delta_total"] == [0, 0]


# ---------------------------------------------------------------- validate_probes


def test_validate_probes_ok() -> None:
    ok, errs = validate_probes([0, 1, 0], n_calls=2, delta_k=[1, -1], delta_total=0)
    assert ok, errs
    assert errs == []


def test_validate_probes_length_mismatch() -> None:
    ok, errs = validate_probes([0, 1], n_calls=3)
    assert not ok
    assert any("n_calls" in e for e in errs)


def test_validate_probes_bad_score() -> None:
    ok, errs = validate_probes([0, 2])
    assert not ok
    assert any("不在" in e for e in errs)


def test_validate_probes_sum_identity_violation() -> None:
    ok, errs = validate_probes([0, 1, 0], n_calls=2, delta_k=[1, -1], delta_total=5)
    assert not ok
    assert any("delta_total" in e for e in errs)


def test_validate_probes_delta_k_mismatch() -> None:
    ok, errs = validate_probes([0, 1, 0], n_calls=2, delta_k=[1, 1])
    assert not ok
    assert any("delta_k" in e for e in errs)


def test_validate_probes_parse_failure() -> None:
    probes = [{"k": 0, "score": 0}, {"k": 1, "score": 1, "parse_ok": False}]
    ok, errs = validate_probes(probes, n_calls=1)
    assert not ok
    assert any("parse_ok" in e for e in errs)


def test_validate_probes_from_trajectory() -> None:
    from medtaco.utils.schema import Probe, Step, ToolCall, ToolResult, Trajectory

    traj = Trajectory(
        id="t1",
        steps=[
            Step(text="think"),  # 非工具轮
            Step(text="call", call=ToolCall("crop", {}), result=ToolResult(ok=True)),
        ],
        probes=[Probe(k=0, answer="a", score=0), Probe(k=1, answer="b", score=1)],
        delta_k=[1],
        delta_total=1,
    )
    ok, errs = validate_probes(traj)
    assert ok, errs


def test_validate_probes_trajectory_bad_delta_total() -> None:
    from medtaco.utils.schema import Probe, Step, ToolCall, ToolResult, Trajectory

    traj = Trajectory(
        id="t2",
        steps=[Step(text="c", call=ToolCall("crop", {}), result=ToolResult(ok=True))],
        probes=[Probe(k=0, answer="a", score=0), Probe(k=1, answer="b", score=1)],
        delta_k=[1],
        delta_total=0,  # 错
    )
    ok, errs = validate_probes(traj)
    assert not ok
    assert any("delta_total" in e for e in errs)
