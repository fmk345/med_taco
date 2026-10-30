"""medtaco.rl.loss 的单测：两趟 policy loss 的数值正确性与反例。

覆盖：
  1. 手工可算的小例子 —— 断言 loss_1 / loss_2 的**逐通道 token 均值**（容差 1e-6）；
  2. 反例 —— 「拼接总 mask 归一化」的错误实现必须与正确实现显著不同；
  3. mask 全 0 的通道返回 0 且不产生 nan；
  4. torch 分支与 numpy 分支数值一致，且梯度只走当前策略 logprobs；
  5. 形状/键名等输入校验。

不联网、不读数据文件、不依赖 GPU（torch 缺失时自动跳过 torch 用例）。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from medtaco.rl.loss import (
    channel_effective_weights,
    concat_masked_agg_loss,
    two_pass_policy_loss,
    two_pass_policy_loss_np,
)

try:
    import torch
except Exception:  # pragma: no cover - 无 torch 环境
    torch = None

requires_torch = pytest.mark.skipif(torch is None, reason="环境未安装 torch")


# ------------------------------------------------------------------ 手工基准

CLIP_EPS = 0.2
ALPHA_1 = 1.0
ALPHA_2 = 0.15

# 2 条序列、长度 4；old_logprobs 全 0 => ratio = exp(logprob)
LOGPROBS = [
    [0.1, 0.3, -0.1, 0.5],
    [0.1, -0.3, 0.0, 0.0],
]
OLD_LOGPROBS = [
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
]
ADVANTAGES = [1.0, -1.0]           # 第 0 条正样本、第 1 条负样本
MASK_1 = [                          # 通道 1（答案）：样本 0 的 3 个 token
    [1, 1, 1, 0],
    [0, 0, 0, 0],
]
MASK_2 = [                          # 通道 2（过程）：样本 1 的 2 个 token
    [0, 0, 0, 0],
    [1, 1, 0, 0],
]
LOSS_MASKS = {"loss_mask_1": MASK_1, "loss_mask_2": MASK_2}


def test_hand_computed_channel_means():
    """手工算例：逐通道的 token 均值必须与手算一致（容差 1e-6）。

    ratio = exp(logprob - old_logprob) = exp(logprob)（old 全 0）：
      样本 0: r = {1.105171, 1.349859(>1.2 被 clip), 0.904837}, A1 = +1
      样本 1: r = {1.105171, 0.740818(<0.8 被 clip)}, A2 = -1
    通道 1 均值 = -(1.105171 + 1.2 + 0.904837)/3 = -1.0700028
    通道 2 均值 = (1.105171 + 0.8)/2              =  0.9525855   （负 adv 取 min 得较负的一侧）
    """
    out = two_pass_policy_loss_np(
        LOGPROBS, OLD_LOGPROBS, ADVANTAGES, LOSS_MASKS, CLIP_EPS, ALPHA_1, ALPHA_2
    )

    r0, r1, r2 = math.exp(0.1), 1.2, math.exp(-0.1)      # t=1 被 clip 到 1+ε
    r3, r4 = math.exp(0.1), 0.8                          # 负 adv：t=1 被 clip 到 1-ε
    l1 = -(r0 + r1 + r2) / 3
    l2 = -((-r3) + (-r4)) / 2

    assert out["n_tokens_1"] == 3
    assert out["n_tokens_2"] == 2
    assert out["loss_1"] == pytest.approx(l1, abs=1e-6)
    assert out["loss_2"] == pytest.approx(l2, abs=1e-6)
    # 总 loss = α1·L1 + α2·L2（各通道已各自归一化）
    assert out["loss"] == pytest.approx(ALPHA_1 * l1 + ALPHA_2 * l2, abs=1e-6)

    # 具体数值也锁一遍，防止将来改动时「复算同错」
    assert out["loss_1"] == pytest.approx(-1.0700027787, abs=1e-6)
    assert out["loss_2"] == pytest.approx(0.9525854590, abs=1e-6)
    assert out["loss"] == pytest.approx(-0.9271149598, abs=1e-6)
    # 通道 1 只有 t=1 超出上界；通道 2 只有 t=1 超出下界
    assert out["pg_clipfrac_1"] == pytest.approx(1 / 3, abs=1e-6)
    assert out["pg_clipfrac_2"] == pytest.approx(1 / 2, abs=1e-6)
    # 并集 5 个 token 里 clip 触发 2 次
    assert out["pg_clipfrac"] == pytest.approx(2 / 5, abs=1e-6)


# ------------------------------------------------------------------ 反例：拼接归一化


def test_concat_aggregation_is_actually_wrong():
    """反例：按「拼接后的总 mask」归一化会把 α2 的尺度改掉，且差异显著。

    正确：L = α1·mean_1 + α2·mean_2 = -1.0700028 + 0.15×0.9525855 = -0.9271150
    错误：L = [α1·Σ_1 + α2·Σ_2] / (n1+n2) = -2.9242327/5 = -0.5848465
    偏差 0.34，足以让 α2 通道的实际权重从 0.15 掉到 0.06。
    """
    good = two_pass_policy_loss_np(
        LOGPROBS, OLD_LOGPROBS, ADVANTAGES, LOSS_MASKS, CLIP_EPS, ALPHA_1, ALPHA_2
    )
    bad = concat_masked_agg_loss(
        LOGPROBS, OLD_LOGPROBS, ADVANTAGES, LOSS_MASKS, CLIP_EPS, ALPHA_1, ALPHA_2
    )

    assert bad == pytest.approx(-0.5848465397, abs=1e-6)
    assert good["loss"] == pytest.approx(-0.9271149598, abs=1e-6)
    assert abs(good["loss"] - bad) > 0.1, "拼接归一化若与正确实现接近，说明反例没造好"

    # 权重视角更直观：α2 从 0.15 被压到 0.15 * 2/5 = 0.06
    w = channel_effective_weights(LOSS_MASKS, ALPHA_1, ALPHA_2)
    assert w["w_1"] == pytest.approx(1.0 * 3 / 5, abs=1e-9)
    assert w["w_2"] == pytest.approx(0.15 * 2 / 5, abs=1e-9)
    assert w["w_2"] == pytest.approx(0.06, abs=1e-9)

    # 偏差随 n1/n2 变化：把通道 2 的 token 数放大，错误权重随之改变
    mask_2_wide = [[0, 0, 0, 0], [1, 1, 1, 1]]
    w_wide = channel_effective_weights({"loss_mask_1": MASK_1, "loss_mask_2": mask_2_wide}, ALPHA_1, ALPHA_2)
    assert w_wide["w_2"] == pytest.approx(0.15 * 4 / 7, abs=1e-9)
    assert abs(w_wide["w_2"] - w["w_2"]) > 1e-3


def test_concat_aggregation_when_channel_sizes_equal():
    """n1 == n2 时错误实现的通道**比例**恰好不变，但整体尺度被砍半。

    这正是该 bug 的隐蔽之处：只调小 batch、看总 loss 量级的变化发现不了，
    必须检查「每个通道是否除以了自己的 token 数」。
    """
    masks = {"loss_mask_1": [[1, 1, 0, 0], [0, 0, 0, 0]], "loss_mask_2": [[0, 0, 1, 1], [0, 0, 0, 0]]}
    good = two_pass_policy_loss_np(LOGPROBS, OLD_LOGPROBS, ADVANTAGES, masks, CLIP_EPS, ALPHA_1, ALPHA_2)
    bad = concat_masked_agg_loss(LOGPROBS, OLD_LOGPROBS, ADVANTAGES, masks, CLIP_EPS, ALPHA_1, ALPHA_2)
    assert good["n_tokens_1"] == good["n_tokens_2"] == 2
    # 错误实现 = 正确实现 × 1/2（分母 n1+n2=4 而每通道只有 2 个 token）
    assert bad == pytest.approx(0.5 * good["loss"], abs=1e-9)


# ------------------------------------------------------------------ 退化 mask


def test_empty_channel_returns_zero_without_nan():
    """mask 全 0 的通道返回 0，不产生 nan/inf，另一通道照常计算。"""
    masks = {"loss_mask_1": [[1, 0, 0, 0], [1, 0, 0, 0]], "loss_mask_2": [[0] * 4, [0] * 4]}
    out = two_pass_policy_loss_np(LOGPROBS, OLD_LOGPROBS, ADVANTAGES, masks, CLIP_EPS, ALPHA_1, ALPHA_2)

    assert out["loss_2"] == 0.0
    assert out["n_tokens_2"] == 0
    assert out["pg_clipfrac_2"] == 0.0
    assert math.isfinite(out["loss"]) and math.isfinite(out["loss_1"])
    assert not any(isinstance(v, float) and math.isnan(v) for v in out.values())
    assert out["loss"] == pytest.approx(ALPHA_1 * out["loss_1"], abs=1e-12)


def test_both_channels_empty_returns_zero():
    masks = {"loss_mask_1": [[0] * 4, [0] * 4], "loss_mask_2": [[0] * 4, [0] * 4]}
    out = two_pass_policy_loss_np(LOGPROBS, OLD_LOGPROBS, ADVANTAGES, masks, CLIP_EPS, ALPHA_1, ALPHA_2)
    assert out["loss"] == 0.0 and out["loss_1"] == 0.0 and out["loss_2"] == 0.0
    assert out["pg_clipfrac"] == 0.0
    assert out["n_tokens_1"] == out["n_tokens_2"] == 0


# ------------------------------------------------------------------ 语义检查


def test_no_clip_when_ratio_is_one():
    """new == old 时 ratio 恒为 1，loss 退化为 -mean(adv)，且无 clip 触发。"""
    lp = [[-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5]]
    masks = {"loss_mask_1": [[1, 1, 0], [0, 0, 0]], "loss_mask_2": [[0, 0, 0], [1, 0, 0]]}
    out = two_pass_policy_loss_np(lp, lp, [2.0, -3.0], masks, 0.2, 1.0, 0.5)
    assert out["loss_1"] == pytest.approx(-2.0)     # -(2+2)/2
    assert out["loss_2"] == pytest.approx(3.0)      # -(-3)/1
    assert out["loss"] == pytest.approx(-2.0 + 0.5 * 3.0)
    assert out["pg_clipfrac"] == 0.0


def test_negative_advantage_clipping_direction():
    """负 advantage 时上界 clip 会生效：r=0.5, A=-1 -> min(-0.5, -0.8) = -0.8，loss=+0.8。

    （若误用 max 而非 min，会得到 -0.5，惩罚被削弱——这里锁死悲观下界的方向。）
    """
    lp = [[math.log(0.5)]]
    masks = {"loss_mask_1": [[1]], "loss_mask_2": [[0]]}
    out = two_pass_policy_loss_np(lp, [[0.0]], [-1.0], masks, 0.2, 1.0, 1.0)
    assert out["loss_1"] == pytest.approx(0.8, abs=1e-6)
    assert out["pg_clipfrac_1"] == 1.0


def test_advantage_shapes_are_equivalent():
    """dict / tuple / (2,B) ndarray / 共享 [B] 四种写法的结果一致。"""
    masks = LOSS_MASKS
    shared = two_pass_policy_loss_np(
        LOGPROBS, OLD_LOGPROBS, [1.0, -1.0], masks, CLIP_EPS, ALPHA_1, ALPHA_2
    )
    as_dict = two_pass_policy_loss_np(
        LOGPROBS, OLD_LOGPROBS, {"adv_1": [1.0, -1.0], "adv_2": [1.0, -1.0]},
        masks, CLIP_EPS, ALPHA_1, ALPHA_2,
    )
    as_tuple = two_pass_policy_loss_np(
        LOGPROBS, OLD_LOGPROBS, ([1.0, -1.0], [1.0, -1.0]), masks, CLIP_EPS, ALPHA_1, ALPHA_2
    )
    # (2,B,T) 才无歧义：第 0 页给通道 1、第 1 页给通道 2
    as_arr = two_pass_policy_loss_np(
        LOGPROBS, OLD_LOGPROBS, np.broadcast_to(np.array([1.0, -1.0])[:, None, None], (2, 2, 4)),
        masks, CLIP_EPS, ALPHA_1, ALPHA_2,
    )
    for other in (as_dict, as_tuple, as_arr):
        assert other["loss"] == pytest.approx(shared["loss"], abs=1e-12)
        assert other["loss_1"] == pytest.approx(shared["loss_1"], abs=1e-12)


def test_alternative_mask_key_names():
    """mask 允许 loss_mask_1/mask_1/m、loss_mask_2/mask_2/g 三种键名。"""
    base = two_pass_policy_loss_np(
        LOGPROBS, OLD_LOGPROBS, ADVANTAGES, LOSS_MASKS, CLIP_EPS, ALPHA_1, ALPHA_2
    )
    alt = two_pass_policy_loss_np(
        LOGPROBS, OLD_LOGPROBS, ADVANTAGES, {"mask_1": MASK_1, "mask_2": MASK_2}, CLIP_EPS, ALPHA_1, ALPHA_2
    )
    assert alt["loss"] == pytest.approx(base["loss"], abs=1e-12)


# ------------------------------------------------------------------ 输入校验


def test_shape_mismatch_raises():
    # 3 条序列的 old_logprobs 对 2 条的 logprobs
    with pytest.raises(ValueError):
        two_pass_policy_loss_np(
            LOGPROBS, LOGPROBS + [[0.0] * 4], ADVANTAGES, LOSS_MASKS, CLIP_EPS, ALPHA_1, ALPHA_2
        )


def test_mask_shape_mismatch_raises():
    bad = {"loss_mask_1": [[1, 1], [1, 1]], "loss_mask_2": MASK_2}
    with pytest.raises(ValueError):
        two_pass_policy_loss_np(LOGPROBS, OLD_LOGPROBS, ADVANTAGES, bad, CLIP_EPS, ALPHA_1, ALPHA_2)


def test_missing_mask_key_raises():
    with pytest.raises(KeyError):
        two_pass_policy_loss_np(LOGPROBS, OLD_LOGPROBS, ADVANTAGES, {"loss_mask_1": MASK_1}, CLIP_EPS, 1.0, 0.15)


def test_advantage_length_mismatch_raises():
    with pytest.raises(ValueError):
        two_pass_policy_loss_np(LOGPROBS, OLD_LOGPROBS, [1.0, -1.0, 0.5], LOSS_MASKS, CLIP_EPS, ALPHA_1, ALPHA_2)


# ------------------------------------------------------------------ 入口分发与 torch


def test_dispatch_python_path_returns_floats():
    out = two_pass_policy_loss(LOGPROBS, OLD_LOGPROBS, ADVANTAGES, LOSS_MASKS, CLIP_EPS, ALPHA_1, ALPHA_2)
    assert isinstance(out["loss"], float)
    assert isinstance(out["n_tokens_1"], int)


@requires_torch
def test_torch_matches_numpy():
    """torch 分支与 numpy 分支数值一致（含 clipfrac 与 token 数）。"""
    lp = torch.tensor(LOGPROBS, dtype=torch.float64, requires_grad=True)
    old = torch.tensor(OLD_LOGPROBS, dtype=torch.float64)
    masks = {
        "loss_mask_1": torch.tensor(MASK_1, dtype=torch.float64),
        "loss_mask_2": torch.tensor(MASK_2, dtype=torch.float64),
    }
    out_t = two_pass_policy_loss(lp, old, torch.tensor(ADVANTAGES, dtype=torch.float64), masks, CLIP_EPS, ALPHA_1, ALPHA_2)
    out_np = two_pass_policy_loss_np(LOGPROBS, OLD_LOGPROBS, ADVANTAGES, LOSS_MASKS, CLIP_EPS, ALPHA_1, ALPHA_2)

    assert torch.is_tensor(out_t["loss"]) and out_t["loss"].dim() == 0
    assert float(out_t["loss"]) == pytest.approx(out_np["loss"], abs=1e-9)
    assert float(out_t["loss_1"]) == pytest.approx(out_np["loss_1"], abs=1e-9)
    assert float(out_t["loss_2"]) == pytest.approx(out_np["loss_2"], abs=1e-9)
    assert out_t["pg_clipfrac"] == pytest.approx(out_np["pg_clipfrac"], abs=1e-9)
    assert out_t["n_tokens_1"] == out_np["n_tokens_1"]
    assert out_t["n_tokens_2"] == out_np["n_tokens_2"]


@requires_torch
def test_torch_gradient_flow_and_unmasked_tokens_get_zero_grad():
    """梯度只经当前策略 logprobs 回传；mask 外的 token 梯度必须为 0。"""
    lp = torch.tensor(LOGPROBS, dtype=torch.float64, requires_grad=True)
    old = torch.tensor(OLD_LOGPROBS, dtype=torch.float64)
    masks = {
        "loss_mask_1": torch.tensor(MASK_1, dtype=torch.float64),
        "loss_mask_2": torch.tensor(MASK_2, dtype=torch.float64),
    }
    out = two_pass_policy_loss(lp, old, torch.tensor(ADVANTAGES, dtype=torch.float64), masks, CLIP_EPS, ALPHA_1, ALPHA_2)
    out["loss"].backward()

    assert lp.grad is not None
    grad = lp.grad.detach().numpy()

    # 两通道 mask 之外一律为 0：样本 0 的 t=3、样本 1 的 t=2,3
    assert np.allclose(grad[0, 3], 0.0, atol=1e-12)
    assert np.allclose(grad[1, 2:], 0.0, atol=1e-12)

    # 解析对拍 d loss/d logp = -α_c·mask_c/count_c · dsur/dr · r
    #   样本 0, t=0：α1=1, n1=3, r=e^0.1 未 clip -> -e^0.1/3
    assert grad[0, 0] == pytest.approx(-ALPHA_1 * math.exp(0.1) / 3, rel=1e-9)
    #   样本 0, t=1：r=e^0.3=1.3499 > 1.2，走 clamp 分支，梯度为 0
    assert grad[0, 1] == pytest.approx(0.0, abs=1e-12)
    #   样本 1, t=0：A2=-1、r=e^0.1 未 clip -> -α2·(-1)·r/2 = +α2·e^0.1/2
    assert grad[1, 0] == pytest.approx(ALPHA_2 * math.exp(0.1) / 2, rel=1e-9)
    #   样本 1, t=1：r=e^-0.3=0.7408 < 0.8，走 clamp 分支，梯度为 0
    assert grad[1, 1] == pytest.approx(0.0, abs=1e-12)


@requires_torch
def test_torch_empty_channel_no_nan_and_grad_finite():
    masks = {
        "loss_mask_1": torch.tensor(MASK_1, dtype=torch.float64),
        "loss_mask_2": torch.zeros((2, 4), dtype=torch.float64),
    }
    lp = torch.tensor(LOGPROBS, dtype=torch.float64, requires_grad=True)
    out = two_pass_policy_loss(lp, torch.zeros_like(lp), torch.tensor(ADVANTAGES, dtype=torch.float64),
                               masks, CLIP_EPS, ALPHA_1, ALPHA_2)
    assert float(out["loss_2"]) == 0.0
    out["loss"].backward()
    assert torch.isfinite(lp.grad).all()
