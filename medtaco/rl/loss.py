"""两趟 policy loss（防 `agg_loss` 按拼接总 mask 归一化改掉 α2 尺度）。

背景（PROPOSAL §7.3）
----------------------
TACO 的目标是两条通道的**加权和**：

    L = α1 · L_GRPO(m ⊙ A1) + α2 · L_GRPO(g · A2)

其中 m / g 是两张 token 级 0/1 掩码（`advantage.build_loss_masks` 产出），
每个通道的 L_GRPO 都是「该通道自己的 token 均值」，即

    L_c = - Σ_t mask_c[t] · min(r_t A_c, clip(r_t, 1-ε, 1+ε) A_c) / Σ_t mask_c[t]

如果像 verl 的 `agg_loss(loss_mat, loss_mask, loss_agg_mode="token-mean")` 那样，
把两个通道的 `loss_mat` 和 `loss_mask` **先拼接/相加再一次性归一化**，分母就变成
n1 + n2，等价于把两个通道的权重偷偷改成 α1·n1/(n1+n2) 与 α2·n2/(n1+n2)——
α2 的尺度被数据相关的 n2/n1 比例污染了。这就是「两趟」的由来。

本模块因此对每个通道**单独**归约，再在通道之间做固定权重求和。
`concat_masked_agg_loss()` 保留了那个错误实现，仅用于反例单测与对照。

两种输入
--------
1. torch：`logprobs/old_logprobs/advantages/loss_masks` 为 `torch.Tensor`，
   返回的 `loss/loss_1/loss_2` 是带梯度的 0-dim Tensor（可直接 backward）。
2. 纯 python / numpy：`list[list[float]]` / `list[float]` / `list[list[int]]`，
   返回 python float，方便无 GPU 环境下单测。
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

try:  # torch 可选
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover - 环境无 torch 时走纯 python 分支
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

__all__ = [
    "two_pass_policy_loss",
    "two_pass_policy_loss_np",
    "concat_masked_agg_loss",
]

# 通道 1 = 答案/结果通道（mask m, advantage A1, 权重 α1）
# 通道 2 = 过程通道（mask g, advantage A2, 权重 α2）
_MASK_KEYS: Tuple[Tuple[str, ...], Tuple[str, ...]] = (
    ("loss_mask_1", "mask_1", "m"),
    ("loss_mask_2", "mask_2", "g"),
)
_ADV_KEYS: Tuple[Tuple[str, ...], Tuple[str, ...]] = (("adv_1", "A_1"), ("adv_2", "A_2"))


# ------------------------------------------------------------------ 输入规整


def _pick_from_dict(d: Dict[str, Any], keys: Sequence[str], what: str) -> Any:
    """按候选键顺序从一个 dict 里取值；全缺则报错。"""
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"loss_masks 缺少 {what}，期望任一键: {tuple(keys)}, 实际: {tuple(d)}")


def _as_2d_float(x: Any) -> np.ndarray:
    """把 [B,T] 的 nested list / ndarray 规整成 float64 的 2-D 数组。"""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:  # 单条序列，视为 B=1
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"logprobs 必须是 2 维 [B,T]，收到 shape={arr.shape}")
    return arr


def _as_2d_mask(x: Any, shape: Tuple[int, int], name: str) -> np.ndarray:
    """把 mask 规整成与 logprobs 同形状的 float64 0/1 数组。"""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape != shape:
        raise ValueError(f"{name} 形状 {arr.shape} 与 logprobs {shape} 不一致")
    # 契约是 0/1 掩码；用 >0 归一，兼容传入 bool / 0.0-1.0 的情况
    return (arr > 0).astype(np.float64)


def _as_advantage(x: Any, b: int, t: int, name: str) -> np.ndarray:
    """把 advantage 规整成可广播到 [B,T] 的 float64 数组。

    支持 `[B]`（每条序列一个 advantage，最常见）、`[B,1]`、`[B,T]`（token 级）。
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        return np.full((b, t), float(arr), dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != b:
            raise ValueError(f"{name} 长度 {arr.shape[0]} 与 batch 大小 {b} 不一致")
        return np.broadcast_to(arr[:, None], (b, t)).copy()
    if arr.ndim == 2:
        if arr.shape[0] != b:
            raise ValueError(f"{name} 首维 {arr.shape[0]} 与 batch 大小 {b} 不一致")
        if arr.shape[1] == 1:
            return np.broadcast_to(arr, (b, t)).copy()
        if arr.shape[1] != t:
            raise ValueError(f"{name} 形状 {arr.shape} 无法广播到 {(b, t)}")
        return arr
    raise ValueError(f"{name} 维度过高: shape={arr.shape}")


def _split_advantages(advantages: Any, b: int, t: int) -> Tuple[np.ndarray, np.ndarray]:
    """拆出两个通道各自的 advantage。

    允许四种写法：
      - dict：`{'adv_1':..., 'adv_2':...}`（`advantage.build_loss_masks` 的产物）；
      - tuple(len=2)：(adv_1, adv_2)；
      - ndarray shape (2,B) / (2,B,T)：第 0 行给通道 1、第 1 行给通道 2；
      - 其它：两通道共用同一个 advantage。
    """
    if isinstance(advantages, dict):
        a1 = _pick_from_dict(advantages, _ADV_KEYS[0], "adv_1")
        a2 = _pick_from_dict(advantages, _ADV_KEYS[1], "adv_2")
        return _as_advantage(a1, b, t, "adv_1"), _as_advantage(a2, b, t, "adv_2")

    if isinstance(advantages, tuple) and len(advantages) == 2:
        # 形如 ([B], [B]) 的逐通道写法（list 不参与，避免与 [B] 的二元素 list 混淆）
        a1, a2 = advantages
        return _as_advantage(a1, b, t, "adv_1"), _as_advantage(a2, b, t, "adv_2")

    arr = np.asarray(advantages, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] == 2:
        return _as_advantage(arr[0], b, t, "adv_1"), _as_advantage(arr[1], b, t, "adv_2")

    shared = _as_advantage(advantages, b, t, "advantages")
    return shared, shared.copy()


def _split_masks_np(loss_masks: Dict[str, Any], shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """从一个 dict 里取出两张同形状的 0/1 掩码。"""
    if not isinstance(loss_masks, dict):
        raise TypeError("loss_masks 必须是 dict，含两张同形状 mask")
    m1 = _as_2d_mask(_pick_from_dict(loss_masks, _MASK_KEYS[0], "通道1 mask"), shape, "loss_mask_1")
    m2 = _as_2d_mask(_pick_from_dict(loss_masks, _MASK_KEYS[1], "通道2 mask"), shape, "loss_mask_2")
    return m1, m2


# ------------------------------------------------------------------ 核心公式


def _channel_loss_np(
    ratio: np.ndarray,
    adv: np.ndarray,
    mask: np.ndarray,
    clip_eps: float,
) -> Tuple[float, float, int]:
    """单通道 loss：返回 (loss_c, clipfrac_c, n_tokens_c)。

    loss_c = - Σ mask·min(rA, clip(r)A) / Σ mask；mask 全 0 时返回 0（不发散、不 nan）。
    """
    n_tok = float(mask.sum())
    if n_tok <= 0:
        return 0.0, 0.0, 0

    unclipped = ratio * adv
    clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    surrogate = np.minimum(unclipped, clipped)  # 悲观下界，与 PPO 一致
    loss_c = -float((mask * surrogate).sum() / n_tok)

    # clipfrac 定义：ratio 落在 [1-ε, 1+ε] 之外即为「clip 被触发」
    clip_hit = (np.abs(ratio - 1.0) > clip_eps).astype(np.float64)
    clipfrac_c = float((mask * clip_hit).sum() / n_tok)
    return loss_c, clipfrac_c, int(round(n_tok))


def two_pass_policy_loss_np(
    logprobs: Any,
    old_logprobs: Any,
    advantages: Any,
    loss_masks: Dict[str, Any],
    clip_eps: float,
    alpha_1: float,
    alpha_2: float,
) -> Dict[str, Any]:
    """纯 python / numpy 版本的两趟 policy loss（无 torch 依赖，单测主力）。

    与 `two_pass_policy_loss` 数值等价，返回的标量都是 python float。

    Args:
        logprobs: 当前策略的 token logprob，`[B,T]` nested list 或 ndarray。
        old_logprobs: 采样时策略的 token logprob，同形状。
        advantages: `[B]` / `[B,T]` / dict{'adv_1','adv_2'} / tuple(adv_1, adv_2)。
        loss_masks: 必须含 `loss_mask_1` 与 `loss_mask_2`（也接受 mask_1/m、mask_2/g），
            两张 0/1 掩码与 logprobs 同形状。
        clip_eps: PPO clip ε（TACO 原论文未给，取 0.2）。
        alpha_1: 通道 1（答案）权重，默认 1.0。
        alpha_2: 通道 2（过程）权重，默认 0.15。

    Returns:
        dict:
            loss / loss_1 / loss_2 —— 加权总 loss 与两通道各自的 token 均值 loss
            pg_clipfrac —— 两通道 mask 并集上，ratio 超出 clip 区间的 token 占比
            pg_clipfrac_1 / pg_clipfrac_2 —— 各通道自己的 clipfrac
            n_tokens_1 / n_tokens_2 —— 各通道有效 token 数
    """
    lp = _as_2d_float(logprobs)
    old = _as_2d_float(old_logprobs)
    if lp.shape != old.shape:
        raise ValueError(f"logprobs {lp.shape} 与 old_logprobs {old.shape} 形状不一致")
    b, t = lp.shape

    m1, m2 = _split_masks_np(loss_masks, (b, t))
    a1, a2 = _split_advantages(advantages, b, t)

    # 数值上用 log 差分再 exp，避免 (p/q) 直接相除的精度损失
    ratio = np.exp(np.clip(lp - old, -50.0, 50.0))

    loss_1, clipfrac_1, n_1 = _channel_loss_np(ratio, a1, m1, clip_eps)
    loss_2, clipfrac_2, n_2 = _channel_loss_np(ratio, a2, m2, clip_eps)

    # 关键：两个通道各自归一化后再加权，绝不用 n1+n2 当分母
    loss = float(alpha_1) * loss_1 + float(alpha_2) * loss_2

    union = ((m1 + m2) > 0).astype(np.float64)
    n_union = float(union.sum())
    if n_union > 0:
        clip_hit = (np.abs(ratio - 1.0) > clip_eps).astype(np.float64)
        pg_clipfrac = float((union * clip_hit).sum() / n_union)
    else:
        pg_clipfrac = 0.0

    return {
        "loss": float(loss),
        "loss_1": float(loss_1),
        "loss_2": float(loss_2),
        "pg_clipfrac": float(pg_clipfrac),
        "pg_clipfrac_1": float(clipfrac_1),
        "pg_clipfrac_2": float(clipfrac_2),
        "n_tokens_1": int(n_1),
        "n_tokens_2": int(n_2),
    }


# ------------------------------------------------------------------ torch 版本


def _torch_channel_loss(
    ratio: "torch.Tensor",
    adv: "torch.Tensor",
    mask: "torch.Tensor",
    clip_eps: float,
) -> Tuple["torch.Tensor", float, int]:
    """torch 单通道 loss：保持计算图；mask 全 0 时 loss 恒为 0 且梯度有限。"""
    n_tok = float(mask.sum().item())
    if n_tok <= 0:
        # 乘 0 而不是 return 0.0，保证丢失梯度时不会出现 None 型依赖
        zero = (ratio * 0.0).sum()
        return zero, 0.0, 0

    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    surrogate = torch.minimum(unclipped, clipped)
    loss_c = -(mask * surrogate).sum() / n_tok

    with torch.no_grad():
        clip_hit = (ratio - 1.0).abs() > clip_eps
        clipfrac_c = float(((mask > 0) & clip_hit).sum().item() / n_tok)
    return loss_c, clipfrac_c, int(round(n_tok))


def _two_pass_policy_loss_torch(
    logprobs: "torch.Tensor",
    old_logprobs: "torch.Tensor",
    advantages: Any,
    loss_masks: Dict[str, Any],
    clip_eps: float,
    alpha_1: float,
    alpha_2: float,
) -> Dict[str, Any]:
    """torch 分支：loss/loss_1/loss_2 为带梯度的 0-dim Tensor。"""
    lp = logprobs if logprobs.dim() == 2 else logprobs.view(1, -1)
    old = old_logprobs if old_logprobs.dim() == 2 else old_logprobs.view(1, -1)
    if lp.shape != old.shape:
        raise ValueError(f"logprobs {tuple(lp.shape)} 与 old_logprobs {tuple(old.shape)} 形状不一致")
    b, t = int(lp.shape[0]), int(lp.shape[1])

    m1 = _pick_from_dict(loss_masks, _MASK_KEYS[0], "通道1 mask")
    m2 = _pick_from_dict(loss_masks, _MASK_KEYS[1], "通道2 mask")
    if not torch.is_tensor(m1):
        m1 = torch.as_tensor(m1, device=lp.device)
    if not torch.is_tensor(m2):
        m2 = torch.as_tensor(m2, device=lp.device)
    m1 = (m1.reshape(lp.shape) > 0).to(dtype=lp.dtype)
    m2 = (m2.reshape(lp.shape) > 0).to(dtype=lp.dtype)

    def _adv_to_tensor(a: Any, name: str) -> "torch.Tensor":
        if not torch.is_tensor(a):
            a = torch.as_tensor(a, device=lp.device, dtype=lp.dtype)
        a = a.to(device=lp.device, dtype=lp.dtype)
        if a.dim() == 0:
            return a.expand(b, t)
        if a.dim() == 1:
            return a.view(b, 1).expand(b, t)
        if a.dim() == 2:
            if a.shape[0] != b:
                raise ValueError(f"{name} 首维 {int(a.shape[0])} 与 batch 大小 {b} 不一致")
            return a.expand(b, t) if a.shape[1] == 1 else a
        raise ValueError(f"{name} 维度过高: shape={tuple(a.shape)}")

    if isinstance(advantages, dict):
        a1 = _adv_to_tensor(_pick_from_dict(advantages, _ADV_KEYS[0], "adv_1"), "adv_1")
        a2 = _adv_to_tensor(_pick_from_dict(advantages, _ADV_KEYS[1], "adv_2"), "adv_2")
    elif isinstance(advantages, (tuple, list)) and len(advantages) == 2 and any(
        isinstance(v, (list, tuple)) or (torch.is_tensor(v) and v.dim() >= 1) or isinstance(v, np.ndarray)
        for v in advantages
    ):
        a1 = _adv_to_tensor(advantages[0], "adv_1")
        a2 = _adv_to_tensor(advantages[1], "adv_2")
    elif torch.is_tensor(advantages) and advantages.dim() == 3 and int(advantages.shape[0]) == 2:
        a1 = _adv_to_tensor(advantages[0], "adv_1")
        a2 = _adv_to_tensor(advantages[1], "adv_2")
    else:
        a1 = _adv_to_tensor(advantages, "advantages")
        a2 = a1

    # old_logprobs 是常量基线，detach 掉；梯度只经当前策略的 logprobs 回传
    ratio = torch.exp(torch.clamp(lp - old.detach(), -50.0, 50.0))

    loss_1, clipfrac_1, n_1 = _torch_channel_loss(ratio, a1, m1, clip_eps)
    loss_2, clipfrac_2, n_2 = _torch_channel_loss(ratio, a2, m2, clip_eps)

    loss = float(alpha_1) * loss_1 + float(alpha_2) * loss_2

    union = ((m1 + m2) > 0)
    n_union = float(union.sum().item())
    if n_union > 0:
        with torch.no_grad():
            clip_hit = (ratio.detach() - 1.0).abs() > clip_eps
            pg_clipfrac = float((union & clip_hit).sum().item() / n_union)
    else:
        pg_clipfrac = 0.0

    return {
        "loss": loss,
        "loss_1": loss_1,
        "loss_2": loss_2,
        "pg_clipfrac": pg_clipfrac,
        "pg_clipfrac_1": clipfrac_1,
        "pg_clipfrac_2": clipfrac_2,
        "n_tokens_1": int(n_1),
        "n_tokens_2": int(n_2),
    }


# ------------------------------------------------------------------ 统一入口


def _looks_like_torch(*objs: Any) -> bool:
    """判断输入里有没有 torch.Tensor（无 torch 环境恒为 False）。"""
    if not _HAS_TORCH:
        return False
    return any(torch.is_tensor(o) for o in objs)


def two_pass_policy_loss(
    logprobs: Any,
    old_logprobs: Any,
    advantages: Any,
    loss_masks: Dict[str, Any],
    clip_eps: float,
    alpha_1: float,
    alpha_2: float,
) -> Dict[str, Any]:
    """两趟 policy loss：两个通道各自按自己的 mask 求 token 均值，再乘 α 相加。

    签名与语义见 `two_pass_policy_loss_np`；输入是 torch.Tensor 时自动走 torch 分支
    （返回带梯度的 0-dim Tensor），否则走 numpy 分支（返回 python float）。

    Raises:
        ValueError: 形状不一致、advantage 长度对不上 batch。
        KeyError: loss_masks / advantages dict 缺少必需键。
    """
    torch_inputs: List[Any] = [logprobs, old_logprobs]
    if isinstance(advantages, dict):
        torch_inputs.extend(advantages.values())
    else:
        torch_inputs.append(advantages)
    if isinstance(loss_masks, dict):
        torch_inputs.extend(loss_masks.values())

    if _looks_like_torch(*torch_inputs):
        return _two_pass_policy_loss_torch(
            logprobs, old_logprobs, advantages, loss_masks, clip_eps, alpha_1, alpha_2
        )
    return two_pass_policy_loss_np(
        logprobs, old_logprobs, advantages, loss_masks, clip_eps, alpha_1, alpha_2
    )


# ------------------------------------------------------------------ 反例（错误实现）


def concat_masked_agg_loss(
    logprobs: Any,
    old_logprobs: Any,
    advantages: Any,
    loss_masks: Dict[str, Any],
    clip_eps: float,
    alpha_1: float,
    alpha_2: float,
) -> float:
    """**错误**实现：verl `agg_loss` 风格的「拼接总 mask 归一化」。

    仅用于反例单测与对照，**绝不能在训练里用**。

    它把两个通道的 target loss 和 mask 先相加，再除以 n1 + n2：

        L_wrong = Σ_c α_c Σ_t m_c·sur_c / (n1 + n2)

    展开后通道 c 的实际权重是 α_c · n_c / (n1 + n2)，而不是 α_c——
    α2 的尺度被 n2/(n1+n2) 这个与数据相关的比例悄悄缩放，
    两个通道 token 数差异越大，偏得越远。
    """
    lp = _as_2d_float(logprobs)
    old = _as_2d_float(old_logprobs)
    b, t = lp.shape
    m1, m2 = _split_masks_np(loss_masks, (b, t))
    a1, a2 = _split_advantages(advantages, b, t)
    ratio = np.exp(np.clip(lp - old, -50.0, 50.0))

    def _sur(a: np.ndarray) -> np.ndarray:
        unclipped = ratio * a
        clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * a
        return np.minimum(unclipped, clipped)

    n_total = float(m1.sum() + m2.sum())
    if n_total <= 0:
        return 0.0
    num = float(alpha_1) * float((m1 * _sur(a1)).sum()) + float(alpha_2) * float((m2 * _sur(a2)).sum())
    return -num / n_total


def channel_effective_weights(
    loss_masks: Dict[str, Any],
    alpha_1: float,
    alpha_2: float,
) -> Dict[str, float]:
    """诊断辅助：给出「拼接总 mask 归一化」下两通道的**实际**权重。

    正确实现里恒为 (α1, α2)；错误实现里是 (α1·n1/(n1+n2), α2·n2/(n1+n2))。
    打印这对数值就能直观看到 α2 被改了多少。
    """
    m1 = np.asarray(_pick_from_dict(loss_masks, _MASK_KEYS[0], "通道1 mask"), dtype=np.float64)
    m2 = np.asarray(_pick_from_dict(loss_masks, _MASK_KEYS[1], "通道2 mask"), dtype=np.float64)
    n1, n2 = float((m1 > 0).sum()), float((m2 > 0).sum())
    tot = n1 + n2
    if tot <= 0:
        return {"alpha_1": float(alpha_1), "alpha_2": float(alpha_2), "w_1": 0.0, "w_2": 0.0}
    return {
        "alpha_1": float(alpha_1),
        "alpha_2": float(alpha_2),
        "w_1": float(alpha_1) * n1 / tot,
        "w_2": float(alpha_2) * n2 / tot,
    }
