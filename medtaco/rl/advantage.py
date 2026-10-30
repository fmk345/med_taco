"""分级前缀探针 + OGAR 分段路由 + group 归一化（PROPOSAL §7.1 / §7.3）。

职责边界（本文件只做三件事，不碰 I/O、不碰 torch）：
  1. 由探针分数序列 ``a_0..a_K`` 算逐次调用的差分 ``Δ_k``（§7.1）；
  2. 按多轮展开的 OGAR 规则生成 token 级 0/1 掩码 ``m[t]``（§7.3），
     并附带两条 loss 通道各自的掩码 / advantage（供 rl/loss.py 两趟计算）；
  3. group 内 advantage 归一化，含零方差跳过 / eps 下限两种工程处理。

记号约定（与 INTERFACES.md 的 CONTRACT 段一致）：
  - ``T1  = segment.t1 = (start, end)``，左闭右开，是第一次调用之前的那段思考；
  - ``calls[k] = (obs_start, obs_end, t2_start, t2_end)``；obs 是观测 token（返回图 /
    沙箱输出），**永远 mask**；t2 是第 k 次调用之后的思考与调用 token；
  - ``C_k ∪ T_{k+1}`` 在多轮展开里就是 ``obs_k ∪ t2_k`` 外加后续那轮思考，本实现按
    契约把两者都算进第 k 次调用的信用区间（t2 段已包含下一轮 T 的全部 token）；
  - 某段不适用时填 ``(-1, -1)``，本模块一律当成"该区间不存在"跳过。

注意：本模块刻意**不** import torch，也不做任何张量运算；所有输出都是纯 python
list，由调用方（verl 侧 adv_est / loss.py）自行转张量。
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# 区间类型：[start, end)
Interval = Tuple[int, int]
# 一次调用的四元组：(obs_start, obs_end, t2_start, t2_end)
CallSpan = Tuple[int, int, int, int]

# 合法取值集合（探针分数即 R_ORM，见 §7.2.1）
VALID_SCORES: Tuple[int, ...] = (-1, 0, 1)
MODES: Tuple[str, ...] = ("paper", "always")
POLICIES: Tuple[str, ...] = ("skip_group", "eps_floor")


# ---------------------------------------------------------------- 数据结构


@dataclass
class TurnMasks:
    """一次 OGAR 路由的结果。

    token_mask:   长度 = 轨迹 token 数，取值 0/1。1 表示该 token 参与 outcome 通道。
    call_indices: 长度 = 调用次数 K，第 k 次调用在 token 序列中的位置
                  （取观测起点 obs_start；缺失时退化为 t2_start；都没有填 -1）。
    """

    token_mask: List[int]
    call_indices: List[int]


@dataclass
class Segment:
    """轨迹的分段信息（契约结构，勿改名）。

    t1:    (start, end)，第一次工具调用之前的思考 token 区间。
    calls: K 个四元组，每次调用的 (obs_start, obs_end, t2_start, t2_end)。
    """

    t1: Interval
    calls: List[CallSpan] = field(default_factory=list)


# ---------------------------------------------------------------- 区间小工具


def _clean_interval(iv: Any) -> Optional[Interval]:
    """规范化一个左闭右开区间；不合法（None / (-1,-1) / 空区间）返回 None。"""
    if iv is None:
        return None
    try:
        s, e = int(iv[0]), int(iv[1])
    except (TypeError, ValueError, IndexError):
        return None
    if s < 0 or e < 0 or e <= s:
        return None
    return (s, e)


def _call_span(c: Sequence[int]) -> Tuple[Optional[Interval], Optional[Interval]]:
    """把 calls[k] 四元组拆成 (obs 区间, t2 区间)，各自可能为 None。"""
    c = list(c)
    while len(c) < 4:  # 容错：短元组用 -1 补齐
        c.append(-1)
    return _clean_interval((c[0], c[1])), _clean_interval((c[2], c[3]))


def _fill(mask: List[int], iv: Optional[Interval], value: int) -> None:
    """把 mask[iv] 段整体置为 value（越界部分自动裁掉）。"""
    if iv is None:
        return
    s, e = max(0, iv[0]), min(len(mask), iv[1])
    for t in range(s, e):
        mask[t] = value


def _as_segments(segments: Any) -> List[Segment]:
    """允许传单个 Segment 或 Segment 列表。"""
    if isinstance(segments, Segment):
        return [segments]
    return list(segments)


def flat_calls(segments: Any) -> List[CallSpan]:
    """把多个 Segment 的 calls 按顺序拍平（跨段调用编号连续）。"""
    out: List[CallSpan] = []
    for seg in _as_segments(segments):
        out.extend(list(seg.calls or []))
    return out


def token_length(segments: Any) -> int:
    """由所有段的右端点推出轨迹 token 总数（不依赖外部传长度）。"""
    n = 0
    for seg in _as_segments(segments):
        iv = _clean_interval(seg.t1)
        if iv is not None:
            n = max(n, iv[1])
        for c in (seg.calls or []):
            obs, t2 = _call_span(c)
            for iv2 in (obs, t2):
                if iv2 is not None:
                    n = max(n, iv2[1])
    return n


def _check_scores(scores: Sequence[int]) -> List[int]:
    """校验探针分数 ∈ {-1,0,1}，返回 int 列表（允许 numpy 整数）。"""
    out: List[int] = []
    for i, s in enumerate(scores):
        if isinstance(s, bool) or not isinstance(s, numbers.Integral):
            raise ValueError(f"探针分数必须是 int，第 {i} 个是 {s!r}")
        if int(s) not in VALID_SCORES:
            raise ValueError(f"探针分数必须在 {VALID_SCORES} 中，第 {i} 个是 {s}")
        out.append(int(s))
    return out


# ---------------------------------------------------------------- Δ_k


def staged_deltas(probe_scores: Sequence[int]) -> List[int]:
    """a_0..a_K -> Δ_1..Δ_K，其中 Δ_k = a_k - a_{k-1}。

    长度 = K = len(probe_scores) - 1。分数必须 ∈ {-1,0,1}，因此 Δ_k ∈ {-2..2}。
    恒等式：``sum(staged_deltas(a)) == a[-1] - a[0]``，供 validate_probes 交叉校验。
    """
    scores = _check_scores(probe_scores)
    if not scores:
        raise ValueError("probe_scores 不能为空（至少要有 a_0）")
    return [scores[k] - scores[k - 1] for k in range(1, len(scores))]


# ---------------------------------------------------------------- OGAR 掩码


def ogar_masks(
    probe_scores: Sequence[int],
    segments: Any,
    mode: str = "paper",
) -> TurnMasks:
    """按 PROPOSAL §7.3 生成 token 级 0/1 掩码。

    paper 模式（默认，论文口径）::

        m[t] = 1  若 t ∈ T1 且 Δ_total >= 0
               1  若 t ∈ obs_k ∪ t2_k 且 Δ_k != 0     （对每个 k 独立判断）
               0  其余
        最后强制 obs 区间为 0（观测 token 永远不给信用，即便上面命中）

    always 模式（消融）：只保留 T1 逻辑且**无条件**置 1（不看 Δ_total），
    不再做逐 call 的 Δ_k 路由；obs 区间仍然为 0。用于回答
    "信用只落在首段思考上、完全不用分级路由"能掉多少点。

    Args:
        probe_scores: a_0..a_K，长度必须是 K+1（K = 调用总数）。
        segments:     Segment 或 Segment 列表；多个 Segment 的 calls 按顺序拍平编号。
        mode:         'paper' | 'always'。

    Returns:
        TurnMasks，token_mask 长度等于轨迹 token 数（由段右端点推出）。
    """
    if mode not in MODES:
        raise ValueError(f"mode 必须是 {MODES} 之一，收到 {mode!r}")
    scores = _check_scores(probe_scores)
    if not scores:
        raise ValueError("probe_scores 不能为空（至少要有 a_0）")

    segs = _as_segments(segments)
    if not segs:
        raise ValueError("segments 不能为空")

    deltas = staged_deltas(scores)
    calls = flat_calls(segs)
    if len(scores) != len(calls) + 1:
        raise ValueError(
            f"探针数 {len(scores)} 与调用数 {len(calls)} 不匹配，"
            f"应满足 len(probe_scores) == n_calls + 1"
        )

    n = token_length(segs)
    mask: List[int] = [0] * n
    delta_total = scores[-1] - scores[0]

    t1_ivals: List[Optional[Interval]] = [_clean_interval(seg.t1) for seg in segs]

    if mode == "paper":
        # 规则 1：T1 只在 Δ_total>=0 时拿 outcome 信用
        if delta_total >= 0:
            for iv in t1_ivals:
                _fill(mask, iv, 1)
        # 规则 2：逐 call —— Δ_k != 0 时该次调用之后的 obs ∪ t2 段拿信用
        for k, d in enumerate(deltas):
            if d == 0:
                continue
            obs, t2 = _call_span(calls[k])
            _fill(mask, obs, 1)
            _fill(mask, t2, 1)
    else:  # mode == "always"：消融，只走 T1 且无条件置 1
        for iv in t1_ivals:
            _fill(mask, iv, 1)

    # 硬约束：观测 token 永远 mask（放在最后，保证覆盖前面任何置 1）
    for c in calls:
        obs, _ = _call_span(c)
        _fill(mask, obs, 0)

    call_indices: List[int] = []
    for c in calls:
        obs, t2 = _call_span(c)
        if obs is not None:
            call_indices.append(obs[0])
        elif t2 is not None:
            call_indices.append(t2[0])
        else:
            call_indices.append(-1)

    return TurnMasks(token_mask=mask, call_indices=call_indices)


# ---------------------------------------------------------------- group 归一化


def group_advantage(
    values: Sequence[float],
    std_eps: float,
    policy: str,
) -> Optional[List[float]]:
    """组内归一化 A_i = (v_i - mean) / std。

    Args:
        values:   同组（G 条 rollout）的标量，如 Δ_total 或最终 R。
        std_eps:  std 下限阈值。
        policy:   'skip_group'：std <= std_eps 时返回 None（该组整体跳过）；
                  'eps_floor'：分母取 max(std, std_eps)，零方差时返回全 0。

    Returns:
        advantage 列表；'skip_group' 判零方差时为 None。

    注意：零方差的判定必须覆盖"组内所有值完全相同"（std 恰为 0）以及
    "组只有一个样本"（std 无定义，同样按 0 处理）两种情形。
    """
    if policy not in POLICIES:
        raise ValueError(f"policy 必须是 {POLICIES} 之一，收到 {policy!r}")
    eps = float(std_eps)
    vals = [float(v) for v in values]
    n = len(vals)

    if n == 0:
        return None if policy == "skip_group" else []

    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n  # 总体方差（GRPO 口径，ddof=0）
    std = math.sqrt(max(var, 0.0))

    if policy == "skip_group":
        # std 恰好为 0 / 数值上落到 eps 以下，都视为零方差
        if std <= eps:
            return None
        return [(v - mean) / std for v in vals]

    # eps_floor：分母有下限，零方差时分子也是 0，自然输出全 0
    denom = max(std, eps)
    if denom <= 0.0:
        return [0.0] * n
    return [(v - mean) / denom for v in vals]


# ---------------------------------------------------------------- 两通道掩码组装


def _covered_mask(segments: Any) -> List[int]:
    """"模型自己生成的 token"掩码：T1 ∪ 所有 t2，不含 obs。

    process 通道（A2）只作用在这些 token 上——观测是环境给的，不该被优化。
    """
    n = token_length(segments)
    mask = [0] * n
    for seg in _as_segments(segments):
        _fill(mask, _clean_interval(seg.t1), 1)
        for c in (seg.calls or []):
            _, t2 = _call_span(c)
            _fill(mask, t2, 1)
    return mask


def build_loss_masks(
    segments_list: Sequence[Any],
    probe_scores_list: Sequence[Sequence[int]],
    final_scores: Optional[Sequence[float]] = None,
    std_eps: Optional[float] = None,
    zero_var_policy: Optional[str] = None,
    mode: Optional[str] = None,
    cfg: Any = None,
) -> Dict[str, Any]:
    """为一个 **group**（同题的 G 条 rollout）组装两条 loss 通道各自的东西。

    对应 ``L = α1·L_GRPO(m ⊙ A1) + α2·L_GRPO(g·A2)``：
      - 通道 1（outcome，A1）：advantage = group 归一化后的最终探针分 R(a_K)；
        token mask = ``ogar_masks`` 的 m[t]。
      - 通道 2（process，A2）：advantage = group 归一化后的 Δ_total；
        token mask = g · covered，g = 1[Δ_total >= 0]，covered 是 T1 ∪ 所有 t2（obs 恒 0）。

    两条通道的 mask 必须分开保存、分别求 token 均值（见 rl/loss.py 的两趟计算），
    绝不能拼起来用总 mask 归一化——否则 α2 的尺度会被悄悄改掉。

    Args:
        segments_list:     长度 G，每条是该 rollout 的 Segment 或 Segment 列表。
        probe_scores_list: 长度 G，每条是 a_0..a_K。
        final_scores:      可选，长度 G 的最终分；缺省取每条 probe_scores 的末项。
        std_eps / zero_var_policy / mode: 缺省时先看 cfg.rl.advantage，再退回
                           1e-6 / 'skip_group' / 'paper'。
        cfg:               medtaco.utils.config.Cfg（可传 None）。

    Returns:
        dict，键：
          'loss_mask_1': list[list[int]] | None   通道 1 每条轨迹的 0/1 mask
          'loss_mask_2': list[list[int]] | None   通道 2 同上
          'adv_1':       list[float] | None       通道 1 标量 advantage（每条一个）
          'adv_2':       list[float] | None       通道 2 同上
          'gate':        list[int]                g = 1[Δ_total>=0]
          'delta_total': list[int]
          'call_indices':list[list[int]]
          'ok':          bool                      两条通道都可用（未被零方差跳过）
        某条通道判为零方差时，它对应的 loss_mask / adv 为 None（该通道跳过），
        'ok' 仅在两条都可用时为 True。
    """
    # ---- 配置兜底：显式参数 > cfg.rl.advantage > 默认值
    cfg_eps: float = 1e-6
    cfg_policy: str = "skip_group"
    cfg_mode: str = "paper"
    if cfg is not None:
        get = getattr(cfg, "get", None)
        if callable(get):
            cfg_eps = float(get("rl.advantage.std_eps", cfg_eps))
            cfg_policy = str(get("rl.advantage.zero_var_policy", cfg_policy))
            cfg_mode = str(get("rl.advantage.answer_tokens_mode", cfg_mode))
    eps = float(std_eps) if std_eps is not None else cfg_eps
    policy = zero_var_policy if zero_var_policy is not None else cfg_policy
    md = mode if mode is not None else cfg_mode

    segs_list = list(segments_list)
    scores_list = [list(s) for s in probe_scores_list]
    if len(segs_list) != len(scores_list):
        raise ValueError(
            f"segments_list 与 probe_scores_list 长度不一致："
            f"{len(segs_list)} vs {len(scores_list)}"
        )
    if not segs_list:
        raise ValueError("group 为空（至少需要 1 条 rollout）")

    if final_scores is None:
        finals = [float(s[-1]) for s in scores_list]
    else:
        finals = [float(v) for v in final_scores]
        if len(finals) != len(segs_list):
            raise ValueError("final_scores 长度必须等于 group size")

    delta_total = [int(s[-1]) - int(s[0]) for s in scores_list]
    gate = [1 if d >= 0 else 0 for d in delta_total]

    adv_1 = group_advantage(finals, eps, policy)
    adv_2 = group_advantage(delta_total, eps, policy)

    masks_1: Optional[List[List[int]]] = None
    if adv_1 is not None:
        masks_1 = [
            ogar_masks(s, sg, mode=md).token_mask
            for s, sg in zip(scores_list, segs_list)
        ]

    masks_2: Optional[List[List[int]]] = None
    if adv_2 is not None:
        masks_2 = []
        for sg, g in zip(segs_list, gate):
            covered = _covered_mask(sg)
            masks_2.append([g * c for c in covered])

    call_idx = [
        ogar_masks(s, sg, mode=md).call_indices
        for s, sg in zip(scores_list, segs_list)
    ]

    return {
        "loss_mask_1": masks_1,
        "loss_mask_2": masks_2,
        "adv_1": adv_1,
        "adv_2": adv_2,
        "gate": gate,
        "delta_total": delta_total,
        "call_indices": call_idx,
        "ok": adv_1 is not None and adv_2 is not None,
    }


# ---------------------------------------------------------------- 探针校验


def _probe_entries(probes: Any) -> Tuple[List[Any], List[Any], List[bool]]:
    """把各种形态的 probes 拆出 (scores, ks, parse_ok_flags)。"""
    scores: List[Any] = []
    ks: List[Any] = []
    oks: List[bool] = []
    for i, p in enumerate(probes):
        if isinstance(p, bool):
            scores.append(p)
            ks.append(i)
            oks.append(True)
        elif isinstance(p, numbers.Integral):
            scores.append(int(p))
            ks.append(i)
            oks.append(True)
        elif isinstance(p, dict):
            scores.append(p.get("score"))
            ks.append(p.get("k", i))
            oks.append(bool(p.get("parse_ok", True)))
        else:  # schema.Probe 之类的对象
            scores.append(getattr(p, "score", None))
            ks.append(getattr(p, "k", i))
            oks.append(bool(getattr(p, "parse_ok", True)))
    return scores, ks, oks


def validate_probes(
    probes: Any,
    n_calls: Optional[int] = None,
    delta_k: Optional[Sequence[int]] = None,
    delta_total: Optional[int] = None,
) -> Tuple[bool, List[str]]:
    """检查一条轨迹的探针是否可用于训练。

    检查项：
      1. ``len(probes) == n_calls + 1``（探针数 = 调用数 + 1）；
      2. 每个探针分数 ∈ {-1,0,1}（R_ORM 的合法取值）；
      3. 每个探针 ``parse_ok`` 为真（解析失败应剔除整条轨迹，§7.1 工程防御）；
      4. 探针编号 k 递增且为 0..K（避免乱序）；
      5. 若给了 delta_k：长度 = K 且与 ``staged_deltas(scores)`` 一致；
      6. 若给了 delta_total：等于 ``sum(delta_k)`` 且等于 ``a_K - a_0``
         （即恒等式 ``sum(Δ_k) == Δ_total``）。

    Args:
        probes: 可以是 ``Trajectory``、``Probe`` 列表、dict 列表或 int 分数列表。
                传 Trajectory 时自动取它的 ``probes / n_calls / delta_k / delta_total``。
        n_calls / delta_k / delta_total: 显式值优先于 Trajectory 自带的字段。

    Returns:
        (ok, errors)。ok 为真时 errors 为空列表。
    """
    errors: List[str] = []

    # ---- 允许直接传 Trajectory
    if probes is not None and not isinstance(probes, (list, tuple, dict, int, bool)) \
            and hasattr(probes, "probes") and not isinstance(probes, str):
        traj = probes
        if n_calls is None:
            try:
                n_calls = int(traj.n_calls)
            except Exception:  # noqa: BLE001 —— 对象不带 steps 时退回 None
                n_calls = None
        if delta_k is None:
            dk = getattr(traj, "delta_k", None)
            delta_k = list(dk) if dk else None
        if delta_total is None:
            dt = getattr(traj, "delta_total", None)
            delta_total = dt if dt is not None else None
        probes = list(getattr(traj, "probes", []) or [])

    entries = list(probes or [])
    if not entries:
        return False, ["探针列表为空（至少需要 a_0）"]

    scores, ks, oks = _probe_entries(entries)

    if n_calls is not None and len(entries) != int(n_calls) + 1:
        errors.append(
            f"探针数 {len(entries)} != n_calls+1 = {int(n_calls) + 1}"
        )

    for i, s in enumerate(scores):
        if isinstance(s, bool) or not isinstance(s, int) or s not in VALID_SCORES:
            errors.append(f"第 {i} 个探针分数 {s!r} 不在 {VALID_SCORES} 中")

    for i, ok in enumerate(oks):
        if not ok:
            errors.append(f"第 {i} 个探针 parse_ok=False（解析失败应剔除整条轨迹）")

    for i, k in enumerate(ks):
        if not isinstance(k, int) or k != i:
            errors.append(f"探针编号乱序：第 {i} 个的 k={k!r}，应为 {i}")

    # 只有在分数全部合法时，差分恒等式才有意义
    scores_int: Optional[List[int]] = None
    if not any("分数" in e for e in errors):
        scores_int = [int(s) for s in scores]

    deltas: Optional[List[int]] = None
    if scores_int is not None:
        deltas = staged_deltas(scores_int)  # 分数已合法，不会抛

    if delta_k is not None:
        dk = [int(v) for v in delta_k]
        if deltas is not None and len(dk) != len(deltas):
            errors.append(f"delta_k 长度 {len(dk)} != K = {len(deltas)}")
        elif deltas is not None and dk != deltas:
            errors.append(f"delta_k 与 staged_deltas(scores) 不一致：{dk} vs {deltas}")
        if deltas is not None and sum(dk) != sum(deltas):
            errors.append(
                f"sum(delta_k)={sum(dk)} != sum(staged_deltas)={sum(deltas)}"
            )

    if delta_total is not None:
        dt = int(delta_total)
        ref = deltas if deltas is not None else ([int(v) for v in delta_k] if delta_k else None)
        if ref is not None and sum(ref) != dt:
            errors.append(f"sum(delta_k)={sum(ref)} != delta_total={dt}")
        if scores_int is not None and (scores_int[-1] - scores_int[0]) != dt:
            errors.append(
                f"a_K - a_0 = {scores_int[-1] - scores_int[0]} != delta_total={dt}"
            )

    return (len(errors) == 0), errors
