"""统计工具：Wilson CI / 配对 McNemar / 聚类 DEFF / bootstrap。

对应 PROPOSAL §8.3。只依赖标准库 + numpy（可选）。
"""
from __future__ import annotations

import math
import random
from typing import Iterable, List, Optional, Sequence, Tuple


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """二项比例的 Wilson 95% CI。VQA-RAD 的 451 题会给出 ±4.2pp。"""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar(b: int, c: int) -> Tuple[float, float]:
    """精确 McNemar。b = A对B错, c = A错B对。

    返回 (统计量 chi2, p 值)。用二项精确检验，小样本也成立。
    """
    n = b + c
    if n == 0:
        return (0.0, 1.0)
    chi2 = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0
    # 精确 p：双尾二项
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return (chi2, min(1.0, p))


def mcnemar_power_n(diff: float, discordance: float, power: float = 0.8,
                    alpha: float = 0.05) -> int:
    """估算配对检验所需样本量。

    diff: 想检测的准确率差（如 0.03）
    discordance: 不一致率（如 0.10）
    """
    z_a = 1.959964 if alpha == 0.05 else 2.575829
    z_b = 0.841621 if power == 0.8 else (1.281552 if power == 0.9 else 0.0)
    # 标准近似：n = (z_a*sqrt(disc) + z_b*sqrt(disc - diff^2))^2 / diff^2
    if diff <= 0:
        return 10 ** 9
    inner = disc = max(discordance, diff ** 2 + 1e-9)
    num = (z_a * math.sqrt(disc) + z_b * math.sqrt(max(0.0, disc - diff * diff))) ** 2
    return int(math.ceil(num / (diff * diff)))


def deff(m: float, icc: float) -> float:
    """设计效应 DEFF = 1 + (m-1)*ICC。m = 每簇平均题数。"""
    return 1 + (m - 1) * icc


def effective_n(n: int, m: float, icc: float) -> float:
    return n / deff(m, icc) if deff(m, icc) > 0 else float(n)


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 42, stat: str = "mean") -> Tuple[float, float, float]:
    """返回 (点估计, 下界, 上界)。stat in {mean, acc}（acc 用于 0/1 序列）。"""
    if not values:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    pool = list(values)
    n = len(pool)
    est = sum(pool) / n
    boots: List[float] = []
    for _ in range(n_boot):
        s = sum(pool[rng.randrange(n)] for _ in range(n)) / n
        boots.append(s)
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot) - 1]
    return (est, lo, hi)


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 2000,
                     alpha: float = 0.05, seed: int = 42) -> Tuple[float, float, float]:
    """配对 bootstrap 的差值 CI（bootstrap 重采样同一批题，保留配对结构）。"""
    assert len(a) == len(b)
    rng = random.Random(seed)
    n = len(a)
    d = [x - y for x, y in zip(a, b)]
    est = sum(d) / n
    boots = []
    for _ in range(n_boot):
        s = sum(d[rng.randrange(n)] for _ in range(n)) / n
        boots.append(s)
    boots.sort()
    return (est, boots[int((alpha / 2) * n_boot)], boots[int((1 - alpha / 2) * n_boot) - 1])


def judge_attenuation(true_delta: float, sens: float, spec: float) -> float:
    """LLM judge 的 sens/spec 对真实差值的衰减：Δ_obs ≈ Δ·(sens+spec-1)。"""
    return true_delta * (sens + spec - 1)


def summarize_acc(correct: Sequence[bool], n_boot: int = 2000) -> dict:
    n = len(correct)
    k = sum(1 for c in correct if c)
    lo, hi = wilson_ci(k, n)
    est, blo, bhi = bootstrap_ci([1.0 if c else 0.0 for c in correct], n_boot)
    return {"n": n, "k": k, "acc": k / n if n else float("nan"),
            "wilson_lo": lo, "wilson_hi": hi,
            "boot_lo": blo, "boot_hi": bhi}
