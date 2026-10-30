"""verl 侧的 advantage estimator 脚手架：`taco_staged`。

⚠️ 诚实声明：本文件 **未在真实 verl 上验证**。本容器没有安装 verl，也没有 GPU。
   下面的注册点、字段名、DataProto 语义都是按 verl main 分支的公开约定写的，
   接进去时务必先用单卡小 batch 跑通一次，再相信它。

【为什么注册的是函数而不是类】
verl 的 `register_adv_est` 注册的是一个 `(data, gamma, lam, ...) -> dict` 的可调用对象，
registy 里存的就是它，`ray_trainer` 直接 `adv = fn(data=..., ...)` 然后取 `adv["advantages"]`。
如果直接注册类，verl 拿到的是一个 **尚未初始化的实例**，`adv["advantages"]` 会 TypeError。
所以：类 `TacoStagedAdvantageEstimator` 承载全部逻辑与可测性，模块底的薄函数
`compute_taco_staged_advantage` 才是被注册的那个 callable。

【它的职责边界】
- 只读 `data.non_tensor_batch` 里 rollout 阶段塞进来的 `probe_scores` / `segments`；
- 全部算法调用 `medtaco/rl/advantage.py`（另一个模块负责，本文件不复制实现）；
- 输出写回 `data.batch`（verl 的 TensorDict）：
    advantages / returns                      ← verl 标准字段，形状 [bs, response_len]
    loss_mask_1 / loss_mask_2 / adv_1 / adv_2 ← 供 medtaco.rl.loss.two_pass_policy_loss 消费
- 组内零方差（`skip_group`）的样本：advantage 置 0、所有 loss mask 置 0，
  并在 `non_tensor_batch["taco_skip_uid"]` 标记 True。**真正"从 batch 里剔除"必须在
  trainer 侧做**（见 README 第 5 条），这里只负责打标，因为 verl 的 batch 是定长的。
"""
from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ verl 可选导入

try:  # pragma: no cover - 本容器无 verl，走 except 分支
    from verl.trainer.ppo.adv_estimator import register_adv_est  # type: ignore

    HAS_VERL = True
except Exception:  # noqa: BLE001 - verl 的 import 失败形态很多（无包 / 缺依赖）
    HAS_VERL = False

    def register_adv_est(name: Any = None):  # type: ignore[misc]
        """verl 缺失时的本地空装饰器：只做登记，保证本模块可 import。"""

        def _wrap(fn_or_cls: Any) -> Any:
            fn_or_cls.__adv_est_name__ = str(getattr(name, "value", name))
            return fn_or_cls

        # 兼容 `@register_adv_est` 与 `@register_adv_est("xxx")` 两种写法
        if callable(name):
            fn = name
            fn.__adv_est_name__ = "taco_staged"
            return fn
        return _wrap


try:  # pragma: no cover
    from verl.trainer.ppo.adv_estimator import AdvantageEstimator  # type: ignore
except Exception:  # noqa: BLE001

    class AdvantageEstimator(str):  # type: ignore[no-redef]
        """verl 的枚举占位；缺失时用字符串常量顶替。"""

        TACO_STAGED = "taco_staged"


ADV_NAME = "taco_staged"

# ------------------------------------------------------------------ 算法模块懒加载

_ADV_MODULE: Any = None


def _load_advantage_module() -> Any:
    """懒加载 medtaco.rl.advantage。

    **导入期绝不 import**：该模块与本脚手架并行开发，可能暂时不存在；
    把它放到调用期，import 本文件永远不会因为同伴文件缺席而崩。
    """
    global _ADV_MODULE
    if _ADV_MODULE is not None:
        return _ADV_MODULE
    try:
        _ADV_MODULE = importlib.import_module("medtaco.rl.advantage")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "无法导入 medtaco.rl.advantage —— advantage 算法实现缺失。"
            "本 estimator 只做 verl 适配，不复制实现。"
        ) from e
    return _ADV_MODULE


# ------------------------------------------------------------------ DataProto 访问

def _batch_get(data: Any, key: str) -> Any:
    """按 verl DataProto 语义取 tensor 字段：`data.batch[key]`（TensorDict）。"""
    batch = getattr(data, "batch", None)
    if batch is not None and hasattr(batch, "__contains__") and key in batch:
        return batch[key]
    if isinstance(data, dict) and key in data:
        return data[key]
    return None


def _nontensor_get(data: Any, key: str) -> Any:
    """按 verl DataProto 语义取非张量字段：`data.non_tensor_batch[key]`（object ndarray）。"""
    nt = getattr(data, "non_tensor_batch", None)
    if nt is not None and hasattr(nt, "__contains__") and key in nt:
        return nt[key]
    if isinstance(data, dict):
        if "non_tensor_batch" in data and key in data["non_tensor_batch"]:
            return data["non_tensor_batch"][key]
        if key in data:
            return data[key]
    return None


def _rows(obj: Any, bs: int) -> List[Any]:
    """把 `[bs]` 的 object ndarray / list 统一成 python list；长度不符直接报错。"""
    if obj is None:
        return [None] * bs
    seq = list(obj)
    if len(seq) != bs:
        raise ValueError(f"non_tensor_batch 行数 {len(seq)} 与 batch size {bs} 不一致")
    return seq


def _to_int_list(x: Any) -> List[int]:
    """object ndarray 里的变长 list / nparray → list[int]。"""
    if x is None:
        return []
    return [int(v) for v in np.asarray(x, dtype=object).reshape(-1).tolist()]


def _to_torch(x: np.ndarray) -> Any:
    """有 torch 就转 tensor，没有就原样返回（离线单测路径）。"""
    try:
        import torch  # noqa: F401

        return torch.as_tensor(x)
    except Exception:  # noqa: BLE001
        return x


# ------------------------------------------------------------------ 核心估计器


class TacoStagedAdvantageEstimator:
    """TACO + 分级探针 + OGAR 分段路由 的 advantage estimator。

    两条通道（PROPOSAL §7.5）：
      A1（outcome）：组内归一化的结果奖励，作用在常规 response mask 上；
      A2（process）：组内归一化的 Δ_total，再乘 OGAR 的 `m[t]` 与轨迹门 `g`。
    两条通道 **各自一套 loss mask**，绝不能在 verl 里拼成同一个 mask 去归一化。

    输入约定（由 rollout.py 写进 DataProto）：
      non_tensor_batch["probe_scores"] : object[bs]，每条是长度 K+1 的 int 列表（a_0..a_K）
      non_tensor_batch["segments"]     : object[bs]，每条是
                                         {"t1": (s, e), "calls": [(obs_s, obs_e, t2_s, t2_e), ...]}
      non_tensor_batch["uid"]          : object[bs]，GRPO 的组 id（verl 原生字段）
      batch["token_level_scores"]      : float[bs, response_len]（可选；缺失时用末次探针分当奖励）
    """

    name: str = ADV_NAME

    def __init__(
        self,
        zero_var_policy: str = "skip_group",
        std_eps: float = 1e-6,
        answer_tokens_mode: str = "paper",
        require_parse_ok: bool = True,
    ) -> None:
        self.zero_var_policy = zero_var_policy
        self.std_eps = float(std_eps)
        self.answer_tokens_mode = answer_tokens_mode
        self.require_parse_ok = bool(require_parse_ok)

    # -------------------------------------------------- 构造
    @classmethod
    def from_cfg(cls, cfg: Any) -> "TacoStagedAdvantageEstimator":
        """从 medtaco 的 Cfg 读取 `rl.advantage.*`。"""
        adv = cfg.get("rl.advantage", {})
        adv = dict(adv) if isinstance(adv, dict) else {}
        return cls(
            zero_var_policy=adv.get("zero_var_policy", "skip_group"),
            std_eps=float(adv.get("std_eps", 1e-6)),
            answer_tokens_mode=adv.get("answer_tokens_mode", "paper"),
        )

    # -------------------------------------------------- 主入口
    def compute_advantage(
        self,
        data: Any,
        gamma: float = 1.0,
        lam: float = 1.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """对齐 verl 的 `compute_advantage(data, gamma, lam, ...) -> dict` 约定。

        返回 `{"advantages": tensor[bs, L], "returns": tensor[bs, L]}` 并就地写回 data。
        `gamma`/`lam` 只为签名兼容而保留：本方法是**基于轨迹分组**的估计，不做 GAE。
        """
        adv_mod = _load_advantage_module()

        bs = self._batch_size(data)
        width = self._response_width(data)
        probe_rows = _rows(_nontensor_get(data, "probe_scores"), bs)
        seg_rows = _rows(_nontensor_get(data, "segments"), bs)
        uids = _rows(_nontensor_get(data, "uid"), bs)
        uids = [str(u) if u is not None else str(i) for i, u in enumerate(uids)]

        # ---- 1. 先建 Segment（校验需要 n_calls），再剔掉解析失败 / 长度不合法的轨迹
        all_segs: List[List[Any]] = [
            [self._build_segment(adv_mod, x) for x in self._iter_segments(s)] for s in seg_rows
        ]
        valid = [
            self._valid_probes(adv_mod, p, n_calls=sum(len(getattr(g, "calls", []) or []) for g in segs))
            for p, segs in zip(probe_rows, all_segs)
        ]
        if not all(valid):
            logger.warning("[taco_staged] 剔除 %d 条探针非法的轨迹", sum(1 for v in valid if not v))

        # ---- 2. 逐样本：Δ_k、Δ_total、OGAR 的 token mask
        deltas: List[List[int]] = []
        total: List[int] = []
        masks: List[List[int]] = []
        obs_masks: List[List[int]] = []
        for i, (p, segs) in enumerate(zip(probe_rows, all_segs)):
            scores = _to_int_list(p)
            if not valid[i] or not scores:
                deltas.append([])
                total.append(0)
                masks.append([0] * width)
                obs_masks.append([0] * width)
                continue
            d = [int(v) for v in adv_mod.staged_deltas(scores)]
            deltas.append(d)
            total.append(int(scores[-1] - scores[0]))
            # 观测 token（返回图 / 沙箱输出）永远不进 loss —— 与 OGAR 的 m[t] 无关，
            # 这是原始 response mask 的补充，A1/A2 两条通道都要乘
            obs_masks.append(self._obs_mask(segs, width))
            try:
                tm = adv_mod.ogar_masks(scores, segs, mode=self.answer_tokens_mode)
                m = _to_int_list(tm.token_mask)
            except Exception as e:  # noqa: BLE001
                logger.warning("[taco_staged] ogar_masks 失败 idx=%d: %s", i, e)
                m = []
            masks.append(self._fit_mask(m, width))

        # ---- 3. 两条通道的组内归一化（零方差组的处理在这里）
        outcome_vals = self._outcome_values(data, bs, probe_rows)
        proc_vals = [float(x) for x in total]
        out_adv, out_skipped = self._group_norm(adv_mod, uids, outcome_vals)
        prc_adv, prc_skipped = self._group_norm(adv_mod, uids, proc_vals)
        skipped = [bool(a or b) for a, b in zip(out_skipped, prc_skipped)]

        # ---- 4. 组装 mask 与逐 token advantage
        response_mask = self._response_mask(data, bs, width)
        obs_mask = np.asarray(obs_masks, dtype=np.int64).reshape(bs, width)
        mask_1 = response_mask * obs_mask             # A1：模型 token，剔除观测
        mask_2 = np.zeros((bs, width), dtype=np.int64)  # A2：只有 m[t] 选中的 process 段
        for i in range(bs):
            if skipped[i]:
                mask_1[i, :] = 0
                continue
            mask_2[i] = (mask_1[i] * np.asarray(masks[i], dtype=np.int64))
            # 轨迹门 g = 1[Δ_total >= 0]，关闭误导性分支的 process 通道
            if total[i] < 0:
                mask_2[i, :] = 0

        adv_1 = mask_1 * np.asarray(out_adv, dtype=np.float32)[:, None]
        adv_2 = mask_2 * np.asarray(prc_adv, dtype=np.float32)[:, None]
        # verl 的 advantages 字段只能有一个：A1 为主，A2 由两趟 loss 单独取用
        advantages = adv_1

        out = {
            "advantages": _to_torch(advantages.astype(np.float32)),
            "returns": _to_torch(advantages.astype(np.float32)),
        }
        self._write_back(
            data,
            {
                "advantages": advantages.astype(np.float32),
                "returns": advantages.astype(np.float32),
                "loss_mask_1": mask_1.astype(np.float32),
                "loss_mask_2": mask_2.astype(np.float32),
                "adv_1": adv_1.astype(np.float32),
                "adv_2": adv_2.astype(np.float32),
            },
            {
                "taco_skip_uid": np.array(skipped, dtype=object),
                "taco_delta_k": np.array([np.asarray(d, dtype=np.int64) for d in deltas], dtype=object),
                "taco_delta_total": np.asarray(total, dtype=np.int64),
            },
        )
        return out

    __call__ = compute_advantage

    # -------------------------------------------------- 内部工具
    @staticmethod
    def _batch_size(data: Any) -> int:
        batch = getattr(data, "batch", None)
        if batch is not None and len(batch) > 0:
            for k in batch.keys():
                return int(batch[k].shape[0])
        nt = getattr(data, "non_tensor_batch", None)
        if nt is not None and len(nt) > 0:
            for k in nt.keys():
                return int(len(nt[k]))
        if isinstance(data, dict):
            src = data.get("batch") or data.get("non_tensor_batch") or {}
            for k in src:
                return int(np.shape(src[k])[0])
        raise ValueError("无法确定 batch size：既没有 data.batch 也没有 non_tensor_batch")

    @staticmethod
    def _response_width(data: Any) -> int:
        for key in ("responses", "response_mask", "input_ids"):
            t = _batch_get(data, key)
            if t is not None and getattr(t, "ndim", 2) >= 2:
                return int(t.shape[-1])
        raise ValueError("data.batch 里找不到 responses/response_mask，无法确定 response 长度")

    @staticmethod
    def _response_mask(data: Any, bs: int, width: int) -> np.ndarray:
        """拿 verl 自己的 response mask；没有就退化成"全 1（扣掉 padding）"。"""
        t = _batch_get(data, "response_mask")
        if t is None:
            return np.ones((bs, width), dtype=np.int64)
        arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        arr = arr.astype(np.int64)
        return TacoStagedAdvantageEstimator._fit_mask(arr.reshape(-1).tolist(), width).reshape(bs, width) \
            if arr.ndim == 1 else TacoStagedAdvantageEstimator._fit_matrix(arr, bs, width)

    @staticmethod
    def _fit_matrix(arr: np.ndarray, bs: int, width: int) -> np.ndarray:
        out = np.zeros((bs, width), dtype=np.int64)
        r = min(bs, arr.shape[0])
        c = min(width, arr.shape[1])
        out[:r, :c] = arr[:r, :c]
        return out

    @staticmethod
    def _obs_mask(segs: Sequence[Any], width: int) -> List[int]:
        """观测区间置 0，其余置 1。观测 = 返回图 + 沙箱输出，两层 mask 都要乘它。"""
        m = [1] * width
        for seg in segs:
            for call in getattr(seg, "calls", []) or []:
                obs_s, obs_e = int(call[0]), int(call[1])
                for i in range(max(0, obs_s), min(width, obs_e)):
                    m[i] = 0
        return m

    @staticmethod
    def _fit_mask(mask: Sequence[int], width: int) -> List[int]:
        """把任意长度的 token_mask 对齐到 response 宽度（截断 / 右侧补 0）。"""
        m = [int(v) for v in mask][:width]
        return m + [0] * (width - len(m))

    def _valid_probes(self, adv_mod: Any, probes: Any, n_calls: Optional[int] = None) -> bool:
        """探针合法性。`validate_probes` 的签名是 `(probes, n_calls=..., ...) -> (ok, errors)`，
        也兼容只返回 bool 的简化实现。

        传 `n_calls` 很关键：不传时真实模块无法检查 `len(probes) == K+1` 与 k 递增，
        非法轨迹会漏进 advantage。
        """
        scores = _to_int_list(probes)
        if not scores:
            return False
        if not self.require_parse_ok:
            return True
        fn = getattr(adv_mod, "validate_probes", None)
        if fn is None:
            return True
        try:
            res = fn(scores, n_calls=n_calls) if n_calls is not None else fn(scores)
        except TypeError:
            res = fn(scores)  # 老版本 / 简化实现不接受关键字
        except Exception:  # noqa: BLE001 - 校验器版本差异，失败按非法处理并告警
            logger.warning("[taco_staged] validate_probes 抛异常，按非法轨迹处理")
            return False
        ok = res[0] if isinstance(res, tuple) else res
        if not ok:
            logger.info("[taco_staged] 探针非法，剔除轨迹: %s", res[1] if isinstance(res, tuple) else scores)
        return bool(ok)

    @staticmethod
    def _iter_segments(seg_obj: Any) -> List[Any]:
        if seg_obj is None:
            return [None]
        if isinstance(seg_obj, dict):
            return [seg_obj]
        return list(seg_obj)

    @staticmethod
    def _build_segment(adv_mod: Any, s: Any) -> Any:
        """dict / tuple → `adv_mod.Segment`。rollout 侧只放 dict，避免跨进程 pickle 类对象。"""
        seg_cls = getattr(adv_mod, "Segment")
        if isinstance(s, seg_cls):
            return s
        if s is None:
            return seg_cls(t1=(-1, -1), calls=[])
        if isinstance(s, dict):
            t1 = tuple(s.get("t1", (-1, -1)))
            calls = [tuple(c) for c in s.get("calls", [])]
        else:
            t1 = tuple(s[0])
            calls = [tuple(c) for c in s[1]]
        return seg_cls(t1=(int(t1[0]), int(t1[1])), calls=calls)

    @staticmethod
    def _outcome_values(data: Any, bs: int, probe_rows: Sequence[Any]) -> List[float]:
        """结果奖励：优先 token_level_scores 行和；否则用末次探针得分（R(a_K)）。"""
        t = _batch_get(data, "token_level_scores")
        if t is None:
            t = _batch_get(data, "rm_scores")
        if t is not None:
            arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 1:
                return [float(v) for v in arr[:bs]]
            return [float(v) for v in arr.reshape(arr.shape[0], -1).sum(axis=1)[:bs]]
        # 兜底：末次探针分
        vals: List[float] = []
        for i in range(bs):
            sc = _to_int_list(probe_rows[i])
            vals.append(float(sc[-1]) if sc else 0.0)
        return vals

    def _group_norm(
        self, adv_mod: Any, uids: Sequence[str], values: Sequence[float]
    ) -> Tuple[List[float], List[bool]]:
        """按 uid 分组做组内归一化；返回 (逐样本 advantage, 逐样本是否被跳过)。"""
        adv = [0.0] * len(values)
        skipped = [False] * len(values)
        groups: Dict[str, List[int]] = {}
        for i, u in enumerate(uids):
            groups.setdefault(u, []).append(i)
        for u, idx in groups.items():
            vals = [float(values[i]) for i in idx]
            try:
                res = adv_mod.group_advantage(vals, self.std_eps, self.zero_var_policy)
            except Exception as e:  # noqa: BLE001
                logger.warning("[taco_staged] group_advantage 失败 uid=%s: %s", u, e)
                res = None
            if res is None:
                # 组内零方差：整组跳过（打标，交给 trainer 剔除，见 README 第 5 条）
                for i in idx:
                    skipped[i] = True
                    adv[i] = 0.0
                continue
            if len(res) != len(idx):
                raise ValueError(f"group_advantage 返回长度 {len(res)} != 组大小 {len(idx)}")
            for i, v in zip(idx, res):
                adv[i] = float(v)
        return adv, skipped

    @staticmethod
    def _write_back(data: Any, tensors: Dict[str, np.ndarray], nontensors: Dict[str, Any]) -> None:
        """写回 DataProto。真实 verl 的 batch 是 TensorDict，直接赋值即可。"""
        batch = getattr(data, "batch", None)
        nt = getattr(data, "non_tensor_batch", None)
        if batch is not None:
            for k, v in tensors.items():
                batch[k] = _to_torch(v)
        elif isinstance(data, dict):
            data.setdefault("batch", {}).update(tensors)
        if nt is not None:
            for k, v in nontensors.items():
                nt[k] = v
        elif isinstance(data, dict):
            data.setdefault("non_tensor_batch", {}).update(nontensors)

    # -------------------------------------------------- 调试
    def describe(self) -> str:
        return (
            f"TacoStagedAdvantageEstimator(name={self.name}, "
            f"zero_var_policy={self.zero_var_policy}, std_eps={self.std_eps}, "
            f"answer_tokens_mode={self.answer_tokens_mode})"
        )


# ------------------------------------------------------------------ 注册点

# TODO(verl 对齐)：verl main 的 registry 想要 `AdvantageEstimator` 枚举 key，
#   老版本接受任意字符串。若你的 verl 版本强校验枚举，把下面这行换成
#   `@register_adv_est(AdvantageEstimator.TACO_STAGED)` 并在
#   `verl/trainer/ppo/adv_estimator.py:AdvantageEstimator` 里补一个成员。
@register_adv_est(ADV_NAME)
def compute_taco_staged_advantage(
    data: Any,
    gamma: float = 1.0,
    lam: float = 1.0,
    estimator: Optional[TacoStagedAdvantageEstimator] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """verl registry 的入口。超参通过 `@register_adv_est` 无参形式调用时由 trainer 注入 cfg。"""
    est = estimator or TacoStagedAdvantageEstimator(
        zero_var_policy=kwargs.get("zero_var_policy", "skip_group"),
        std_eps=float(kwargs.get("std_eps", 1e-6)),
        answer_tokens_mode=kwargs.get("answer_tokens_mode", "paper"),
    )
    return est.compute_advantage(data=data, gamma=gamma, lam=lam, **kwargs)


ESTIMATOR = TacoStagedAdvantageEstimator

__all__ = [
    "ADV_NAME",
    "HAS_VERL",
    "AdvantageEstimator",
    "TacoStagedAdvantageEstimator",
    "compute_taco_staged_advantage",
    "register_adv_est",
    "ESTIMATOR",
]
