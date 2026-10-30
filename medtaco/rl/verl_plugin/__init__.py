"""verl 侧集成脚手架。

⚠️ 全部代码 **未在真实 verl 上验证**（本容器无 verl、无 GPU）。

  adv_est.py : `@register_adv_est("taco_staged")` 的 advantage estimator
  rollout.py : `TacoAgentLoop` 多轮工具 rollout 骨架

export 故意不做 eager import：`medtaco.rl.advantage` 是并行开发的兄弟模块，
import 本包不应该因为同伴文件缺席或 verl 缺席而崩。
"""
from __future__ import annotations

__all__ = ["adv_est", "rollout"]
