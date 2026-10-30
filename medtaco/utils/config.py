"""yaml 配置：支持点号取值与命令行式覆盖。

用法:
    cfg = load_config("configs/default.yaml")
    cfg.rl.alpha_1          # 1.0
    cfg.get("rl.nope", 3)   # 3
"""
from __future__ import annotations

import os
from typing import Any, Dict

import yaml


class Cfg(dict):
    """dict 的薄包装，属性访问递归包装子字典。"""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) else v

    def get(self, path: str, default: Any = None) -> Any:
        """支持 'rl.alpha_1' 形式的路径取值。"""
        cur: Any = self
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return Cfg(cur) if isinstance(cur, dict) else cur

    def set(self, path: str, value: Any) -> "Cfg":
        parts = path.split(".")
        cur = self
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value
        return self


def _expand(node, table: Dict[str, Any]):
    """把 ``${a.b.c}`` 换成 table 里解析好的值，支持一整串里的多处引用与链式引用。

    例如 ``raw: ${paths.root}/raw`` 会拼成 ``<root>/raw``。引用可链式
    （``a: ${b}/x``，``b: ${c}``），所以按需递归求值；**环引用抛 ValueError**——
    静默保留 ``${...}`` 字面量只会让下游写出一个叫 ``${paths.root}`` 的目录，
    比直接失败难查得多。
    """
    import re

    pat = re.compile(r"\$\{([^{}]+)\}")
    memo: Dict[str, Any] = {}

    def resolve(key: str, stack: tuple):
        if key in memo:
            return memo[key]
        if key in stack:
            raise ValueError(f"配置引用成环: {' -> '.join(stack + (key,))}")
        if key not in table:
            return None
        v = table[key]
        if isinstance(v, str):
            v = pat.sub(lambda m: _sub(m.group(1).strip(), stack + (key,)), v)
        memo[key] = v
        return v

    def _sub(key: str, stack: tuple) -> str:
        got = resolve(key, stack)
        return f"${{{key}}}" if got is None else str(got)   # 未定义引用原样保留，便于发现

    def walk(n):
        if isinstance(n, dict):
            return {k: walk(v) for k, v in n.items()}
        if isinstance(n, list):
            return [walk(v) for v in n]
        if isinstance(n, str) and "${" in n:
            return pat.sub(lambda m: _sub(m.group(1).strip(), ()), n)
        return n

    return walk(node)


def _flatten(node, prefix: str = "") -> Dict[str, Any]:
    """把嵌套 dict 摊成 'a.b.c' → 值，供 ${} 引用。"""
    out: Dict[str, Any] = {}
    if isinstance(node, dict):
        for k, v in node.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            out[path] = v
            out.update(_flatten(v, path))
    return out


def load_config(path: str, overrides: Dict[str, Any] | None = None) -> Cfg:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    # 两趟：先摊平拿引用表，再统一替换 ${...}。引用只能指向"已经字面写好"的值，
    # 不做递归求值——嵌套引用会让配置变成一门小语言，得不偿失。
    raw = _expand(raw, _flatten(raw))
    cfg = Cfg(raw)
    for k, v in (overrides or {}).items():
        cfg.set(k, v)
    # 环境变量覆盖最高优先级：MEDTACO_<DOTTED_KEY>
    for env_k, env_v in os.environ.items():
        if env_k.startswith("MEDTACO_"):
            cfg.set(env_k[len("MEDTACO_"):].lower().replace("__", "."), env_v)
    return cfg
