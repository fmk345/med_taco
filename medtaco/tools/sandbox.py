"""模型写的 Python 代码的安全执行（沙箱）。

防护是分层的，**任何一层单独拿出来都不等于真隔离**：

1. 静态层 ``static_check``：AST 白名单 + 禁用语句（Import/While/With/Try/...）+
   禁止双下划线属性与危险内建名（`eval/exec/open/__import__/...`）。
   链式逃逸如 ``().__class__.__bases__[0].__subclasses__()`` 会在第一跳
   ``. __class__`` 上被拒。
2. 命名空间层：``__builtins__`` 换成白名单字典，不给 ``open/eval/getattr`` 等；
   只注入图像、工具函数、``math``/``statistics`` 与受限的 numpy 门面。
3. 时间层：``signal.setitimer`` 强制打断（Linux、主线程）。
   非主线程或需要进程级隔离时改走子进程（``subprocess`` + 超时）。
4. 进程层（可选）：``backend=gvisor`` 时用 ``runsc`` 拉起子进程。

诚实汇报后端：``runsc`` 不可用时**必须**打 warning 并降级到受限 exec，
实际生效的后端写进 ``ToolResult.meta['backend_used']``，绝不假装已经隔离。

    meta['backend_used'] ∈ {'gvisor', 'subprocess', 'restricted_exec',
                            'none(static_check)'}

关于 gVisor：这里用的是 ``runsc do``，它**共享宿主文件系统**，防护重点是系统调用面
（比裸 exec 强，但不等价于一个干净 rootfs 的 OCI bundle）。需要完整隔离时把
``rl.sandbox.gvisor_bundle_dir`` 指向准备好的 OCI bundle，并把
``rl.sandbox.runsc_path`` 指到 runsc 二进制。
"""
from __future__ import annotations

import ast
import builtins as _py_builtins
import importlib
import io
import itertools
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ..utils.schema import Box, ToolResult

__all__ = ["SandboxError", "static_check", "run_code"]

# 仓库根目录（med_taco/），子进程靠它设 PYTHONPATH
_REPO_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 单次执行的资源上限（硬编码兜底，防止模型扔一段病态代码进来）
_MAX_CODE_CHARS: int = 100_000
_MAX_AST_NODES: int = 20_000
_MAX_OBS_IMAGES: int = 16          # 一次执行最多落盘多少张返回图
_MAX_TEXT_CHARS: int = 8_000       # ToolResult.text 截断长度
_MAX_LIST_REPR: int = 400          # 单个 result 值的 repr 截断长度

# 观测图文件名计数器（不用随机数，保证同名文件不撞）
_obs_seq = itertools.count()


class SandboxError(Exception):
    """代码被静态检查拒绝，或运行期触发了沙箱自身的限制。"""


# ---------------------------------------------------------------- 配置读取


_DEFAULT_SANDBOX: Dict[str, Any] = {
    "backend": "gvisor",
    "timeout_s": 10.0,
    "mem_mb": 2048,
    "allow_imports": ["math", "statistics"],
    "blacklist_attrs": ["__subclasses__", "__globals__", "__builtins__", "__code__", "__class__"],
    "runsc_path": "",
    "gvisor_bundle_dir": "",
    "read_roots": [],          # load_image 额外允许读的目录（默认只有 ctx.out_dir）
}


def _cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    """按点号路径取值，同时兼容 ``Cfg``（dict 子类）与普通 dict / 对象。"""
    cur: Any = cfg
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
        else:
            cur = getattr(cur, part, None)
            if cur is None:
                return default
    return default if cur is None else cur


def _sandbox_cfg(cfg: Any) -> Dict[str, Any]:
    """把 ``cfg.rl.sandbox`` 归一成带默认值的普通 dict。"""
    out = dict(_DEFAULT_SANDBOX)
    raw = _cfg_get(cfg, "rl.sandbox", None)
    if isinstance(raw, dict):
        for k, v in raw.items():
            out[k] = v
    out["backend"] = str(out.get("backend") or "gvisor").strip().lower()
    try:
        out["timeout_s"] = float(out.get("timeout_s") or 10.0)
    except (TypeError, ValueError):
        out["timeout_s"] = 10.0
    try:
        out["mem_mb"] = int(out.get("mem_mb") or 2048)
    except (TypeError, ValueError):
        out["mem_mb"] = 2048
    out["allow_imports"] = [str(x) for x in (out.get("allow_imports") or [])]
    out["blacklist_attrs"] = [str(x) for x in (out.get("blacklist_attrs") or [])]
    out["read_roots"] = [str(x) for x in (out.get("read_roots") or [])]
    return out


# ---------------------------------------------------------------- 静态检查


# 允许的 AST 节点。INTERFACES 里列的是语句/表达式主干，这里补齐必需的
# 上下文节点（Load/Store）、算符和函数签名节点，否则正常代码都过不去。
_ALLOWED_NODES: frozenset = frozenset({
    # 顶层与语句
    "Module", "Assign", "AugAssign", "AnnAssign", "Expr", "For", "If",
    "FunctionDef", "Return", "Break", "Continue", "Pass", "Assert",
    # 表达式
    "Name", "Constant", "Attribute", "Subscript", "Slice", "Call", "keyword",
    "Starred", "BinOp", "UnaryOp", "BoolOp", "Compare", "IfExp", "NamedExpr",
    "Tuple", "List", "Dict", "Set",
    "ListComp", "SetComp", "DictComp", "GeneratorExp", "comprehension",
    # 函数签名
    "arguments", "arg",
    # 名字上下文
    "Load", "Store",
    # 运算符
    "Add", "Sub", "Mult", "Div", "FloorDiv", "Mod", "Pow", "MatMult",
    "UAdd", "USub", "Not", "Invert",
    "And", "Or",
    "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "In", "NotIn", "Is", "IsNot",
})

# 明确拒绝的节点，报错信息直接给模型的修复方向（优先于白名单命中）
_BANNED_NODES: Dict[str, str] = {
    "Import": "禁止 import：模块请用预注入的 math / statistics（见 cfg.rl.sandbox.allow_imports）",
    "ImportFrom": "禁止 from ... import：模块请用预注入的 math / statistics",
    "While": "禁止 while 循环：请用 for + range 表达迭代",
    "With": "禁止 with 语句：沙箱里没有需要上下文管理器的资源",
    "AsyncWith": "禁止 async with",
    "Try": "禁止 try/except：沙箱不允许吞掉异常",
    "TryStar": "禁止 try/except*",
    "ExceptHandler": "禁止 except 子句",
    "Lambda": "禁止 lambda：请用 def 定义函数",
    "ClassDef": "禁止 class 定义",
    "Global": "禁止 global",
    "Nonlocal": "禁止 nonlocal",
    "Delete": "禁止 del 语句",
    "Raise": "禁止 raise 语句",
    "Yield": "禁止 yield（不支持生成器）",
    "YieldFrom": "禁止 yield from",
    "Await": "禁止 await",
    "AsyncFunctionDef": "禁止 async def",
    "AsyncFor": "禁止 async for",
}

# 危险内建名 / 逃逸入口。任何位置出现（包括赋值目标）一律拒绝。
_BANNED_NAMES: frozenset = frozenset({
    "eval", "exec", "compile", "open", "input", "__import__",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "dir",
    "type", "object", "super", "memoryview", "breakpoint", "help", "exit", "quit",
    "os", "sys", "subprocess", "shutil", "importlib", "pathlib", "socket",
    "ctypes", "pickle", "marshal", "builtins", "site", "platform", "threading",
})

# 不带双下划线但对 CPython 来说就是"逃逸电梯"的属性名
_EXTRA_BANNED_ATTRS: frozenset = frozenset({
    "f_globals", "f_locals", "f_back", "f_builtins", "f_code",
    "func_globals", "func_code", "func_closure",
    "gi_frame", "gi_code", "gi_yieldfrom",
    "cr_frame", "cr_code", "cr_await",
    "ag_frame", "ag_code",
    "tb_frame", "tb_next",
    "mro",
})


class _AstChecker(ast.NodeVisitor):
    """AST 白名单遍历器；命中任何非法节点就抛 SandboxError。"""

    def __init__(self, allow_imports: Sequence[str], blacklist_attrs: Sequence[str]) -> None:
        self.allow_imports: List[str] = list(allow_imports)
        self.banned_attrs = _EXTRA_BANNED_ATTRS | frozenset(blacklist_attrs)

    # -- 覆盖 NodeVisitor 的默认分发：类型不在白名单就拒 -------------------
    def generic_visit(self, node: ast.AST) -> None:
        name = type(node).__name__
        if name in _BANNED_NODES:
            raise SandboxError(f"{_BANNED_NODES[name]}（第 {getattr(node, 'lineno', '?')} 行）")
        if name not in _ALLOWED_NODES:
            raise SandboxError(
                f"不允许的语法节点 {name}（第 {getattr(node, 'lineno', '?')} 行）："
                f"只允许赋值 / for / if / def / return 与算术、比较、调用、下标、属性访问"
            )
        super().generic_visit(node)

    # -- import：给出"已预注入"的提示 --------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        raise SandboxError(self._import_msg([a.name.split(".")[0] for a in node.names]))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise SandboxError(self._import_msg([node.module or ""]))

    def _import_msg(self, names: Sequence[str]) -> str:
        msg = "禁止 import：" + ", ".join(n for n in names if n)
        injected = [n for n in names if n in self.allow_imports]
        if injected:
            msg += f"；{'/'.join(injected)} 已预注入命名空间，直接使用即可"
        return msg

    # -- 属性：双下划线 + 危险名 -------------------------------------------
    def visit_Attribute(self, node: ast.Attribute) -> None:
        attr = node.attr
        if attr.startswith("__") or attr in self.banned_attrs:
            raise SandboxError(
                f"禁止访问属性 .{attr}（第 {node.lineno} 行）：双下划线 / 反射属性可能用于逃逸沙箱"
            )
        self.generic_visit(node)

    # -- 名字：双下划线 + 危险内建 -----------------------------------------
    def visit_Name(self, node: ast.Name) -> None:
        nid = node.id
        if nid.startswith("__"):
            raise SandboxError(f"禁止使用名字 {nid}（第 {node.lineno} 行）：双下划线标识符可访问运行期内部")
        if nid in _BANNED_NAMES:
            raise SandboxError(
                f"禁止使用 {nid}（第 {node.lineno} 行）：危险内建或逃逸入口，"
                f"可用工具为 image / crop / zoom / window / measure / load_image / math / statistics"
            )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.decorator_list:
            raise SandboxError(f"禁止装饰器（第 {node.lineno} 行）")
        self.generic_visit(node)


def static_check(code: str, cfg: Any) -> None:
    """AST 白名单检查；非法代码抛 :class:`SandboxError`（带中文原因）。"""
    sb = _sandbox_cfg(cfg)
    if not isinstance(code, str):
        raise SandboxError(f"代码必须是 str，收到 {type(code).__name__}")
    if not code.strip():
        raise SandboxError("代码为空")
    if len(code) > _MAX_CODE_CHARS:
        raise SandboxError(f"代码过长（{len(code)} 字符 > {_MAX_CODE_CHARS}）")
    try:
        tree = ast.parse(code, filename="<model_code>", mode="exec")
    except SyntaxError as e:
        raise SandboxError(f"语法错误：第 {e.lineno} 行 {e.msg}") from None
    n_nodes = sum(1 for _ in ast.walk(tree))
    if n_nodes > _MAX_AST_NODES:
        raise SandboxError(f"代码过于复杂（{n_nodes} 个 AST 节点 > {_MAX_AST_NODES}）")
    _AstChecker(sb["allow_imports"], sb["blacklist_attrs"]).visit(tree)


# ---------------------------------------------------------------- 命名空间


# 允许出现在 __builtins__ 里的名字。刻意不含 open / eval / exec / getattr / type / object。
_SAFE_BUILTIN_NAMES: Tuple[str, ...] = (
    "abs", "all", "any", "bin", "bool", "chr", "dict", "divmod", "enumerate",
    "filter", "float", "format", "frozenset", "hex", "int", "isinstance",
    "issubclass", "iter", "len", "list", "map", "max", "min", "next", "oct",
    "ord", "pow", "range", "repr", "reversed", "round", "set", "slice",
    "sorted", "str", "sum", "tuple", "zip",
)

_SAFE_EXCEPTIONS: Tuple[str, ...] = (
    "ArithmeticError", "AssertionError", "Exception", "IndexError", "KeyError",
    "NotImplementedError", "OverflowError", "RuntimeError", "TypeError",
    "ValueError", "ZeroDivisionError", "AttributeError", "StopIteration",
)

# 只有这些标准库允许通过 allow_imports 预注入（其余一律忽略并告警）
_SAFE_MODULES: frozenset = frozenset({
    "math", "statistics", "cmath", "fractions", "decimal", "random",
    "itertools", "collections", "functools", "operator", "re",
})

# numpy 门面：只暴露纯计算函数，挡住 np.save / np.load 这类文件 IO
_NP_FACADE_NAMES: Tuple[str, ...] = (
    "abs", "all", "any", "argmax", "argmin", "argsort", "around", "array",
    "asarray", "clip", "concatenate", "cos", "ceil", "cumsum", "degrees",
    "exp", "floor", "histogram", "isnan", "log", "log2", "logical_and",
    "logical_not", "logical_or", "max", "mean", "median", "min", "nonzero",
    "percentile", "pi", "prod", "radians", "ravel", "reshape", "sin", "sort",
    "sqrt", "stack", "std", "sum", "unique", "var", "where", "zeros_like",
    "ones_like", "full_like", "e", "inf", "nan", "float64", "int32", "int64",
    "uint8", "ndarray", "newaxis",
)


def _make_builtins(buf: io.StringIO) -> Dict[str, Any]:
    """构造受限 __builtins__：白名单内建 + 少数异常类型 + 捕获到 buf 的 print。"""
    safe: Dict[str, Any] = {}
    for name in _SAFE_BUILTIN_NAMES:
        if hasattr(_py_builtins, name):
            safe[name] = getattr(_py_builtins, name)
    for name in _SAFE_EXCEPTIONS:
        if hasattr(_py_builtins, name):
            safe[name] = getattr(_py_builtins, name)

    def _print(*args: Any, sep: str = " ", end: str = "\n", file: Any = None, **_: Any) -> None:
        # 忽略 file=，统一写到捕获缓冲，防止模型往进程 stdout 灌垃圾
        buf.write(sep.join(str(a) for a in args) + end)

    safe["print"] = _print
    return safe


def _make_np_facade() -> types.SimpleNamespace:
    """numpy 的只读门面：只给纯计算入口，不给文件 IO / 随机数。"""
    attrs = {n: getattr(np, n) for n in _NP_FACADE_NAMES if hasattr(np, n)}
    return types.SimpleNamespace(**attrs)


def _ctx_get(ctx: Any, key: str, default: Any = None) -> Any:
    """从 ToolContext（或测试用的 dict / SimpleNamespace）取字段。"""
    if ctx is None:
        return default
    if isinstance(ctx, dict):
        return ctx.get(key, default)
    return getattr(ctx, key, default)


def _load_ops() -> types.ModuleType:
    """惰性导入 medtaco.tools.ops（由别的模块实现，可能尚未就绪）。

    测试可以 monkeypatch 本函数注入替身——子进程后端里也要能生效，
    所以状态放在模块级 ``_OPS_MODULE`` 上而不是 sys.modules 里。
    """
    if _OPS_MODULE[0] is not None:
        return _OPS_MODULE[0]
    try:
        return importlib.import_module("medtaco.tools.ops")
    except Exception as e:  # noqa: BLE001 - 任何导入失败都转成沙箱错误
        raise SandboxError(f"工具实现不可用（medtaco.tools.ops 未就绪）：{e}") from None


# 测试注入点：非 None 时 _load_ops 直接返回它
_OPS_MODULE: List[Optional[types.ModuleType]] = [None]


# ---------------------------------------------------------------- 返回图落盘


def _to_uint8(arr: Any) -> np.ndarray:
    """把任意 dtype / 通道数的数组归一成 HxWx3 uint8。"""
    a = np.asarray(arr)
    if a.ndim == 2:
        a = a[:, :, None]
    if a.ndim != 3:
        raise SandboxError(f"期望 2D/3D 图像数组，收到 shape={a.shape}")
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    elif a.shape[2] == 4:
        a = a[:, :, :3]
    elif a.shape[2] != 3:
        raise SandboxError(f"期望 1/3/4 通道，收到 shape={a.shape}")
    if a.dtype != np.uint8:
        a = a.astype(np.float64)
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        if hi <= 1.0 and lo >= 0.0:      # 0~1 浮点图
            a = a * 255.0
        elif hi > 255.0 or lo < 0.0:     # 其它动态范围按 min-max 拉伸
            a = (a - lo) / max(hi - lo, 1e-8) * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


def _save_obs(arr: Any, ctx: Any, produced: List[str]) -> str:
    """把返回图压成 ≤obs_max_side 的 JPEG 落盘，返回路径。"""
    if len(produced) >= _MAX_OBS_IMAGES:
        raise SandboxError(f"一次执行最多保存 {_MAX_OBS_IMAGES} 张返回图")
    out_dir = str(_ctx_get(ctx, "out_dir", "") or "") or os.path.join(
        tempfile.gettempdir(), "medtaco_sandbox"
    )
    os.makedirs(out_dir, exist_ok=True)
    max_side = int(_ctx_get(ctx, "obs_max_side", 1024) or 1024)
    quality = int(_ctx_get(ctx, "jpeg_quality", 90) or 90)

    img = Image.fromarray(_to_uint8(arr), mode="RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.BILINEAR)

    seq = next(_obs_seq)
    path = os.path.join(out_dir, f"sandbox_{seq:04d}_{os.getpid()}.jpg")
    img.save(path, format="JPEG", quality=quality)
    produced.append(path)
    return path


# ---------------------------------------------------------------- 工具注入


def _coerce_box(value: Any) -> Box:
    """把模型给的框（Box / 4~5 元序列 / dict）统一成 Box。"""
    if isinstance(value, Box):
        return value
    if isinstance(value, dict):
        try:
            return Box(
                float(value["x0"]), float(value["y0"]),
                float(value["x1"]), float(value["y1"]),
                str(value.get("label", "")),
            )
        except KeyError as e:
            raise SandboxError(f"框字典缺少字段 {e}：需要 x0/y0/x1/y1") from None
    if isinstance(value, (list, tuple, np.ndarray)):
        seq = list(np.asarray(value).ravel()) if isinstance(value, np.ndarray) else list(value)
        if len(seq) < 4:
            raise SandboxError(f"框需要 4 个坐标 [x0, y0, x1, y1]，收到 {len(seq)} 个")
        label = str(seq[4]) if len(seq) > 4 else ""
        try:
            return Box(float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3]), label)
        except (TypeError, ValueError) as e:
            raise SandboxError(f"框坐标非法：{e}") from None
    raise SandboxError(f"无法识别的框类型 {type(value).__name__}（需要 Box 或 [x0,y0,x1,y1]）")


def _as_image(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    raise SandboxError(f"图像参数必须是 numpy 数组，收到 {type(value).__name__}")


def _make_tools(
    ctx: Any,
    produced: List[str],
    readings: List[str],
    crop_boxes: List[List[float]],
    read_roots: Sequence[str] = (),
) -> Dict[str, Callable[..., Any]]:
    """构造注入给模型的工具函数。每个调用都落到 medtaco.tools.ops 的原语上。"""

    def crop(img: Any, box: Any, pad_ratio: float = 0.0) -> np.ndarray:
        """裁剪：crop(image, [x0,y0,x1,y1]) -> ndarray。"""
        ops = _load_ops()
        b = _coerce_box(box)
        arr = ops.crop(_as_image(img), b, pad_ratio=float(pad_ratio))
        _save_obs(arr, ctx, produced)
        crop_boxes.append([b.x0, b.y0, b.x1, b.y1])
        return arr

    def zoom(img: Any, box: Any, out_side: int = 1024) -> np.ndarray:
        """裁剪 + 上采样到 out_side 长边（不超过原图像素上限）。"""
        ops = _load_ops()
        b = _coerce_box(box)
        arr = ops.zoom(_as_image(img), b, out_side=int(out_side))
        _save_obs(arr, ctx, produced)
        crop_boxes.append([b.x0, b.y0, b.x1, b.y1])
        return arr

    def window(img: Any, center: float, width: float, meta: Any = None) -> np.ndarray:
        """按 WW/WL 窗宽窗位还原，缺 HU 信息时 ops 会抛 ToolUnavailable。"""
        ops = _load_ops()
        m = _ctx_get(ctx, "meta", {}) if meta is None else meta
        if not isinstance(m, dict):
            raise SandboxError("window 的 meta 参数必须是 dict")
        arr = ops.window(_as_image(img), float(center), float(width), m)
        _save_obs(arr, ctx, produced)
        return arr

    def measure(box: Any, meta: Any = None, mode: str = "long") -> float:
        """按像素间距量长度（mm）；缺 spacing 时 ops 会抛 ToolUnavailable。"""
        ops = _load_ops()
        b = _coerce_box(box)
        m = _ctx_get(ctx, "meta", {}) if meta is None else meta
        if not isinstance(m, dict):
            raise SandboxError("measure 的 meta 参数必须是 dict")
        value = float(ops.measure(b, m, mode=str(mode)))
        readings.append(f"measure({mode}) = {value:.3f} mm")
        return value

    def load_image(path: str) -> np.ndarray:
        """从磁盘读图（沙箱目录里的返回图等），只允许读白名单目录。"""
        ops = _load_ops()
        if not isinstance(path, str):
            raise SandboxError(f"load_image 需要 str 路径，收到 {type(path).__name__}")
        _check_read_path(path, ctx, read_roots)
        return ops.load_image(path)

    return {
        "crop": crop, "zoom": zoom, "window": window,
        "measure": measure, "load_image": load_image,
    }


def _check_read_path(path: str, ctx: Any, read_roots: Sequence[str]) -> None:
    """限制 load_image 的可读范围：默认只允许 ctx.out_dir（模型自己产出的返回图）。

    Python 层沙箱本身挡不住文件读取，这一层是补的，但也只是尽力而为；
    真正的隔离靠 runsc / 子进程 + 只读根目录。
    """
    roots: List[str] = []
    out_dir = str(_ctx_get(ctx, "out_dir", "") or "")
    if out_dir:
        roots.append(out_dir)
    roots.extend(str(r) for r in read_roots if r)
    real = os.path.realpath(path)
    for root in roots:
        base = os.path.realpath(root)
        if real == base or real.startswith(base + os.sep):
            return
    raise SandboxError(
        f"load_image 只允许读返回图目录，拒绝路径 {path!r}（允许的根：{roots or '无'}）"
    )


def _build_namespace(
    ctx: Any,
    cfg: Any,
    sb: Dict[str, Any],
    buf: io.StringIO,
    produced: List[str],
    readings: List[str],
    crop_boxes: List[List[float]],
) -> Dict[str, Any]:
    """组装 exec 用的全局命名空间。"""
    ns: Dict[str, Any] = {"__builtins__": _make_builtins(buf)}
    read_roots = [str(r) for r in (sb.get("read_roots") or [])]
    ns.update(_make_tools(ctx, produced, readings, crop_boxes, read_roots))

    image = _ctx_get(ctx, "image", None)
    ns["image"] = np.array(image, copy=True) if image is not None else None
    ns["Box"] = Box
    ns["np"] = _make_np_facade()

    # math / statistics 常驻；其它 allow_imports 里的安全模块按需注入
    seed = _cfg_get(cfg, "seed", 0)
    try:
        seed_i = int(seed) if seed is not None else 0
    except (TypeError, ValueError):
        seed_i = 0
    for name in sb["allow_imports"]:
        if name in ("math", "statistics"):
            ns[name] = importlib.import_module(name)
            continue
        if name == "random":
            # 用独立实例 + 显式 seed，避免污染进程全局随机状态
            mod = importlib.import_module("random")
            ns["random"] = mod.Random(seed_i)
            continue
        if name not in _SAFE_MODULES:
            warnings.warn(
                f"[sandbox] allow_imports 里的 {name!r} 不在安全模块白名单内，已跳过注入",
                RuntimeWarning, stacklevel=3,
            )
            continue
        try:
            ns[name] = importlib.import_module(name)
        except ImportError:
            warnings.warn(f"[sandbox] 预注入模块 {name!r} 导入失败，已跳过", RuntimeWarning, stacklevel=3)
    return ns


# ---------------------------------------------------------------- 执行


class _SandboxTimeout(Exception):
    """内部用：signal 超时。"""


def _on_alarm(signum: int, frame: Any) -> None:
    raise _SandboxTimeout()


def _in_main_thread() -> bool:
    return threading.current_thread() is threading.main_thread()


def _exec_with_alarm(code: str, ns: Dict[str, Any], timeout_s: float) -> None:
    """在主线程里用 SIGALRM 限时 exec；非主线程调用方必须先走子进程。"""
    if not _in_main_thread() or not hasattr(signal, "setitimer"):
        raise SandboxError("当前线程无法使用 signal 超时，应改走子进程后端")
    old_handler = signal.signal(signal.SIGALRM, _on_alarm)
    old_delay, old_interval = signal.setitimer(signal.ITIMER_REAL, max(float(timeout_s), 0.001))
    try:
        exec(compile(code, "<model_code>", "exec"), ns)  # noqa: S102 - 受限命名空间
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_delay > 0:
            signal.setitimer(signal.ITIMER_REAL, old_delay, old_interval)


def _short_repr(value: Any, limit: int = _MAX_LIST_REPR) -> str:
    try:
        s = repr(value)
    except Exception:  # noqa: BLE001
        s = f"<{type(value).__name__} 对象>"
    return s if len(s) <= limit else s[:limit] + "…"


def _format_result(
    ns: Dict[str, Any],
    ctx: Any,
    buf: io.StringIO,
    produced: List[str],
    readings: List[str],
) -> Tuple[str, Optional[str]]:
    """把 result 变量 + 捕获的 stdout + 测量读数拼成 text，并给出最后一张返回图。"""
    parts: List[str] = []
    stdout = buf.getvalue().strip()
    if stdout:
        parts.append(stdout)

    value = ns.get("result", _MISSING)
    if isinstance(value, np.ndarray):
        path = _save_obs(value, ctx, produced)
        parts.append(f"result = ndarray shape={value.shape} dtype={value.dtype} -> {path}")
    elif value is not _MISSING:
        parts.append(f"result = {_short_repr(value)}")

    if readings:
        parts.append("\n".join(readings))

    text = "\n".join(p for p in parts if p)[:_MAX_TEXT_CHARS]
    return text, (produced[-1] if produced else None)


class _Missing:
    """哨兵：区分 result 未赋值与 result = None。"""


_MISSING = _Missing()


def _run_inprocess(code: str, ctx: Any, cfg: Any, sb: Dict[str, Any], meta: Dict[str, Any]) -> ToolResult:
    """在当前进程里受限 exec（已通过 static_check）。"""
    t0 = time.time()
    buf = io.StringIO()
    produced: List[str] = []
    readings: List[str] = []
    crop_boxes: List[List[float]] = []
    ns = _build_namespace(ctx, cfg, sb, buf, produced, readings, crop_boxes)

    timeout_s = sb["timeout_s"]
    try:
        _exec_with_alarm(code, ns, timeout_s)
    except _SandboxTimeout:
        meta.update(timed_out=True, elapsed=time.time() - t0,
                    stdout=buf.getvalue(), images=list(produced))
        return ToolResult(ok=False, error=f"执行超时（>{timeout_s:g}s）", meta=meta)
    except SandboxError as e:
        meta.update(elapsed=time.time() - t0, stdout=buf.getvalue(), images=list(produced))
        return ToolResult(ok=False, error=str(e), meta=meta)
    except Exception as e:  # noqa: BLE001 - 模型代码的任何异常都要变成 ok=False
        meta.update(elapsed=time.time() - t0, stdout=buf.getvalue(), images=list(produced))
        return ToolResult(ok=False, error=f"{type(e).__name__}: {e}", meta=meta)

    text, image = _format_result(ns, ctx, buf, produced, readings)
    meta.update(
        elapsed=time.time() - t0,
        stdout=buf.getvalue(),
        images=list(produced),
        measure=list(readings),
        crop_boxes=crop_boxes,
    )
    return ToolResult(ok=True, image=image, text=text, meta=meta)


# ---------------------------------------------------------------- 子进程后端


def _jsonable(obj: Any) -> Any:
    """尽量把 ctx.meta 之类的对象转成可 JSON 序列化的结构。"""
    try:
        return json.loads(json.dumps(obj, default=str, ensure_ascii=False))
    except Exception:  # noqa: BLE001
        return {}


def _find_runsc(sb: Dict[str, Any]) -> Optional[str]:
    """定位 runsc：先看 cfg，再看 PATH。"""
    cand = str(sb.get("runsc_path") or "").strip()
    if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
        return cand
    return shutil.which("runsc")


def _warn_degrade(reason: str) -> None:
    """降级告警：stderr + warnings，绝不静默假装隔离。"""
    msg = (
        f"[sandbox][WARNING] {reason}；已降级为受限 exec（AST 白名单 + 受限内建 + 超时），"
        f"这不是进程/系统调用级隔离"
    )
    print(msg, file=sys.stderr, flush=True)
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


def _run_via_subprocess(
    code: str,
    ctx: Any,
    cfg: Any,
    sb: Dict[str, Any],
    meta: Dict[str, Any],
    prefix: Sequence[str],
    backend_used: str,
) -> Optional[ToolResult]:
    """在子进程里跑（可套 runsc）。返回 None 表示子进程压根起不来，调用方应降级。"""
    t0 = time.time()
    timeout_s = sb["timeout_s"]

    job_dir = tempfile.mkdtemp(prefix="medtaco_sbx_")
    try:
        image = _ctx_get(ctx, "image", None)
        np.save(os.path.join(job_dir, "image.npy"), np.asarray(image) if image is not None else np.zeros((1, 1, 3), np.uint8))
        payload = {
            "code": code,
            "meta": _jsonable(_ctx_get(ctx, "meta", {})),
            "out_dir": str(_ctx_get(ctx, "out_dir", "") or ""),
            "obs_max_side": int(_ctx_get(ctx, "obs_max_side", 1024) or 1024),
            "jpeg_quality": int(_ctx_get(ctx, "jpeg_quality", 90) or 90),
            "timeout_s": timeout_s,
            "mem_mb": sb["mem_mb"],
            "cfg": {"seed": _cfg_get(cfg, "seed", 0),
                    "rl": {"sandbox": {k: sb[k] for k in (
                        "timeout_s", "allow_imports", "blacklist_attrs", "read_roots")}}},
        }
        with open(os.path.join(job_dir, "job.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        env = dict(os.environ)
        old_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = _REPO_ROOT + (os.pathsep + old_pp if old_pp else "")
        prefix = list(prefix)
        for k, v in _THREAD_CAP_ENV.items():
            env[k] = v
            if prefix and os.path.basename(prefix[0]).startswith("runsc"):
                # runsc 不会无条件透传宿主环境，显式声明
                prefix.extend(["--env", f"{k}={v}"])
        cmd = prefix + [sys.executable, "-m", "medtaco.tools.sandbox", job_dir]
        try:
            proc = subprocess.run(
                cmd, cwd=_REPO_ROOT, env=env, capture_output=True, text=True,
                timeout=timeout_s + 5.0,
            )
        except subprocess.TimeoutExpired:
            meta.update(backend_used=backend_used, timed_out=True, elapsed=time.time() - t0)
            return ToolResult(ok=False, error=f"执行超时（>{timeout_s:g}s，子进程被终止）", meta=meta)
        except OSError as e:
            meta["subprocess_error"] = str(e)
            return None

        result_path = os.path.join(job_dir, "result.json")
        if not os.path.exists(result_path):
            tail = ((proc.stderr or "") + (proc.stdout or "")).strip().splitlines()[-3:]
            meta.update(backend_used=backend_used, elapsed=time.time() - t0,
                        returncode=proc.returncode)
            return ToolResult(
                ok=False,
                error="沙箱子进程未产出结果：" + " | ".join(tail),
                meta=meta,
            )

        with open(result_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        meta.update(d.get("meta", {}) or {})
        meta["backend_used"] = backend_used
        meta.setdefault("elapsed", time.time() - t0)
        return ToolResult(
            ok=bool(d.get("ok")),
            image=d.get("image"),
            text=d.get("text", ""),
            error=d.get("error", ""),
            meta=meta,
        )
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


# 子进程必须压住 BLAS/OpenMP 的线程栈与 arena 预留：默认配置下 numpy 一导入
# 就占 ~2.7 GB 虚拟地址空间，RLIMIT_AS 会把 import 本身打成 MemoryError。
# 限成单线程后 numpy 导入后 VmSize ≈ 115 MB。
_THREAD_CAP_ENV: Dict[str, str] = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

# 内存下限：低于这个值连解释器 + numpy 都起不来，设了也只会让子进程崩溃
_MIN_MEM_MB: int = 256


def _apply_mem_limit(mem_mb: int) -> None:
    """只在子进程里设 RLIMIT_AS（对父进程不可逆，绝不能在主进程设）。"""
    try:
        import resource

        mb = int(mem_mb)
        if mb < _MIN_MEM_MB:
            print(f"[sandbox][WARNING] mem_mb={mb} 低于下限 {_MIN_MEM_MB} MB，已按下限设置",
                  file=sys.stderr, flush=True)
            mb = _MIN_MEM_MB
        limit = mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except Exception:  # noqa: BLE001 - 平台不支持就算了
        pass


def _child_main(job_dir: str) -> int:
    """子进程入口：python -m medtaco.tools.sandbox <job_dir>。"""
    with open(os.path.join(job_dir, "job.json"), "r", encoding="utf-8") as f:
        job = json.load(f)

    _apply_mem_limit(int(job.get("mem_mb", 2048)))

    ctx = types.SimpleNamespace(
        image=np.load(os.path.join(job_dir, "image.npy")),
        meta=job.get("meta", {}) or {},
        out_dir=job.get("out_dir", "") or "",
        obs_max_side=int(job.get("obs_max_side", 1024)),
        jpeg_quality=int(job.get("jpeg_quality", 90)),
    )
    cfg = job.get("cfg", {}) or {}
    sb = _sandbox_cfg(cfg)
    sb["timeout_s"] = float(job.get("timeout_s", sb["timeout_s"]))

    meta: Dict[str, Any] = {"backend_used": "subprocess(child)"}
    code = job.get("code", "")
    try:
        static_check(code, cfg)
    except SandboxError as e:
        res = ToolResult(ok=False, error=f"静态检查未通过：{e}", meta=meta)
    else:
        res = _run_inprocess(code, ctx, cfg, sb, meta)

    with open(os.path.join(job_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(res.to_json(), f, ensure_ascii=False, default=str)
    return 0


# ---------------------------------------------------------------- 对外入口


def run_code(code: str, ctx: Any, cfg: Any = None) -> ToolResult:
    """执行模型写的 Python 代码，任何情况下都不抛异常，统一返回 ToolResult。

    ctx 是 ``medtaco.tools.ops.ToolContext``（需有 image / meta / out_dir /
    obs_max_side / jpeg_quality 字段）；测试里可用 SimpleNamespace 代替。
    """
    sb = _sandbox_cfg(cfg)
    t0 = time.time()
    meta: Dict[str, Any] = {
        "backend_requested": sb["backend"],
        "backend_used": "",
        "timed_out": False,
        "elapsed": 0.0,
        "images": [],
        "stdout": "",
    }

    # ---- 1) 静态检查 ----
    try:
        static_check(code, cfg)
    except SandboxError as e:
        meta["backend_used"] = "none(static_check)"
        meta["elapsed"] = time.time() - t0
        return ToolResult(ok=False, error=f"静态检查未通过：{e}", meta=meta)

    # ---- 2) 选后端 ----
    prefix: Optional[List[str]] = None
    backend_used = "restricted_exec"

    if sb["backend"] == "gvisor":
        runsc = _find_runsc(sb)
        if runsc is None:
            _warn_degrade("backend=gvisor 但未找到 runsc 可执行文件（gVisor 未安装）")
        else:
            bundle = str(sb.get("gvisor_bundle_dir") or "").strip()
            if bundle and not os.path.isdir(bundle):
                _warn_degrade(f"gvisor_bundle_dir={bundle!r} 不存在")
            else:
                # runsc do 共享宿主 FS，防护重点是系统调用面（见模块 docstring）
                prefix = [runsc, "do", "--network=none"]
                backend_used = "gvisor"
    elif sb["backend"] in ("subprocess", "process"):
        prefix = []
        backend_used = "subprocess"
    elif sb["backend"] in ("restricted_exec", "inprocess"):
        # 显式要求进程内受限 exec（无进程隔离，但路径最短）
        backend_used = "restricted_exec"
    else:
        _warn_degrade(f"未知 backend={sb['backend']!r}")

    # ---- 3) 子进程 -----
    if prefix is not None:
        res = _run_via_subprocess(code, ctx, cfg, sb, meta, prefix, backend_used)
        if res is not None:
            return res
        _warn_degrade(f"{backend_used} 子进程启动失败，回退")

    if not _in_main_thread():
        # 非主线程用不了 SIGALRM，只能走子进程
        res = _run_via_subprocess(code, ctx, cfg, sb, meta, [], "subprocess")
        if res is not None:
            return res
        meta["backend_used"] = "restricted_exec"
        meta["elapsed"] = time.time() - t0
        return ToolResult(ok=False, error="无法在非主线程启用信号超时，且子进程启动失败", meta=meta)

    # ---- 4) 受限 exec（降级后端）----
    meta["backend_used"] = "restricted_exec"
    return _run_inprocess(code, ctx, cfg, sb, meta)


if __name__ == "__main__":  # pragma: no cover - 子进程入口
    sys.exit(_child_main(sys.argv[1]))
