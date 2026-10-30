"""medtaco/tools/sandbox.py 的单测：不联网、不读大数据、不依赖未装的东西。

工具原语用的是真实的 medtaco.tools.ops（用它自带的小图），“ops 缺失”那条路径
单独用 monkeypatch 把注入点打成空来验证降级行为。
"""
from __future__ import annotations

import os
import sys
import time
import types
from typing import Any, Dict

import numpy as np
import pytest
from PIL import Image

from medtaco.tools import sandbox
from medtaco.utils.config import Cfg
from medtaco.utils.schema import ToolResult


# ---------------------------------------------------------------- 夹具


def make_cfg(**sandbox_overrides: Any) -> Cfg:
    """造一个只含沙箱相关字段的 Cfg（结构与 configs/default.yaml 一致）。"""
    sb: Dict[str, Any] = {
        "backend": "subprocess",
        "timeout_s": 5,
        "mem_mb": 512,
        "allow_imports": ["math", "statistics"],
        "blacklist_attrs": ["__subclasses__", "__globals__", "__builtins__", "__code__", "__class__"],
    }
    sb.update(sandbox_overrides)
    return Cfg({"seed": 42, "rl": {"sandbox": sb}})


@pytest.fixture()
def ctx(tmp_path: Any) -> types.SimpleNamespace:
    """64x64 随机图；spacing 取 1 mm/px 便于断言长度。"""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    return types.SimpleNamespace(
        image=img,
        meta={"pixel_spacing": (1.0, 1.0), "rescale_slope": 1.0,
              "rescale_intercept": -1024.0, "photometric": "MONOCHROME2"},
        out_dir=str(tmp_path / "obs"),
        obs_max_side=64,
        jpeg_quality=90,
    )


@pytest.fixture()
def real_ops() -> Any:
    """确保真实的 ops 模块可导入，并返回它。"""
    from medtaco.tools import ops

    return ops


# ---------------------------------------------------------------- 静态检查


def test_static_check_accepts_normal_code() -> None:
    code = (
        "a = 1 + 2 * 3\n"
        "vals = [x ** 2 for x in range(5) if x % 2 == 0]\n"
        "def f(x):\n"
        "    return x * 2\n"
        "b = f(3)\n"
        "result = {'a': a, 'b': b, 'n': sum(vals)}\n"
    )
    sandbox.static_check(code, make_cfg())  # 不抛即通过


def test_static_check_rejects_import() -> None:
    with pytest.raises(sandbox.SandboxError) as e:
        sandbox.static_check("import os", make_cfg())
    assert "禁止 import" in str(e.value)


def test_static_check_rejects_import_from() -> None:
    with pytest.raises(sandbox.SandboxError):
        sandbox.static_check("from os import path", make_cfg())


def test_static_check_rejects_dunder_attribute() -> None:
    with pytest.raises(sandbox.SandboxError) as e:
        sandbox.static_check("x = ().__class__", make_cfg())
    assert "__class__" in str(e.value)


def test_static_check_rejects_dunder_chain() -> None:
    """().__class__.__bases__[0].__subclasses__() 必须在第一跳就被拦。"""
    code = "x = ().__class__.__bases__[0].__subclasses__()"
    with pytest.raises(sandbox.SandboxError):
        sandbox.static_check(code, make_cfg())


def test_static_check_rejects_frame_attr() -> None:
    """不带双下划线但同样是逃逸电梯的属性也要拦。"""
    with pytest.raises(sandbox.SandboxError):
        sandbox.static_check("x = image.f_globals", make_cfg())


def test_static_check_rejects_dunder_name() -> None:
    with pytest.raises(sandbox.SandboxError):
        sandbox.static_check("x = __builtins__", make_cfg())


@pytest.mark.parametrize("bad", [
    "eval('1+1')",
    "exec('x=1')",
    "open('/etc/passwd')",
    "__import__('os')",
    "getattr(image, 'shape')",
    "type(image)",
    "while True:\n    pass",
    "with open('x') as f:\n    pass",
    "try:\n    x = 1\nexcept Exception:\n    x = 2",
    "import math",
    "f = lambda x: x",
])
def test_static_check_rejects_dangerous(bad: str) -> None:
    with pytest.raises(sandbox.SandboxError):
        sandbox.static_check(bad, make_cfg())


def test_static_check_rejects_syntax_error() -> None:
    with pytest.raises(sandbox.SandboxError) as e:
        sandbox.static_check("def f(:\n  pass", make_cfg())
    assert "语法错误" in str(e.value)


def test_static_check_rejects_empty() -> None:
    with pytest.raises(sandbox.SandboxError):
        sandbox.static_check("   \n", make_cfg())


# ---------------------------------------------------------------- 执行


def test_run_code_returns_result_variable(ctx: Any) -> None:
    res = sandbox.run_code("result = sum(i * i for i in range(4))", ctx, make_cfg())
    assert isinstance(res, ToolResult)
    assert res.ok, res.error
    assert "14" in res.text


def test_run_code_captures_print(ctx: Any) -> None:
    res = sandbox.run_code("print('hello'); result = 1", ctx, make_cfg())
    assert res.ok
    assert "hello" in res.text


def test_run_code_import_os_rejected(ctx: Any) -> None:
    res = sandbox.run_code("import os\nresult = os.getcwd()", ctx, make_cfg())
    assert not res.ok
    assert "禁止 import" in res.error
    assert res.meta["backend_used"] == "none(static_check)"


def test_run_code_dunder_rejected(ctx: Any) -> None:
    res = sandbox.run_code("result = ().__class__.__bases__", ctx, make_cfg())
    assert not res.ok
    assert not res.image


def test_run_code_runtime_error_is_not_raised(ctx: Any) -> None:
    res = sandbox.run_code("result = 1 / 0", ctx, make_cfg())
    assert not res.ok
    assert "ZeroDivisionError" in res.error


def test_run_code_timeout(ctx: Any) -> None:
    t0 = time.time()
    res = sandbox.run_code("for i in range(10 ** 12):\n    pass", ctx, make_cfg(timeout_s=0.5))
    elapsed = time.time() - t0
    assert not res.ok
    assert "超时" in res.error
    assert elapsed < 20, "超时没生效，测试被卡住了"


def test_run_code_math_injected(ctx: Any) -> None:
    res = sandbox.run_code("result = math.sqrt(16) + statistics.mean([1, 3])", ctx, make_cfg())
    assert res.ok, res.error
    assert "6.0" in res.text


def test_run_code_image_injected(ctx: Any) -> None:
    res = sandbox.run_code("result = image.shape", ctx, make_cfg())
    assert res.ok
    assert "(64, 64, 3)" in res.text


def test_run_code_crop_saves_image(ctx: Any, tmp_path: Any) -> None:
    code = "sub = crop(image, [10, 10, 40, 40])\nresult = sub.shape"
    res = sandbox.run_code(code, ctx, make_cfg())
    assert res.ok, res.error
    assert res.image is not None and os.path.exists(res.image)
    with Image.open(res.image) as im:
        assert im.size == (30, 30)
    assert res.meta["crop_boxes"] == [[10.0, 10.0, 40.0, 40.0]]


def test_run_code_measure_text(ctx: Any) -> None:
    res = sandbox.run_code("d = measure([0, 0, 20, 10])", ctx, make_cfg())
    assert res.ok, res.error
    assert "20.000" in res.text


def test_run_code_measure_unavailable_is_clean_error(ctx: Any) -> None:
    """缺 pixel_spacing 时 ops 抛 ToolUnavailable，要变成 ok=False 而不是崩掉。"""
    ctx.meta = {}
    res = sandbox.run_code("measure([0, 0, 20, 10])", ctx, make_cfg())
    assert not res.ok
    assert "pixel_spacing" in res.error


def test_run_code_ops_missing_degrades_gracefully(
    ctx: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ops 尚未就绪时，调用工具要返回 ok=False 而不是让 ImportError 冒出来。

    必须是进程内后端：子进程会重新 import，monkeypatch 传不进去。
    """
    monkeypatch.setattr(sandbox, "_OPS_MODULE", [None])
    monkeypatch.setitem(sys.modules, "medtaco.tools.ops", None)
    res = sandbox.run_code("crop(image, [0, 0, 10, 10])", ctx, make_cfg(backend="restricted_exec"))
    assert not res.ok
    assert "工具实现不可用" in res.error


def test_run_code_load_image_outside_out_dir_rejected(ctx: Any) -> None:
    """load_image 只允许读返回图目录，防止当成任意文件读探针。"""
    res = sandbox.run_code("load_image('/etc/passwd')", ctx, make_cfg())
    assert not res.ok
    assert "load_image" in res.error


def test_read_path_whitelist(ctx: Any) -> None:
    """目录白名单本身：out_dir 内放行，之外拒绝。"""
    os.makedirs(ctx.out_dir, exist_ok=True)
    inside = os.path.join(ctx.out_dir, "x.jpg")
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(inside)

    sandbox._check_read_path(inside, ctx, [])          # 不抛即放行
    with pytest.raises(sandbox.SandboxError):
        sandbox._check_read_path("/etc/passwd", ctx, [])
    # 显式 read_roots 里列出的目录也放行
    if os.path.exists("/etc/hostname"):
        sandbox._check_read_path("/etc/hostname", ctx, ["/etc"])


def test_read_path_rejects_prefix_lookalike(ctx: Any) -> None:
    """out_dir=/a/obs 时不能让 /a/obs_evil/... 通过（前缀必须带分隔符）。"""
    os.makedirs(ctx.out_dir, exist_ok=True)
    evil = ctx.out_dir + "_evil"
    os.makedirs(evil, exist_ok=True)
    try:
        with pytest.raises(sandbox.SandboxError):
            sandbox._check_read_path(os.path.join(evil, "x.jpg"), ctx, [])
    finally:
        os.rmdir(evil)


def test_run_code_backend_gvisor_degrades_with_warning(ctx: Any) -> None:
    if sandbox._find_runsc({"runsc_path": ""}) is not None:
        pytest.skip("本机装了 runsc，走不到降级分支")
    with pytest.warns(RuntimeWarning, match="降级"):
        res = sandbox.run_code("result = 1", ctx, make_cfg(backend="gvisor"))
    assert res.ok, res.error
    assert res.meta["backend_requested"] == "gvisor"
    assert res.meta["backend_used"] != "gvisor"


def test_run_code_subprocess_backend(ctx: Any) -> None:
    res = sandbox.run_code("result = 7 * 6", ctx, make_cfg(backend="subprocess"))
    assert res.ok, res.error
    assert res.meta["backend_used"] == "subprocess"
    assert "42" in res.text


def test_run_code_unknown_backend_degrades(ctx: Any) -> None:
    """不认识的 backend 名要告警并降级成受限 exec，实际后端写进 meta。"""
    with pytest.warns(RuntimeWarning, match="降级"):
        res = sandbox.run_code("result = 3", ctx, make_cfg(backend="docker"))
    assert res.ok, res.error
    assert res.meta["backend_used"] == "restricted_exec"
    assert "3" in res.text


def test_run_code_plain_dict_cfg(ctx: Any) -> None:
    """cfg 传普通 dict 也要能用（Cfg 是 dict 子类，走同一个 reader）。"""
    cfg = {"seed": 1, "rl": {"sandbox": {"backend": "subprocess", "timeout_s": 5}}}
    res = sandbox.run_code("result = 1 + 1", ctx, cfg)
    assert res.ok, res.error
    assert "2" in res.text


def test_static_check_allows_preinjected_module_use() -> None:
    """allow_imports 里的模块名可以直接当名字用，不需要 import 语句。"""
    sandbox.static_check("result = math.floor(3.7) + statistics.median([1, 2])", make_cfg())
