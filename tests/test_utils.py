"""基础模块单测：schema / config / io / stats。"""
import json
import math
import os

import pytest

from medtaco.utils.config import Cfg, load_config
from medtaco.utils.io import JsonlWriter, atomic_write_json, iter_jsonl, seen_ids
from medtaco.utils.schema import Box, Probe, Sample, Step, ToolCall, ToolResult, Trajectory
from medtaco.utils.stats import (
    bootstrap_ci,
    deff,
    effective_n,
    judge_attenuation,
    mcnemar,
    mcnemar_power_n,
    paired_bootstrap,
    wilson_ci,
)


# ---------------------------------------------------------------- schema
def test_box_basic_and_clip():
    b = Box(10, 20, 110, 220, "nodule")
    assert b.w == 100 and b.h == 200 and b.area == 20000
    assert b.center == (60, 120)
    c = b.clip(50, 50)
    assert c.as_tuple() == (10, 20, 50, 50)
    with pytest.raises(ValueError):
        Box(10, 10, 5, 5)


def test_sample_roundtrip():
    s = Sample(id="x1", image="/a.png", question="q", answer="a", tier="A",
               answer_type="short", boxes=[Box(1, 2, 3, 4, "lesion")],
               meta={"spacing": [0.7, 0.7]})
    s2 = Sample.from_json(json.loads(json.dumps(s.to_json())))
    assert s2.id == "x1" and s2.boxes[0].label == "lesion"
    assert s2.meta["spacing"] == [0.7, 0.7]


def test_trajectory_roundtrip_and_n_calls():
    t = Trajectory(
        id="t1", sample_id="s1",
        steps=[
            Step(text="想一下"),
            Step(text="裁剪", call=ToolCall("crop", {"box": [1, 2, 3, 4]}),
                 result=ToolResult(ok=True, image="/c.png", meta={"crop_box": [1, 2, 3, 4]})),
            Step(text="答案"),
        ],
        probes=[Probe(0, "old", 0), Probe(1, "new", 1)],
        delta_k=[1], delta_total=1,
    )
    assert t.n_calls == 1
    t2 = Trajectory.from_json(json.loads(json.dumps(t.to_json())))
    assert t2.n_calls == 1 and t2.steps[1].call.name == "crop"
    assert t2.probes[1].score == 1


# ---------------------------------------------------------------- config
def test_config_dotted_access(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("rl:\n  alpha_1: 1.0\n  nested:\n    x: 3\n", encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg.rl.alpha_1 == 1.0
    assert cfg.rl.nested.x == 3
    assert cfg.get("rl.nope", "d") == "d"
    cfg.set("rl.alpha_2", 0.15)
    assert cfg.rl.alpha_2 == 0.15


def test_default_config_loads():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(os.path.join(here, "configs", "default.yaml"))
    assert cfg.rl.alpha_2 == 0.15
    assert cfg.teacher.model == "claude-opus-5"
    assert cfg.data.datasets["SLAKE"]["tier"] == "A"
    assert isinstance(cfg.qgen.mix, Cfg)


def test_config_interpolates_paths(tmp_path):
    """``${paths.root}/raw`` 必须拼成真路径，否则下游会建一个叫 ${paths.root} 的目录。"""
    p = tmp_path / "c.yaml"
    p.write_text(
        "paths:\n  root: /data/med\n  raw: ${paths.root}/raw\n"
        "  nested: ${paths.raw}/sub\n  log: ${paths.missing}/x.log\n",
        encoding="utf-8",
    )
    cfg = load_config(str(p))
    assert cfg.paths.raw == "/data/med/raw"
    assert cfg.paths.nested == "/data/med/raw/sub"          # 链式引用
    assert cfg.paths.log == "${paths.missing}/x.log"        # 未定义引用原样保留


def test_config_reference_cycle_raises(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: ${b}\nb: ${a}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="成环"):
        load_config(str(p))


def test_config_interpolation_after_overrides(tmp_path):
    """覆盖发生在插值之后：--set paths.root=... 不会反向改写已展开的值。

    这是刻意的——让覆盖"只影响它自己那一项"比重新跑一遍插值可预测。
    """
    p = tmp_path / "c.yaml"
    p.write_text("paths:\n  root: /a\n  raw: ${paths.root}/raw\n", encoding="utf-8")
    cfg = load_config(str(p), {"paths.root": "/b"})
    assert cfg.paths.root == "/b"
    assert cfg.paths.raw == "/a/raw"


# ---------------------------------------------------------------- io
def test_jsonl_writer_resume(tmp_path):
    p = str(tmp_path / "o.jsonl")
    with JsonlWriter(p) as w:
        assert w.write({"id": "a", "v": 1})
        assert w.write({"id": "b", "v": 2})
    assert len(list(iter_jsonl(p))) == 2
    # 续跑：已存在的 id 被跳过
    with JsonlWriter(p) as w:
        assert not w.write({"id": "a", "v": 9})
        assert w.write({"id": "c", "v": 3})
    rows = list(iter_jsonl(p))
    assert [r["id"] for r in rows] == ["a", "b", "c"]
    assert rows[0]["v"] == 1  # 未被覆盖
    assert seen_ids(p) == {"a", "b", "c"}


def test_jsonl_bad_line_quarantined(tmp_path):
    p = str(tmp_path / "o.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        f.write('{"id":"a"}\n')
        f.write("{broken\n")
        f.write('{"id":"b"}\n')
    assert [r["id"] for r in iter_jsonl(p)] == ["a", "b"]
    assert os.path.exists(p + ".bad")


def test_atomic_write_json(tmp_path):
    p = str(tmp_path / "x" / "y.json")
    s = Sample(id="q", image="i", question="?", answer="!")
    atomic_write_json(p, s)
    assert json.load(open(p, encoding="utf-8"))["id"] == "q"
    assert not os.path.exists(p + ".tmp")


# ---------------------------------------------------------------- stats
def test_wilson_ci_matches_known_value():
    lo, hi = wilson_ci(316, 451)  # p≈0.70
    assert abs((hi - lo) / 2 - 0.042) < 0.006   # PROPOSAL 里写的 ±4.2pp


def test_mcnemar_basic():
    chi2, p = mcnemar(0, 0)
    assert chi2 == 0 and p == 1.0
    chi2, p = mcnemar(30, 5)
    assert p < 0.001
    _, p2 = mcnemar(10, 10)
    assert p2 > 0.9


def test_power_matches_proposal():
    # PROPOSAL §8.3：3pp/10% 不一致 -> ~870；5pp -> ~312
    assert 700 < mcnemar_power_n(0.03, 0.10) < 1000
    assert 250 < mcnemar_power_n(0.05, 0.10) < 400


def test_deff_and_effective_n():
    assert deff(1, 0.2) == 1.0
    assert abs(deff(5, 0.2) - 1.8) < 1e-9
    assert abs(effective_n(1000, 5, 0.2) - 1000 / 1.8) < 1e-6


def test_bootstrap_deterministic():
    v = [1.0, 0, 1, 1, 0, 1, 1, 1]
    a = bootstrap_ci(v, n_boot=500, seed=7)
    b = bootstrap_ci(v, n_boot=500, seed=7)
    assert a == b
    assert a[1] <= a[0] <= a[2]


def test_paired_bootstrap_and_attenuation():
    a = [1, 1, 1, 0, 1, 0, 1, 1]
    b = [1, 0, 1, 0, 0, 0, 1, 0]
    est, lo, hi = paired_bootstrap(a, b, n_boot=500, seed=1)
    assert abs(est - (sum(a) - sum(b)) / len(a)) < 1e-12
    assert abs(judge_attenuation(0.05, 0.9, 0.9) - 0.04) < 1e-9
