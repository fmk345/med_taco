"""medtaco.synth.qgen 单测：五个生成器 + 三道过滤闸。

约束（与全仓一致）：不联网、不读大数据、不依赖本地未安装的包。
图像仅用 PIL 在 tmp_path 造 64x64 随机数组；生成器本身不读像素，
但这些样本走的是真实的 Sample 结构，下游可直接复用。
"""
import json
import random

import pytest
from PIL import Image

from medtaco.synth import qgen
from medtaco.utils.config import load_config
from medtaco.utils.schema import Box, Sample


# ---------------------------------------------------------------- fixture


@pytest.fixture()
def cfg():
    """真实配置：确保测试跑的是 configs/default.yaml 的口径。"""
    return load_config("configs/default.yaml")


def _mk_images(tmp_path, n):
    """造 n 张 64x64 随机小图，返回路径列表。"""
    paths = []
    rnd = random.Random(20261028)
    for i in range(n):
        arr = [[(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256))
                for _ in range(64)] for _ in range(64)]
        p = tmp_path / f"img_{i}.png"
        Image.frombytes("RGB", (64, 64), bytes(c for row in arr for px in row for c in px)).save(p)
        paths.append(str(p))
    return paths


def _mk_samples(tmp_path, n=8, with_spacing=True, with_boxes=True):
    """构造 n 个带/不带框与 spacing 的样本。"""
    paths = _mk_images(tmp_path, n)
    out = []
    for i, p in enumerate(paths):
        boxes = [Box(16, 16, 48, 48, "nodule")] if with_boxes else []
        meta = {
            "image_size": [64, 64],
            "finding": "lung nodule",
            "organ": "right lung",
        }
        if with_spacing:
            meta["pixel_spacing"] = [0.7, 0.7]
        out.append(Sample(id=f"img{i}", image=p, question="", answer="",
                          dataset="MS-CXR", tier="A", boxes=boxes, meta=meta))
    return out


@pytest.fixture()
def samples(tmp_path):
    return _mk_samples(tmp_path, 8)


@pytest.fixture()
def samples_no_spacing(tmp_path):
    return _mk_samples(tmp_path, 8, with_spacing=False)


# ---------------------------------------------------------------- 模板库


def test_template_library_bilingual_and_complete():
    """每类 >= 8 条中/英模板；占位符只用白名单里的四个。"""
    assert set(qgen.TEMPLATES) == {"G1", "G2", "G3", "G4", "G5"}
    for gen, tpls in qgen.TEMPLATES.items():
        for lang in ("zh", "en"):
            sub = [t for t in tpls if t.lang == lang]
            assert len(sub) >= 8, f"{gen}/{lang} 模板数 {len(sub)} < 8"
        tids = [t.tid for t in tpls]
        assert len(tids) == len(set(tids)), f"{gen} 模板 id 重复"
        used_all = set()
        for t in tpls:
            assert t.gen == gen
            used = {m.group(1) for m in qgen._PLACEHOLDER_RE.finditer(t.text)}
            assert used <= set(qgen.PLACEHOLDERS), f"{t.tid} 用了未定义占位符 {used}"
            used_all |= used
        # 单条可以是通用问法（如"被框出的区域对应哪个结构"）。
        # G3/G4 是全局或纯测量问法，本来就不需要填写位；其余类要真的用到。
        if gen in ("G1", "G2", "G5"):
            assert len(used_all) >= 2, f"{gen} 只用到占位符 {used_all}"


def test_g4_templates_require_spacing_and_g3_is_tool_free():
    """G4 必须依赖 pixel_spacing；G3 模板不得指代标记区域。"""
    assert all(t.requires_spacing for t in qgen.TEMPLATES["G4"])
    for t in qgen.TEMPLATES["G3"]:
        assert not qgen._G1_ANCHOR.search(t.text), f"G3 模板 {t.tid} 指代了标记区域"


# ---------------------------------------------------------------- 五个生成器（无 model_fn）


def test_all_generators_run_without_model_fn(samples, cfg):
    """G1..G5 在无 API、无网络时都能产出可用问题。"""
    for gen, fn in qgen._GEN_FNS.items():
        out = fn(samples, cfg, None)
        assert out, f"{gen} 没有产出"
        for s in out:
            assert isinstance(s.question, str) and s.question.strip()
            assert s.meta["generator"] == gen
            assert s.meta["template_id"].startswith(gen)
            assert "is_negative" in s.meta
            assert s.meta["rewritten"] is False


def test_g1_outputs_carry_boxes_and_mark_region(samples, cfg):
    """G1 必须带框，且问题里明确指代标记区域（构造性蕴含裁剪）。"""
    out = qgen.g1_box_anchored(samples, cfg, None)
    assert out
    for s in out:
        assert s.boxes, "G1 产出丢了 box"
        assert s.meta["need_crop"] is True
        assert s.meta["anchor_box"] is not None
        assert qgen._G1_ANCHOR.search(s.question), f"G1 问题缺少锚点: {s.question}"


def test_g1_skips_boxless_samples(tmp_path, cfg):
    """没有框就锚定不了，G1 应跳过而不是造出无锚问题。"""
    boxless = _mk_samples(tmp_path, 3, with_boxes=False)
    assert qgen.g1_box_anchored(boxless, cfg, None) == []
    # G3 不受影响：它本来就覆盖无框样本（B 层数据的用法）
    assert qgen.g3_negative(boxless, cfg, None)


def test_g2_nested_boxes(samples, cfg):
    """G2 产出 2~3 级嵌套框，最内层是 gold 框。"""
    out = qgen.g2_multi_scale(samples, cfg, None)
    assert out
    for s in out:
        n = s.meta["n_scales"]
        assert 2 <= n <= 3
        assert len(s.boxes) == n
        assert len(s.meta["scale_boxes"]) == n
        assert list(s.boxes[-1].as_tuple()) == list(s.meta["anchor_box"][:4])
        # 外层面积不平庸：放大框应严格包含 gold 框
        outer = s.boxes[0]
        gold = s.boxes[-1]
        assert outer.x0 <= gold.x0 and outer.y0 <= gold.y0
        assert outer.x1 >= gold.x1 and outer.y1 >= gold.y1


def test_g3_is_negative_and_boxless(samples, cfg):
    """G3 是全局/知识题：清空 boxes，标 is_negative，不指代标记区域。"""
    out = qgen.g3_negative(samples, cfg, None)
    assert out
    for s in out:
        assert s.meta["is_negative"] is True
        assert s.meta["tool_free"] is True
        assert s.meta["need_crop"] is False
        assert s.boxes == []
        assert not qgen._G1_ANCHOR.search(s.question), f"G3 问题不应指代标记区域: {s.question}"


def test_g4_measurement_answer_derived_from_spacing(samples, cfg):
    """G4 答案由框角点 × spacing 确定性派生，answer_type=measure。"""
    out = qgen.g4_measure(samples, cfg, None)
    assert out
    for s in out:
        assert s.answer_type == "measure"
        assert s.answer.endswith("mm") or s.answer.endswith("mm^2")
        sx, sy = s.meta["pixel_spacing"]
        assert [sx, sy] == [0.7, 0.7]
        mode = s.meta["measure_mode"]
        box = Box.from_json(s.meta["measure_box"])
        if mode == "long":
            expect = box.w * sx
        elif mode == "short":
            expect = box.h * sy
        elif mode == "area":
            expect = box.w * sx * box.h * sy
        else:  # diag
            expect = ((box.w * sx) ** 2 + (box.h * sy) ** 2) ** 0.5
        # derived_mm 落库前做过 round(.,4)，容差按 4 位小数给
        assert abs(s.meta["derived_mm"] - expect) < 1e-3
        assert abs(float(s.answer.split()[0]) - expect) < 0.05 + 1e-9


def test_g4_requires_spacing(samples_no_spacing, cfg):
    """没有 pixel_spacing 就不能算 mm，G4 必须跳过。"""
    assert qgen.g4_measure(samples_no_spacing, cfg, None) == []


def test_g5_has_mirror_compare_box(samples, cfg):
    """G5 带上下文标记与对侧镜像框。"""
    out = qgen.g5_compare(samples, cfg, None)
    assert out
    for s in out:
        assert s.meta["needs_global_context"] is True
        assert s.meta["compare_box"] is not None
        assert len(s.boxes) == 2


# ---------------------------------------------------------------- model_fn 注入


def test_model_fn_rewrites_question(samples, cfg):
    """注入 model_fn 时用模型输出改写；G1 保留锚点则采纳。"""
    calls = []

    def fake_model(prompt, images):
        calls.append((prompt, images))
        return "在标记框内，该病灶的边界是否清晰？"

    out = qgen.g1_box_anchored(samples[:1], cfg, fake_model)
    assert out
    assert calls and calls[0][1] == [samples[0].image]  # 图路径原样传给模型
    assert out[0].meta["rewritten"] is True
    assert out[0].question == "在标记框内，该病灶的边界是否清晰？"


def test_g1_rewrite_losing_anchor_falls_back(samples, cfg):
    """G1 改写若丢掉"标记区域"的指代，裁剪就不再被蕴含 -> 回退模板原文。"""

    def bad_model(prompt, images):
        return "这个病灶的边界是否清晰？"

    out = qgen.g1_box_anchored(samples[:1], cfg, bad_model)
    assert out
    assert out[0].meta["rewritten"] is False
    assert qgen._G1_ANCHOR.search(out[0].question)
    assert "这个病灶的边界是否清晰？" != out[0].question


def test_model_fn_exception_falls_back(samples, cfg):
    """模型报错/空串不能把整批打断，静默回落模板。"""

    def boom(prompt, images):
        raise RuntimeError("network down")

    def empty(prompt, images):
        return "   "

    for fn in (boom, empty):
        out = qgen.g3_negative(samples[:1], cfg, fn)
        assert out and all(not s.meta["rewritten"] for s in out)


# ---------------------------------------------------------------- 比例与确定性


def test_generate_ratios_and_meta(samples, cfg):
    """generate 的产出：正负 ~60:40、yes/no < 5%、meta 三键齐全。"""
    out = qgen.generate(samples, cfg)
    assert out
    st = qgen.question_stats(out)
    assert abs(st["negative_ratio"] - cfg.qgen.target_negative_ratio) <= 0.02
    assert st["yesno_ratio"] < cfg.qgen.yes_no_max_ratio
    assert st["n_yes_no"] == sum(1 for s in out if s.meta.get("yes_no"))
    for s in out:
        for k in ("generator", "is_negative", "template_id"):
            assert k in s.meta
    # 正样本都带框，负样本都不带
    for s in out:
        if s.meta["is_negative"]:
            assert not s.boxes
        else:
            assert s.boxes
    # id 唯一
    ids = [s.id for s in out]
    assert len(ids) == len(set(ids))


def test_generate_deterministic_same_seed(samples, cfg):
    """同 seed 两次生成完全一致。"""
    a = qgen.generate(samples, cfg)
    b = qgen.generate(samples, cfg)
    assert [s.id for s in a] == [s.id for s in b]
    assert [s.question for s in a] == [s.question for s in b]
    assert [s.meta["generator"] for s in a] == [s.meta["generator"] for s in b]


def test_generate_deterministic_with_model_fn(samples, cfg):
    """改写路径下也要确定（模型 mock 固定输出）。"""

    def fake(prompt, images):
        return (prompt.split("原问题：")[-1].strip() + " 请说明依据。")

    a = qgen.generate(samples[:3], cfg, fake)
    b = qgen.generate(samples[:3], cfg, fake)
    assert [s.question for s in a] == [s.question for s in b]


def test_generate_seed_changes_order(samples, cfg):
    """换 seed 后顺序/取样可以不同（证明 seed 真的接进去了）。"""
    cfg2 = load_config("configs/default.yaml")
    cfg2.set("qgen.seed", 7)
    a = qgen.generate(samples, cfg)
    b = qgen.generate(samples, cfg2)
    assert [s.id for s in a] != [s.id for s in b]


def test_generate_handles_missing_g4_capacity(samples_no_spacing, cfg):
    """G4 无产能时应把配额让给其他生成器，而不是产出空问题。"""
    out = qgen.generate(samples_no_spacing, cfg)
    assert out
    assert "G4" not in {s.meta["generator"] for s in out}


def test_generate_empty_input(cfg):
    assert qgen.generate([], cfg) == []


def test_question_stats_fields(samples, cfg):
    out = qgen.generate(samples, cfg)
    st = qgen.question_stats(out)
    for k in ("n", "n_positive", "n_negative", "negative_ratio",
              "n_yes_no", "yesno_ratio", "n_rewritten", "by_generator", "by_lang"):
        assert k in st
    assert st["n"] == st["n_positive"] + st["n_negative"]
    assert sum(st["by_generator"].values()) == st["n"]


# ---------------------------------------------------------------- 过滤闸


def _qout(samples, cfg):
    return qgen.generate(samples, cfg)


def test_filter_gate_agree_routes_disagreement(samples, cfg):
    """闸 1：双模型不一致 -> arbitration（不丢，进人工/仲裁队列）。"""
    items = _qout(samples, cfg)
    for s in items:
        s.answer = s.answer or "yes"
    audit = qgen.filter_questions_with_audit(
        items, cfg, agree_fn=lambda a, b: False, base_model_fn=lambda p, im: "no",
    )
    assert len(audit.arbitration) == len(items)
    assert audit.kept == []
    assert all(s.meta["filter"]["route"] == "arbitration" for s in audit.arbitration)


def test_filter_gate_agree_kept_when_consistent(samples, cfg):
    items = _qout(samples, cfg)
    for s in items:
        s.answer = "yes"
    audit = qgen.filter_questions_with_audit(
        items, cfg, agree_fn=lambda a, b: True, base_model_fn=lambda p, im: "yes",
    )
    assert audit.arbitration == []
    assert audit.kept


def test_filter_gate_agree_skipped_without_fn(samples, cfg):
    """未注入 agree_fn -> 闸 1 跳过并记账，不影响后续闸门。"""
    items = _qout(samples, cfg)
    audit = qgen.filter_questions_with_audit(items, cfg)
    assert all(s.meta["filter"]["agree"] == "skipped" for s in audit.kept)
    assert audit.kept


def test_filter_gate_agree_skipped_on_missing_answer(samples, cfg):
    """教师答案未回填（answer 为空）时不能瞎判一致，记 missing_answer。"""
    items = _qout(samples, cfg)
    for s in items:
        s.answer = ""
    audit = qgen.filter_questions_with_audit(
        items, cfg, agree_fn=lambda a, b: True, base_model_fn=lambda p, im: "yes",
    )
    seen = [s for s in audit.kept if s.meta["filter"].get("agree_note") == "missing_answer"]
    assert seen


def test_filter_gate_necessity_moves_to_negative_pool(samples, cfg):
    """闸 2：无工具 pass@8 全对 -> 转负样本池，is_negative 翻转、框清空。"""
    items = [s for s in _qout(samples, cfg) if not s.meta["is_negative"]]
    assert items
    audit = qgen.filter_questions_with_audit(items, cfg, pass8_fn=lambda s: True)
    assert len(audit.to_negative_pool) == len(items)
    assert audit.kept == []
    for s in audit.to_negative_pool:
        assert s.meta["is_negative"] is True
        assert s.meta["need_crop"] is False
        assert s.meta["tool_free"] is True
        assert s.boxes == []
        assert s.meta["filter"]["route"] == "to_negative_pool"


def test_filter_gate_necessity_mark_too_hard(samples, cfg):
    """闸 2（续）：给了 gold 框仍全错 -> too_hard 删除。"""
    items = [s for s in _qout(samples, cfg) if not s.meta["is_negative"]]
    audit = qgen.filter_questions_with_audit(
        items, cfg, pass8_fn=lambda s: False, with_box_fn=lambda s: False,
    )
    assert len(audit.too_hard) == len(items)
    assert audit.kept == []
    assert all(s.meta["filter"]["route"] == "too_hard" for s in audit.too_hard)


def test_filter_gate_necessity_keeps_when_hard_but_solvable(samples, cfg):
    items = [s for s in _qout(samples, cfg) if not s.meta["is_negative"]]
    audit = qgen.filter_questions_with_audit(
        items, cfg, pass8_fn=lambda s: False, with_box_fn=lambda s: True,
    )
    assert audit.too_hard == []
    assert audit.to_negative_pool == []
    assert audit.kept


def test_filter_gate_necessity_not_applicable_to_negatives(samples, cfg):
    """G3 负样本本就不该用工具，闸 2 直接 not_applicable。"""
    items = [s for s in _qout(samples, cfg) if s.meta["is_negative"]]
    assert items
    audit = qgen.filter_questions_with_audit(items, cfg, pass8_fn=lambda s: True)
    assert audit.to_negative_pool == []
    assert all(s.meta["filter"]["necessity"] == "not_applicable" for s in audit.kept)


def test_filter_gate_ratio_downsample(samples, cfg):
    """闸 3：比例失衡时按固定 seed 下采样，最终两条红线都满足。"""
    items = _qout(samples, cfg)
    # 人为打破比例：把负样本砍到只剩几条，正样本全留
    pos = [s for s in items if not s.meta["is_negative"]]
    neg = [s for s in items if s.meta["is_negative"]]
    skewed = pos + neg[:2]
    audit = qgen.filter_questions_with_audit(skewed, cfg)
    st = audit.stats
    assert st["n"]
    assert abs(st["negative_ratio"] - cfg.qgen.target_negative_ratio) <= 0.05
    assert audit.ratio_dropped
    assert all(s.meta["filter"]["route"] == "ratio_dropped" for s in audit.ratio_dropped)
    # 下采样是确定的：同输入两次结果一致
    audit2 = qgen.filter_questions_with_audit(skewed, cfg)
    assert [s.id for s in audit.kept] == [s.id for s in audit2.kept]
    assert [s.id for s in audit.ratio_dropped] == [s.id for s in audit2.ratio_dropped]


def test_filter_gate_ratio_caps_yes_no(samples, cfg):
    """闸 3（yes/no）：二元题超过 5% 时被换出到 ratio_dropped。"""
    items = [s for s in _qout(samples, cfg) if not s.meta["is_negative"]]
    # 人为把约 1/3 的正样本标成 yes/no，明显超过 5%
    for i, s in enumerate(items):
        s.meta["yes_no"] = (i % 3 == 0)
    n_yn_before = sum(1 for s in items if s.meta["yes_no"])
    assert n_yn_before > cfg.qgen.yes_no_max_ratio * len(items)

    audit = qgen.filter_questions_with_audit(items, cfg)
    st = audit.stats
    assert st["n"]
    assert st["yesno_ratio"] < cfg.qgen.yes_no_max_ratio
    assert audit.ratio_dropped
    # 被换出的都应原本是 yes/no 题
    assert all(s.meta["yes_no"] for s in audit.ratio_dropped)


def test_filter_does_not_mutate_input(samples, cfg):
    """过滤会写 meta/filter，必须先浅拷贝，别改调用方对象。"""
    items = _qout(samples, cfg)
    before = [json.dumps(s.meta, sort_keys=True, default=str) for s in items]
    qgen.filter_questions_with_audit(items, cfg, pass8_fn=lambda s: False)
    after = [json.dumps(s.meta, sort_keys=True, default=str) for s in items]
    assert before == after


def test_filter_questions_returns_kept_list(samples, cfg):
    """契约入口 filter_questions 只返回通过样本。"""
    items = _qout(samples, cfg)
    out = qgen.filter_questions(items, cfg)
    assert out and all(isinstance(s, Sample) for s in out)
    assert [s.id for s in out] == [
        s.id for s in qgen.filter_questions_with_audit(items, cfg).kept
    ]


# ---------------------------------------------------------------- 契约面


def test_contract_signatures():
    """INTERFACES.md 里的签名一个不少，且可被关键字调用。"""
    assert callable(qgen.generate)
    assert callable(qgen.filter_questions)
    for name in ("g1_box_anchored", "g2_multi_scale", "g3_negative",
                 "g4_measure", "g5_compare"):
        assert callable(getattr(qgen, name))
    import inspect

    sig = inspect.signature(qgen.generate)
    assert list(sig.parameters)[:3] == ["samples", "cfg", "model_fn"]
    assert sig.parameters["model_fn"].default is None

    sig = inspect.signature(qgen.filter_questions)
    assert list(sig.parameters)[:4] == ["samples", "cfg", "base_model_fn", "agree_fn"]
    assert sig.parameters["base_model_fn"].default is None
    assert sig.parameters["agree_fn"].default is None


def test_sample_json_roundtrip_after_generate(samples, cfg):
    """生成的问题能按 schema 序列化/反序列化（下游 jsonl 落盘的要求）。"""
    out = qgen.generate(samples[:2], cfg)
    assert out
    for s in out:
        s2 = Sample.from_json(json.loads(json.dumps(s.to_json())))
        assert s2.id == s.id and s2.question == s.question
        assert s2.meta["generator"] == s.meta["generator"]
        assert [b.as_tuple() for b in s2.boxes] == [b.as_tuple() for b in s.boxes]
