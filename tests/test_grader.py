"""medtaco/rl/grader.py + synonyms.py 的单测。

覆盖：同义词等价、否定极性、单位换算、两档容差、解析失败、幂等/确定性、
序列化 round-trip、外部 json 覆盖。不联网、不读大文件、不依赖 torch。
"""
from __future__ import annotations

import json

import pytest

from medtaco.rl.grader import DeterministicGrader, normalize
from medtaco.rl.synonyms import (
    DEFAULT_SYNONYMS,
    build_alias_map,
    merge_synonyms,
)
from medtaco.utils.schema import Sample


# ------------------------------------------------------------------ 脚手架


def S(answer: str, answer_type: str = "short", sid: str = "s0") -> Sample:
    """造一个只关心答案与类型的样本。"""
    return Sample(id=sid, image="", question="q", answer=answer, answer_type=answer_type)


@pytest.fixture()
def g() -> DeterministicGrader:
    return DeterministicGrader()


# ------------------------------------------------------------------ normalize


def test_normalize_basic() -> None:
    assert normalize("The  Masses.") == "mass"
    assert normalize("  A  NODULE  ") == "nodule"
    assert normalize("Effusions") == "effusion"
    assert normalize("studies") == "study"
    assert normalize("Atelectases") == "atelectase"  # 词表层面再兜底映射
    assert normalize("") == ""
    assert normalize(None) == ""  # type: ignore[arg-type]


def test_normalize_fullwidth_and_punct() -> None:
    # 全角字母/数字 -> 半角
    assert normalize("ＡＢＣ１２") == "abc12"
    # 中文标点被去掉，中文实体保留
    assert normalize("胸腔积液，未见气胸。") == "胸腔积液 未见气胸"
    assert normalize("no---effusion") == "no effusion"


def test_normalize_units_to_mm() -> None:
    assert normalize("1.5 cm") == "15 mm"
    assert normalize("15mm") == "15 mm"
    assert normalize("0.02 m") == "20 mm"
    assert normalize("1.5CM") == "15 mm"
    # 无单位数字不动
    assert normalize("15") == "15"


# ------------------------------------------------------------------ closed


def test_closed_exact_and_polarity(g: DeterministicGrader) -> None:
    assert g.score(S("yes", "closed"), "yes") == 1
    assert g.score(S("Yes.", "closed"), "YES") == 1
    assert g.score(S("yes", "closed"), "present") == 1      # affirm 等价类
    assert g.score(S("no", "closed"), "absent") == 1        # deny 等价类
    assert g.score(S("positive", "closed"), "present") == 1
    assert g.score(S("yes", "closed"), "no") == 0           # 极性相反判错
    assert g.score(S("正常", "closed"), "未见异常") == 1     # 中文阴性同义
    assert g.score(S("normal", "closed"), "unremarkable") == 1
    assert g.score(S("A", "closed"), "A") == 1
    assert g.score(S("A", "closed"), "B") == 0


def test_closed_negation_equivalence(g: DeterministicGrader) -> None:
    # "no evidence of pneumothorax" 与 "negative" 在 closed 口径下等价
    assert g.score(S("negative", "closed"), "no evidence of pneumothorax") == 1
    assert g.score(S("negative", "closed"), "pneumothorax") == 0
    # "正常/正常胸片" 视作否定极
    assert g.score(S("no", "closed"), "normal") == 1
    assert g.score(S("no", "closed"), "正常") == 1
    assert g.score(S("yes", "closed"), "normal") == 0


# ------------------------------------------------------------------ short


def test_short_synonyms_equivalent(g: DeterministicGrader) -> None:
    assert g.score(S("pleural effusion"), "effusion") == 1
    assert g.score(S("effusion"), "pleural effusion") == 1
    assert g.score(S("Effusions."), "pleural effusion") == 1
    assert g.score(S("胸腔积液"), "pleural effusion") == 1
    assert g.score(S("cardiomegaly"), "enlarged heart") == 1
    assert g.score(S("心影增大"), "cardiomegaly") == 1
    assert g.score(S("肺不张"), "atelectasis") == 1
    assert g.score(S("consolidation"), "consolidative change") == 1
    # 不应过度合并：不同分级不算同义
    assert g.score(S("mass"), "nodule") == 0
    assert g.score(S("pneumothorax"), "pneumonia") == 0


def test_short_bidirectional_contains(g: DeterministicGrader) -> None:
    assert g.score(S("left pleural effusion"), "pleural effusion") == 1
    assert g.score(S("pleural effusion"), "moderate left pleural effusion") == 1
    # 语序无关：实词集合一致即等价（"in"/"of" 等虚词不参与）
    assert g.score(S("nodule in right upper lobe"), "right upper lobe nodule") == 1
    assert g.score(S("consolidation of the left lower lobe"), "left lower lobe consolidation") == 1
    # 但部位不同仍判错：实词集合不是子集
    assert g.score(S("right upper lobe nodule"), "left lower lobe nodule") == 0
    # 过短的串只允许完全相等，防止 "pe" 命中一切
    assert g.score(S("pe"), "pulmonary embolism") == 0


def test_short_negation_must_match(g: DeterministicGrader) -> None:
    assert g.score(S("effusion"), "no effusion") == 0
    assert g.score(S("no effusion"), "effusion") == 0
    assert g.score(S("no pleural effusion"), "pleural effusion") == 0
    assert g.score(S("no effusion"), "no effusion") == 1
    assert g.score(S("未见胸腔积液"), "no pleural effusion") == 1
    assert g.score(S("未见胸腔积液"), "胸腔积液") == 0
    assert g.score(S("without atelectasis"), "no atelectasis") == 1
    assert g.score(S("absent"), "no effusion") == 0
    assert g.score(S("absence of effusion"), "no effusion") == 1
    # 两侧都是"裸否定"（剥离后不再剩实体）：同为阴性即算对上
    assert g.score(S("none", "closed"), "none") == 1
    assert g.score(S("no effusion"), "absent") == 0
    # 描述里的 not / negative 也算否定
    assert g.score(S("effusion is not present"), "effusion") == 0
    # 长句里的否定短语与中文否定都要识别
    assert g.score(S("no effusion"), "there is no evidence of pleural effusion") == 1
    assert g.score(S("no pneumothorax"), "未见气胸") == 1
    assert g.score(S("气胸"), "无气胸") == 0
    assert g.score(S("无气胸"), "气胸") == 0
    # 裸「不」不是否定标记：肺不张 不能被剥成"没有不张"
    assert g.score(S("atelectasis"), "肺不张") == 1
    assert g.analyze("肺不张")[1] is False
    assert g.analyze("未见气胸")[1] is True


# ------------------------------------------------------------------ measure


def test_measure_unit_conversion(g: DeterministicGrader) -> None:
    assert g.score(S("15 mm", "measure"), "1.5 cm") == 1
    assert g.score(S("1.5cm", "measure"), "15mm") == 1
    assert g.score(S("0.02 m", "measure"), "20 mm") == 1
    assert g.score(S("20 mm", "measure"), "1.0 cm") == 0        # 100% 误差
    assert g.last_meta["reason"] == "out_of_tol"


def test_measure_tolerance_tiers(g: DeterministicGrader) -> None:
    # ±10% 以内满分
    assert g.score(S("15 mm", "measure"), "16.4 mm") == 1
    assert g.last_meta["half_credit"] is False
    # 10%~25% 半分档：线上仍返回 0，但 meta 记 half_credit
    assert g.score(S("15 mm", "measure"), "17 mm") == 0
    assert g.last_meta["half_credit"] is True
    assert g.last_meta["reason"] == "within_half_tol"
    # 恰好 25% 边界仍算半分档
    assert g.score(S("20 mm", "measure"), "25 mm") == 0
    assert g.last_meta["half_credit"] is True
    # 超过 25%：0 分，且 half_credit=False（不做负分）
    assert g.score(S("20 mm", "measure"), "26 mm") == 0
    assert g.last_meta["half_credit"] is False
    # 多次测量：只要有一个对上就算对
    assert g.score(S("15 mm and 30 mm", "measure"), "30 mm") == 1


def test_measure_tolerance_configurable() -> None:
    tight = DeterministicGrader({"rl": {"grader": {"measure_full_tol": 0.05}}})
    assert tight.score(S("15 mm", "measure"), "16 mm") == 0
    loose = DeterministicGrader({"rl": {"grader": {"measure_full_tol": 0.20}}})
    assert loose.score(S("15 mm", "measure"), "17 mm") == 1


def test_measure_no_number(g: DeterministicGrader) -> None:
    assert g.score(S("15 mm", "measure"), "cannot tell") == 0
    assert g.last_meta["reason"] == "no_pred_number"
    # 金标无数字时退化到 short 口径，并且 reason 保留"走了退化路径"
    assert g.score(S("not measurable", "measure"), "not measurable") == 1
    assert g.last_meta["reason"] == "no_gold_number_fallback_short"
    assert g.last_meta["fallback_short_reason"] == "equal"


# ------------------------------------------------------------------ parse


def test_parse_extracts_answer_tag(g: DeterministicGrader) -> None:
    assert g.parse("<answer> no effusion </answer>") == ("no effusion", True)
    # 多个标签取最后一个（模型复读时以最终答案为准）
    assert g.parse("<answer>a</answer> ... <answer>b</answer>") == ("b", True)
    # 大小写与换行
    ans, ok = g.parse("<ANSWER>\n 1.5 cm \n</ANSWER>")
    assert ok and ans == "1.5 cm"
    # think 块不应混进答案
    ans, ok = g.parse("<think>hmm</think>\n<answer>yes</answer>")
    assert ok and ans == "yes"


def test_parse_fallback_and_failure(g: DeterministicGrader) -> None:
    assert g.parse("reasoning...\nfinal: pneumonia\n") == ("final: pneumonia", True)
    assert g.parse("<answer></answer>") == ("", False)
    assert g.parse("") == ("", False)
    assert g.parse("   \n  ") == ("", False)
    assert g.parse(None) == ("", False)  # type: ignore[arg-type]


def test_parse_failure_scores_minus_one(g: DeterministicGrader) -> None:
    assert g.score(S("effusion"), "") == -1
    assert g.score(S("effusion"), "<answer></answer>") == -1
    assert g.last_meta["reason"] == "empty_pred"
    assert g.score(S("effusion"), "n/a") == -1


def test_score_uses_parse_on_raw_output(g: DeterministicGrader) -> None:
    raw = "I would look closer.<answer>pleural effusion</answer>"
    assert g.score(S("effusion"), raw) == 1
    assert g.last_meta["parsed"] == "pleural effusion"


# ------------------------------------------------------------------ gradable


def test_gradable(g: DeterministicGrader) -> None:
    assert g.gradable(S("yes", "closed")) is True
    assert g.gradable(S("effusion", "short")) is True
    assert g.gradable(S("15 mm", "measure")) is True
    assert g.gradable(S("long description ...", "long")) is False
    assert g.gradable(S("", "short")) is False
    assert g.gradable(S("x", "weird_type")) is False


def test_long_scores_zero_and_judge_raises(g: DeterministicGrader) -> None:
    sample = S("a long radiology description", "long")
    assert g.score(sample, "a long radiology description") == 0
    with pytest.raises(NotImplementedError):
        g.score_offline_judge(sample, "whatever")


# ------------------------------------------------------------------ 确定性


def test_score_is_idempotent_and_deterministic(g: DeterministicGrader) -> None:
    cases = [
        (S("pleural effusion"), "effusion"),
        (S("no effusion"), "effusion"),
        (S("15 mm", "measure"), "1.6 cm"),
        (S("yes", "closed"), "present"),
        (S("mass"), "nodule"),
    ]
    first = [g.score(s, p) for s, p in cases]
    second = [g.score(s, p) for s, p in cases]
    third = [DeterministicGrader().score(s, p) for s, p in cases]
    assert first == second == third
    # 另一个实例的 meta 也一致
    assert first == [1, 0, 1, 1, 0]


def test_last_meta_is_reset_between_calls(g: DeterministicGrader) -> None:
    g.score(S("15 mm", "measure"), "17 mm")
    assert g.last_meta["half_credit"] is True
    g.score(S("yes", "closed"), "yes")
    assert "half_credit" not in g.last_meta


# ------------------------------------------------------------------ 序列化


def test_to_dict_from_dict_roundtrip(g: DeterministicGrader) -> None:
    d = g.to_dict()
    assert json.loads(json.dumps(d)) == d          # 可 json 落盘
    g2 = DeterministicGrader.from_dict(json.loads(json.dumps(d)))
    assert g2.to_dict() == d
    assert g2 == g
    sample = S("15 mm", "measure")
    assert g2.score(sample, "17 mm") == g.score(sample, "17 mm")


def test_from_dict_preserves_custom_tolerance() -> None:
    g = DeterministicGrader({"rl": {"grader": {"measure_full_tol": 0.3}}})
    g2 = DeterministicGrader.from_dict(json.loads(json.dumps(g.to_dict())))
    assert g2.full_tol == pytest.approx(0.3)
    assert g2.score(S("15 mm", "measure"), "19 mm") == 1


# ------------------------------------------------------------------ 同义词表


def test_external_synonyms_override(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "syn.json"  # type: ignore[operator]
    path.write_text(
        json.dumps({"ground glass": ["GGO", "磨玻璃密度"], "effusion": ["effusion"]}),
        encoding="utf-8",
    )
    g = DeterministicGrader(synonyms=str(path))
    assert g.score(S("ground glass"), "GGO") == 1
    assert g.score(S("ground glass"), "磨玻璃密度") == 1
    # 内置表仍然生效
    assert g.score(S("pleural effusion"), "effusion") == 1


def test_synonyms_dict_argument() -> None:
    g = DeterministicGrader(synonyms={"scar": ["cicatrix"]})
    assert g.score(S("scar"), "cicatrix") == 1
    assert g.gradable(S("scar")) is True


def test_merge_synonyms_union_and_order() -> None:
    merged = merge_synonyms({"a": ["x"]}, {"a": ["y", "x"], "b": ["z"]})
    assert merged["a"] == ["x", "y"]
    assert merged["b"] == ["z"]
    # 不修改输入
    base = {"a": ["x"]}
    merge_synonyms(base, {"a": ["y"]})
    assert base == {"a": ["x"]}


def test_default_synonyms_well_formed() -> None:
    for canon, variants in DEFAULT_SYNONYMS.items():
        assert isinstance(canon, str) and canon
        assert isinstance(variants, list) and variants
        assert all(isinstance(v, str) and v for v in variants)
    # 变体归一化后不应互相冲突到自相矛盾：map 是确定性的
    m1, _ = build_alias_map(DEFAULT_SYNONYMS, normalize)
    m2, _ = build_alias_map(DEFAULT_SYNONYMS, normalize)
    assert m1 == m2
    assert m1[normalize("pleural effusion")] == normalize("pleural effusion")
