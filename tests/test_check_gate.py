"""scripts/check_gate.py 的单测：四处指标偏差的回归测试。

每个用例都构造"旧实现会算错"的输入，断言修复后的口径正确：

  1. tool_oks：``工具执行失败：…`` / ``退化框`` / ``执行超时`` / ``需要参数`` 这类
     失败串不含「错误」「error」，旧实现全判 ok=True；
  2. format_ok：缺失时旧实现无标记地退化成"答案非空"；现在必须标 degraded，
     且带 ``raw`` 时优先走严格 ``<answer>`` 解析；
  3. answer_change_rate：旧实现拿原始字符串比，同义改写/标点/单位差异算 changed；
     现在过 grader.normalize，原始值另存 raw_changed；
  4. tool_call_rate：混合文件里 no_tool 行把分母撑大，调用率被机械拉低。
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import check_gate as C  # noqa: E402


def _obs(body: str) -> Dict[str, Any]:
    """构造一条完整 messages 记录：一次工具调用 + 一条观测。"""
    return {
        "messages": [
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "<tool_call>crop(box=[0,0,10,10])</tool_call>"},
            {"role": "user", "content": C.OBS_HEADER + body + C.OBS_TAIL},
            {"role": "assistant", "content": "<answer>ok</answer>"},
        ]
    }


# ============================================================ 1. tool_oks 失败识别


@pytest.mark.parametrize("body", [
    "工具执行失败：退化框: clip 后 2.0x1.0px，宽高均需 >= 4px",
    "工具执行失败：执行超时（>30s）",
    "工具执行失败：measure 需要参数 box",
    "工具执行失败：未知工具: 'segment'",
    "工具执行失败：ToolUnavailable: 缺少 rescale_slope/rescale_intercept，无法还原 HU",
    "退化框: clip 后 2.0x1.0px",
    "measure 需要参数 box",
    "执行超时（>30s，子进程被终止）",
])
def test_observation_error_markers_detected(body: str) -> None:
    """所有 ops/sandbox 的真实失败串都必须判失败（旧实现漏掉前 8 种）。"""
    assert C.obs_is_error(body) is True
    flat = C._flat(_obs(body))
    assert flat["tool_oks"] == [False]


@pytest.mark.parametrize("body", [
    "12.34 mm",
    "测量结果 15.0 mm，误差 0.1",
    "",
    "<|vision_start|><|image_pad|><|vision_end|>",
])
def test_observation_success_not_flagged(body: str) -> None:
    """成功观测（含图占位/读数）不能被误判为失败。"""
    assert C.obs_is_error(body) is False
    assert C._flat(_obs(body))["tool_oks"] == [True]


def test_legacy_error_keywords_still_count() -> None:
    """旧产物里可能写字面量「错误」/ error，保持向后兼容。"""
    assert C.obs_is_error("输出错误: 越界") is True
    assert C.obs_is_error("Runtime error: divided by zero") is True


def test_failure_marker_only_in_body_not_header() -> None:
    """观测头/尾本身不含失败标记，剥头尾不影响正常判定。"""
    assert C.obs_is_error(C.OBS_HEADER + "5 mm" + C.OBS_TAIL) is False


# ============================================================ 2. format_ok 退化


def test_format_ok_missing_marks_degraded_but_still_passes() -> None:
    """缺 format_ok 不假装配通过：报告标 degraded，闸门仍按推断值判。"""
    recs = [{"id": "1", "answer": "yes", "n_calls": 1, "tool_oks": [True]},
            {"id": "2", "answer": "no", "n_calls": 1, "tool_oks": [True]}]
    rep = C.evaluate_gate(recs, {})
    fv = rep["details"]["format_valid"]
    assert fv["degraded"] is True
    assert fv["n_inferred"] == 2
    assert fv["rate"] == pytest.approx(1.0)
    assert rep["degraded"] is True
    chk = [c for c in rep["checks"] if c["gate"] == "format_valid_rate"][0]
    assert chk["degraded"] is True
    assert "inferred_from_nonempty" in chk["note"]
    # 仍是 PASS：不能因为字段缺失就废掉没有该字段的合法记录
    assert chk["passed"] is True


def test_format_ok_explicit_is_not_degraded() -> None:
    """显式给了 format_ok 就不是推断，degraded=False。"""
    recs = [{"id": "1", "answer": "yes", "format_ok": True, "n_calls": 1,
             "tool_oks": [True]}]
    rep = C.evaluate_gate(recs, {})
    assert rep["details"]["format_valid"]["degraded"] is False
    assert rep["degraded"] is False
    assert rep["details"]["format_ok_source"] == "explicit"
    assert not [c for c in rep["checks"] if c["gate"] == "format_valid_rate"][0].get("note")


def test_raw_strict_parse_preferred_over_nonempty() -> None:
    """有 raw 时用严格 <answer> 解析：有文本但协议不合格 → format_ok=False。

    旧实现按"答案非空"会把这条判合格（rate=1.0），这是偏差所在。
    """
    recs = [{"id": "1", "answer": "yes", "raw": "我直接回答 yes，没有标签",
             "n_calls": 1, "tool_oks": [True]}]
    fv = C.format_valid_report(recs)
    assert fv["n_strict_parsed"] == 1
    assert fv["n_inferred"] == 0
    assert fv["degraded"] is False
    assert fv["rate"] == pytest.approx(0.0)
    # 旧口径（只看 answer 非空）会是 1.0
    assert C.format_valid_rate(recs) < 1.0


def test_raw_with_answer_tag_passes() -> None:
    recs = [{"id": "1", "answer": "yes", "raw": "<think>x</think><answer>yes</answer>"}]
    fv = C.format_valid_report(recs)
    assert fv["n_strict_parsed"] == 1 and fv["rate"] == pytest.approx(1.0)


def test_empty_answer_without_raw_is_inferred_and_fails() -> None:
    recs = [{"id": "1", "answer": ""}]
    fv = C.format_valid_report(recs)
    assert fv["n_inferred"] == 1 and fv["rate"] == pytest.approx(0.0)


def test_format_valid_rate_empty_records_is_nan() -> None:
    assert C.format_valid_rate([]) != C.format_valid_rate([])  # nan != nan


# ============================================================ 3. answer_change_rate


def _pair(a: str, b: str, cond_ids: str = "1") -> List[Dict[str, Any]]:
    return [
        {"id": cond_ids, "condition": "gold_box", "answer": a, "n_calls": 1},
        {"id": cond_ids, "condition": "no_tool", "answer": b, "n_calls": 0},
    ]


def test_answer_change_ignores_case_punct_unit_synonym() -> None:
    """同义改写/标点/单位/大小写差异不算 answer changed（旧实现全算 changed）。

    ``normalize`` 做全角转半角、小写、单位统一 mm、去标点/冠词、单复数归一；
    **不做**同义词改写（那是 grader 内部 canonicalize 的职责，本闸门只要求 normalize）。
    """
    recs = _pair("the 15 MM.", "15 mm") + _pair("no effusions", "No effusion", "2")
    chg = C.answer_change_rate(recs)
    assert chg["n_paired"] == 2
    assert chg["n_changed"] == 0
    assert chg["rate"] == pytest.approx(0.0)
    # 原始比较仍是 changed，保留作对照
    assert chg["raw_changed"] == 2
    assert chg["raw_rate"] == pytest.approx(1.0)


def test_answer_change_detects_real_change() -> None:
    recs = _pair("no effusion", "effusion present")
    chg = C.answer_change_rate(recs)
    assert chg["n_changed"] == 1 and chg["raw_changed"] == 1


def test_answer_change_nan_when_unpaired() -> None:
    recs = [{"id": "1", "condition": "gold_box", "answer": "a", "n_calls": 1}]
    chg = C.answer_change_rate(recs)
    assert chg["n_paired"] == 0
    assert chg["rate"] != chg["rate"]  # nan
    assert chg["raw_rate"] != chg["raw_rate"]


def test_answer_change_gate_uses_normalized_rate() -> None:
    """闸门值必须是归一化口径：同义改写集不该被判为高 change。"""
    recs = _pair("The 15 MM.", "15 mm", "1") + _pair("no effusion", "effusion", "2")
    rep = C.evaluate_gate(recs, {})
    chk = [c for c in rep["checks"] if c["gate"] == "answer_change_rate"][0]
    assert chk["value"] == pytest.approx(0.5)
    assert rep["details"]["answer_change"]["raw_rate"] == pytest.approx(1.0)


# ============================================================ 4. call_rate 分母


def test_tool_call_rate_excludes_no_tool_rows() -> None:
    """混合文件上 no_tool 行不进分母（旧口径 2/4=0.5，正确口径 2/2=1.0）。"""
    recs = [
        {"id": "1", "condition": "gold_box", "answer": "a", "n_calls": 1},
        {"id": "2", "condition": "gold_box", "answer": "b", "n_calls": 2},
        {"id": "1", "condition": "no_tool", "answer": "a", "n_calls": 0},
        {"id": "2", "condition": "no_tool", "answer": "b", "n_calls": 0},
    ]
    cr = C.tool_call_rate(recs)
    assert cr["rate"] == pytest.approx(1.0)
    assert cr["denominator"] == "condition!=no_tool"
    assert cr["n"] == 2
    assert cr["n_no_tool_excluded"] == 2
    # 旧口径（metrics.call_rate）依然给 0.5，用来体现偏差
    assert C.metrics.call_rate([C._flat(r) for r in recs]) == pytest.approx(0.5)


def test_tool_call_rate_backward_compatible_without_condition() -> None:
    """无 condition 字段时按全部行算，保持向后兼容。"""
    recs = [{"id": "1", "answer": "a", "n_calls": 1},
            {"id": "2", "answer": "b", "n_calls": 0}]
    cr = C.tool_call_rate(recs)
    assert cr["denominator"] == "all"
    assert cr["rate"] == pytest.approx(0.5)
    assert cr["n"] == 2


def test_tool_call_rate_view_alias_counts_as_condition() -> None:
    """view 是 condition 的别名，也要被识别（否则 no_tool 行仍进分母）。"""
    recs = [{"id": "1", "view": "gold_box", "answer": "a", "n_calls": 1},
            {"id": "1", "view": "no_tool", "answer": "a", "n_calls": 0}]
    cr = C.tool_call_rate(recs)
    assert cr["rate"] == pytest.approx(1.0)
    assert cr["n"] == 1


def test_tool_call_rate_gate_and_details_expose_denominator() -> None:
    recs = [
        {"id": "1", "condition": "gold_box", "answer": "a", "n_calls": 1,
         "tool_oks": [True], "format_ok": True},
        {"id": "1", "condition": "no_tool", "answer": "a", "n_calls": 0,
         "tool_oks": [], "format_ok": True},
    ]
    rep = C.evaluate_gate(recs, {})
    chk = [c for c in rep["checks"] if c["gate"] == "tool_call_rate"][0]
    assert chk["value"] == pytest.approx(1.0)
    assert chk["denominator"] == "condition!=no_tool"
    assert rep["details"]["tool_call"]["n_no_tool_excluded"] == 1


# ============================================================ over_call_rate 不变坏


def test_over_call_rate_denominator_is_needs_tool_false() -> None:
    recs = [
        {"id": "1", "needs_tool": False, "n_calls": 1, "answer": "a"},
        {"id": "2", "needs_tool": False, "n_calls": 0, "answer": "a"},
        {"id": "3", "needs_tool": True, "n_calls": 1, "answer": "a"},
        {"id": "4", "answer": "a", "n_calls": 1},  # 缺 needs_tool，不进分母
    ]
    oc = C.over_call_rate(recs)
    assert oc["n"] == 2
    assert oc["n_called"] == 1
    assert oc["rate"] == pytest.approx(0.5)
    assert oc["denominator"] == "needs_tool==False"
    assert oc["n_without_needs_tool"] == 1


def test_over_call_rate_nan_without_negatives() -> None:
    oc = C.over_call_rate([{"id": "1", "needs_tool": True, "n_calls": 1}])
    assert oc["n"] == 0 and oc["rate"] != oc["rate"]


# ============================================================ 报告结构向后兼容


def test_report_keys_kept_and_only_extended() -> None:
    """原有键一个不少，新字段只增不改。"""
    recs = [
        {"id": "1", "condition": "gold_box", "answer": "a", "n_calls": 1,
         "tool_oks": [True], "format_ok": True, "needs_tool": False},
        {"id": "1", "condition": "no_tool", "answer": "b", "n_calls": 0,
         "tool_oks": [], "format_ok": True, "needs_tool": False},
    ]
    rep = C.evaluate_gate(recs, {})
    for k in ("n", "checks", "passed", "details"):
        assert k in rep
    for k in ("format_valid_rate", "tool_success_rate", "tool_call_rate",
              "avg_calls", "avg_tokens", "over_call", "answer_change", "per_source"):
        assert k in rep["details"]
    for k in ("n_paired", "n_changed", "rate"):
        assert k in rep["details"]["answer_change"]
    chg = C.answer_change_rate(recs)
    for k in ("raw_changed", "raw_rate"):
        assert k in chg
    assert rep["degraded"] is False
    assert rep["details"]["format_ok_source"] == "explicit"
    for c in rep["checks"]:
        for k in ("gate", "value", "limit", "passed", "note"):
            assert k in c


def test_default_gate_unchanged() -> None:
    """阈值表不许动。"""
    assert C.DEFAULT_GATE == {
        "min_format_valid_rate": 0.99,
        "min_tool_success_rate": 0.95,
        "min_tool_call_rate": 0.55,
        "max_over_call_rate": 0.25,
        "min_answer_change_rate": 0.60,
    }


def test_tool_success_rate_flows_from_messages() -> None:
    """端到端：messages 里失败观测 → tool_success_rate 低于 1（旧实现会读成 1.0）。"""
    recs = [_obs("工具执行失败：执行超时（>30s）"), _obs("12.34 mm")]
    rep = C.evaluate_gate(recs, {})
    assert rep["details"]["tool_success_rate"] == pytest.approx(0.5)
    assert rep["details"]["tool_call_rate"] == pytest.approx(1.0)
