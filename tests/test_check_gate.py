"""scripts/check_gate.py 的单测：四处指标偏差的回归测试。

每个用例都构造"旧实现会算错"的输入，断言修复后的口径正确：

  1. tool_oks：``工具执行失败：…`` / ``退化框`` / ``执行超时`` / ``需要参数`` 这类
     失败串不含「错误」「error」，旧实现全判 ok=True；
  2. format_ok：缺失时旧实现无标记地退化成"答案非空"；现在必须标 degraded，
     且带 ``raw`` 时优先走严格 ``<answer>`` 解析；
  3. answer_change_rate：旧实现拿原始字符串比，同义改写/标点/单位差异算 changed；
     现在过 grader.normalize，原始值另存 raw_changed；
  4. tool_call_rate：混合文件里 no_tool 行把分母撑大，调用率被机械拉低。
  5. 非 dict 行（``null`` / ``[1,2]`` / ``"abc"``）会让旧实现整份闸门崩在
     AttributeError；现在入口过滤并计数；
  6. format_ok 藏在 ``meta`` 里、或是字符串布尔（``"false"``）时的解析口径；
  7. 严格解析取**最后一个非空** ``<answer>``，与 ``eval_gate._strict_answer`` 一致；
  8. 截断轨迹（没有 ``<answer>``）不能把整段文本当答案；
  9. 一轮多 ``<tool_call>`` 的计数口径；
 10. format_ok_source 四态；答 change 的 (dataset, id) 配对与重复行计数；
 11. over_call_rate 两条分支返回同一套键；has_condition 认 view 别名；
 12. 工具成功观测里模型自己 print 的 stdout 不能触发失败判定。
"""
from __future__ import annotations

import os
import re
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


# ============================================================ 5. 非 dict 行


@pytest.mark.parametrize("bad", [None, [1, 2], "abc", 3, True, 1.5])
def test_evaluate_gate_survives_non_dict_lines(bad: Any) -> None:
    """合法 JSON 但不是对象的行不能让闸门崩；跳过并计入 n_bad_lines。

    旧实现：answer_change_rate 里 ``r.get`` → AttributeError，整份报告崩掉。
    """
    good = {"id": "1", "answer": "a", "n_calls": 1, "format_ok": True}
    rep = C.evaluate_gate([bad, good], {})
    assert rep["n"] == 1
    assert rep["n_bad_lines"] == 1
    assert rep["details"]["format_valid"]["n"] == 1


def test_evaluate_gate_all_bad_lines_reports_nan_not_crash() -> None:
    rep = C.evaluate_gate([None, [1, 2], "abc"], {})
    assert rep["n"] == 0 and rep["n_bad_lines"] == 3
    assert rep["passed"] is False  # 没有样本，不许放行
    assert rep["details"]["format_valid_rate"] != rep["details"]["format_valid_rate"]


def test_baseline_with_non_dict_lines_does_not_crash() -> None:
    """baseline 分支旧实现直接 ``r["id"]``，非 dict 行会 TypeError/KeyError。"""
    recs = [{"id": "1", "correct": True, "answer": "a", "n_calls": 1}]
    base = [None, [1, 2], {"id": "1", "correct": False}]
    rep = C.evaluate_gate(recs, {}, baseline=base)
    vb = rep["details"]["vs_baseline"]
    assert vb["n_common"] == 1
    assert vb["n_bad_lines"] == 2


def test_sanitize_counts_and_preserves_order() -> None:
    good, n_bad = C._sanitize([{"id": "b"}, [1], {"id": "a"}, None])
    assert n_bad == 2
    assert [g["id"] for g in good] == ["b", "a"]


# ============================================================ 6. format_ok 来源与布尔解析


def test_format_ok_nested_in_meta_is_honored() -> None:
    """meta 里的显式 format_ok=False 不能被当成缺失而按"答案非空"判合格。

    旧实现读 ``r.get("format_ok")``，对 messages 形态拿到 None → n_inferred=1、
    rate=1.0、degraded=True。
    """
    rec = {"id": "a", "messages": [{"role": "assistant", "content": "<answer>yes</answer>"}],
           "meta": {"format_ok": False}}
    fv = C.format_valid_report([rec])
    assert fv["n_explicit"] == 1
    assert fv["n_inferred"] == 0
    assert fv["degraded"] is False
    assert fv["rate"] == pytest.approx(0.0)


@pytest.mark.parametrize("raw_value,expected", [
    ("false", False), ("False", False), ("0", False), ("no", False), ("FAIL", False),
    ("true", True), ("1", True), ("yes", True),
    (0, False), (1, True), (False, False), (True, True),
])
def test_parse_bool_handles_string_booleans(raw_value: Any, expected: bool) -> None:
    assert C.parse_bool(raw_value) is expected


@pytest.mark.parametrize("raw_value", ["maybe", "", None, float("nan"), [], {}])
def test_parse_bool_unknown_returns_none(raw_value: Any) -> None:
    assert C.parse_bool(raw_value) is None


def test_string_false_format_ok_is_not_counted_as_pass() -> None:
    """旧实现 ``bool("false") is True`` → rate=1.0、degraded=False。"""
    fv = C.format_valid_report([{"id": "a", "answer": "x", "format_ok": "false"}])
    assert fv["n_explicit"] == 1
    assert fv["rate"] == pytest.approx(0.0)
    assert C.format_valid_report([{"id": "a", "answer": "x",
                                   "format_ok": "true"}])["rate"] == pytest.approx(1.0)


# ============================================================ 7. 严格解析口径对齐 eval_gate


def test_strict_parse_takes_last_nonempty_answer_block() -> None:
    """``<answer></answer><answer>5mm</answer>`` 应与 eval_gate 一样判合格（不依赖
    eval_gate 的 import：那边取 reversed 后的第一个非空块，等价于"最后一个非空"）。"""
    raw = "<answer></answer><answer>5mm</answer>"
    assert C.last_nonempty_answer(raw) == "5mm"
    assert C._strict_answer_ok({"raw": raw}) is True
    assert C.format_valid_report([{"id": "a", "raw": raw}])["rate"] == pytest.approx(1.0)


@pytest.mark.parametrize("raw,expected", [
    ("<answer>5mm</answer><answer>6mm</answer>", "6mm"),
    ("<think>x</think>\n<answer>\n 5 mm \n</answer>\n<answer></answer>", "5 mm"),
    ("<answer>a</answer>trailing", "a"),
    ("<answer>a</answer><answer>\n \t \n</answer>", "a"),
])
def test_last_nonempty_answer_picks_last_block(raw: str, expected: str) -> None:
    """与 ``eval_gate._strict_answer`` 同口径：取最后一个非空块。"""
    assert C.last_nonempty_answer(raw) == expected


def test_strict_parse_all_blocks_empty_fails() -> None:
    raw = "<answer></answer><answer>   </answer>"
    assert C._strict_answer_ok({"raw": raw}) is False
    assert C.last_nonempty_answer(raw) is None


# ============================================================ 8. 截断轨迹不伪造答案


def test_truncated_trajectory_without_answer_is_not_valid() -> None:
    """最后一轮是 tool_call/半句话时没有答案：旧实现把整段文本当 answer，
    格式闸门按"答案非空"判合格（rate=1.0）。

    现在 answer 保持空 → n_inferred 仍走非空判据，但值为 False。
    """
    rec = {"id": "a", "messages": [
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "<tool_call>crop(box=[0,0,1,1])</tool_call>"},
        {"role": "user", "content": C.OBS_HEADER + "ok" + C.OBS_TAIL},
        {"role": "assistant", "content": "<think>我还需要再看一"},
    ]}
    flat = C._flat(rec)
    assert flat["answer"] == ""
    fv = C.format_valid_report([rec])
    assert fv["rate"] == pytest.approx(0.0)
    assert fv["degraded"] is True


def test_answer_from_last_assistant_message_with_tag() -> None:
    rec = {"messages": [
        {"role": "assistant", "content": "<answer>first</answer>"},
        {"role": "assistant", "content": "中间话"},
        {"role": "assistant", "content": "<think>y</think><answer>final</answer>"},
    ]}
    assert C._flat(rec)["answer"] == "final"  # 中间无标签的轮不覆盖


def test_content_blocks_form_is_supported() -> None:
    rec = {"messages": [{"role": "assistant",
                         "content": [{"type": "text", "text": "<answer>blk</answer>"}]}]}
    assert C._flat(rec)["answer"] == "blk"


# ============================================================ 9. n_calls 与 tool_oks 对齐


def test_two_tool_calls_in_one_turn_count_as_two() -> None:
    """一轮并发两个 tool_call：n_calls 按出现次数计（旧实现按轮数 → 1）。"""
    rec = {"id": "a", "messages": [
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content":
            "<tool_call>crop(box=[0,0,1,1])</tool_call><tool_call>crop(box=[1,1,2,2])</tool_call>"},
        {"role": "user", "content": C.OBS_HEADER + "a" + C.OBS_TAIL},
        {"role": "user", "content": C.OBS_HEADER + "b" + C.OBS_TAIL},
        {"role": "assistant", "content": "<answer>x</answer>"},
    ]}
    flat = C._flat(rec)
    assert flat["n_calls"] == 2
    assert flat["tool_oks"] == [True, True]
    assert len(flat["tool_oks"]) == flat["n_calls"]
    assert "diag_calls_vs_obs" not in flat


def test_call_obs_mismatch_is_diagnosed() -> None:
    """调了两次只回一条观测：不算错，但要在字段里暴露，别静默。"""
    rec = {"messages": [
        {"role": "assistant", "content": "<tool_call>a</tool_call><tool_call>b</tool_call>"},
        {"role": "user", "content": C.OBS_HEADER + "x" + C.OBS_TAIL},
    ]}
    flat = C._flat(rec)
    assert flat["n_calls"] == 2
    assert flat["diag_calls_vs_obs"] == {"n_calls": 2, "n_obs": 1}


def test_avg_calls_reflects_tool_call_occurrences() -> None:
    rec = {"id": "a", "messages": [
        {"role": "assistant", "content": "<tool_call>a</tool_call><tool_call>b</tool_call>"},
        {"role": "user", "content": C.OBS_HEADER + "x" + C.OBS_TAIL},
        {"role": "user", "content": C.OBS_HEADER + "y" + C.OBS_TAIL},
        {"role": "assistant", "content": "<answer>z</answer>"},
    ]}
    rep = C.evaluate_gate([rec], {})
    assert rep["details"]["avg_calls"] == pytest.approx(2.0)


# ============================================================ 10. format_ok_source 四态


def test_format_ok_source_strict_only() -> None:
    """纯 strict 解析旧实现被错标成 explicit。"""
    rep = C.evaluate_gate([{"id": "a", "raw": "<answer>5mm</answer>"}], {})
    assert rep["details"]["format_valid"]["n_explicit"] == 0
    assert rep["details"]["format_valid"]["n_strict_parsed"] == 1
    assert rep["details"]["format_ok_source"] == "strict"


def test_format_ok_source_all_four_states() -> None:
    def src(recs: List[Dict[str, Any]]) -> str:
        return C.evaluate_gate(recs, {})["details"]["format_ok_source"]

    assert src([{"id": "1", "format_ok": True}]) == "explicit"
    assert src([{"id": "1", "raw": "<answer>x</answer>"}]) == "strict"
    assert src([{"id": "1", "answer": "x"}]) == "inferred_from_nonempty"
    assert src([{"id": "1", "format_ok": True},
                {"id": "2", "raw": "<answer>x</answer>"}]) == "mixed"
    assert src([{"id": "1", "format_ok": True},
                {"id": "2", "answer": "x"}]) == "mixed"
    assert src([]) == "none"


# ============================================================ 11. answer_change_rate 配对与重复


def test_answer_change_pairs_by_dataset_and_id() -> None:
    """不同数据集同 id 不能配成一对（旧实现 n_paired=1、rate=1.0）。"""
    recs = [
        {"id": 1, "dataset": "A", "condition": "gold_box", "answer": "x"},
        {"id": 1, "dataset": "B", "condition": "no_tool", "answer": "y"},
    ]
    chg = C.answer_change_rate(recs)
    assert chg["n_paired"] == 0
    assert chg["rate"] != chg["rate"]  # nan
    # 同 dataset 同 id 仍能正常配对
    recs.append({"id": 1, "dataset": "A", "condition": "no_tool", "answer": "x"})
    chg = C.answer_change_rate(recs)
    assert chg["n_paired"] == 1 and chg["n_changed"] == 0


def test_answer_change_counts_duplicate_rows() -> None:
    """同 (dataset, id, condition) 重复行是 last-write-wins，必须计数。"""
    recs = [
        {"id": "1", "dataset": "A", "condition": "gold_box", "answer": "x"},
        {"id": "1", "dataset": "A", "condition": "gold_box", "answer": "z"},
        {"id": "1", "dataset": "A", "condition": "no_tool", "answer": "x"},
    ]
    chg = C.answer_change_rate(recs)
    assert chg["n_paired"] == 1
    assert chg["n_duplicate"] == 1
    assert chg["n_changed"] == 1  # 后者覆盖前者（z vs x）


def test_answer_change_ignores_non_dict_rows() -> None:
    chg = C.answer_change_rate([None, [1, 2]])
    assert chg["n_bad_lines"] == 2 and chg["n_paired"] == 0


# ============================================================ 12. over_call_rate / has_condition


def test_over_call_rate_key_schema_is_stable() -> None:
    """两条分支必须同一套键（旧实现早退分支缺 n_without_needs_tool）。"""
    keys = {"n", "n_called", "rate", "denominator", "n_without_needs_tool"}
    empty = C.over_call_rate([{"id": "1", "needs_tool": True, "n_calls": 1}])
    normal = C.over_call_rate([{"id": "1", "needs_tool": False, "n_calls": 1}])
    none_field = C.over_call_rate([{"id": "1", "n_calls": 1}])
    assert set(empty) == keys
    assert set(normal) == keys
    assert set(none_field) == keys
    assert none_field["n_without_needs_tool"] == 1


def test_answer_change_note_recognizes_view_alias() -> None:
    """只有 view 字段（无 condition）时，不该拿到"无 condition 配对"的误导性 note。

    旧实现 has_condition 只认 condition → 明明没有可配对行，note 却指向字段缺失。
    """
    recs = [{"id": "1", "view": "full", "answer": "a", "n_calls": 1}]
    rep = C.evaluate_gate(recs, {})
    chk = [c for c in rep["checks"] if c["gate"] == "answer_change_rate"][0]
    assert chk["passed"] is False
    # 有 view 字段（= 有配对口径），只是这题没配对上，note 是通用"指标缺失"，
    # 而不是"输入无 condition 配对"这种字段缺失的误导说明。
    assert chk["note"] == "指标缺失，按不通过处理"


# ============================================================ 13. stdout 不触发失败判定


def _obs_text(body: str) -> str:
    return C.OBS_HEADER + body + C.OBS_TAIL


def test_stdout_error_word_does_not_flag_failure() -> None:
    """模型自己 print 的 stdout 里有 "no error found" 不能记成失败调用。

    旧实现：obs_is_error(该观测) == True → tool_success_rate 被压低。
    """
    text = _obs_text("\n<stdout>\nno error found\n</stdout>\n")
    assert C.obs_is_error(text) is False
    assert C._flat(_obs("\n<stdout>\nno error found\n</stdout>\n"))["tool_oks"] == [True]


def test_stdout_stripped_but_structural_failure_still_detected() -> None:
    """剥 stdout 之后的结构性失败标记仍要抓到（sandbox 成功时没有这个前缀）。"""
    text = _obs_text("\n<stdout>\n12.3\n</stdout>\n工具执行失败：执行超时（>30s）")
    assert C.obs_is_error(text) is True


@pytest.mark.parametrize("body", [
    "<stdout>Error: 内部变量名 error_flag\n</stdout>",
    "<STDOUT>\r\nToolUnavailable 是下游概念\r\n</STDOUT>",  # 大小写/CRLF 的肉眼看不出差异，但标签必须可剥
])
def test_stdout_tag_matching_is_sloppy_tolerant(body: str) -> None:
    assert C.obs_is_error(_obs_text(body)) is False


def test_legacy_markers_are_last_resort_only() -> None:
    """旧关键词仍是兜底（不剥 stdout 时照旧生效），结构性标记优先且不混在一起。"""
    assert C.obs_is_error("我遇到了 error，无法继续") is True
    assert "错误" not in C.STRUCTURAL_FAILURE_MARKERS
    assert set(C.STRUCTURAL_FAILURE_MARKERS).issubset(C.TOOL_FAILURE_MARKERS)


def test_stdout_regex_strips_multiline_body() -> None:
    assert C._STDOUT_BLOCK_RE.search("<stdout>\na\nb\nerror\n</stdout>") is not None
    body = C._STDOUT_BLOCK_RE.sub(" ", "<stdout>\na\nerror\n</stdout>")
    assert "error" not in body and re.search(r"error", body) is None
