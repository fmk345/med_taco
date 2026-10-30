"""scripts/eval_gate.py 的单测：完全离线，只造小样本 + 假模型。

覆盖点（对应任务书要求）：
  - gold_box 答对 / no_tool 答错时，answer_change_rate=1.0，correct 逐条正确；
  - format_ok **严格**：没有 ``<answer>`` 标签时判 False，不能被"答案非空"蒙混；
  - 两条件的 messages 结构真的不同（图片数、消息数）；
  - build_gate_items 跳过 long 题、无金标题、无 tool 的题不画裁剪图；
  - gold_box 的裁剪失败如实记 tool_oks=[False]，而不是假装没调用。
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import check_gate as C  # noqa: E402
from scripts import eval_gate as G  # noqa: E402
from medtaco.rl.grader import DeterministicGrader  # noqa: E402
from medtaco.utils.config import load_config  # noqa: E402
from medtaco.utils.schema import Box, Sample  # noqa: E402


# ---------------------------------------------------------------- 造数据


def _make_png(path: str, size: int = 96) -> str:
    """造一张 96x96 假图：左上一块亮斑，保证裁剪出的是真子图。"""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[20:50, 20:50] = 220
    Image.fromarray(arr).save(path)
    return path


def _write_questions(tmp_path, specs: Sequence[Dict[str, Any]]) -> str:
    path = str(tmp_path / "questions.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for s in specs:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    return path


def _spec(sid: str, image: str, question: str, answer: str,
          answer_type: str = "short", need_crop: bool = True,
          with_box: bool = True) -> Dict[str, Any]:
    boxes = [[20.0, 20.0, 50.0, 50.0, "nodule"]] if with_box else []
    return {
        "id": sid, "image": image, "question": question, "answer": answer,
        "dataset": "Mock", "tier": "A", "answer_type": answer_type,
        "boxes": boxes, "masks": [], "license": "synthetic-mock",
        "commercial_ok": True,
        "meta": {"source_sample_id": "s:" + sid, "need_crop": need_crop,
                 "generator": "G1" if need_crop else "G3"},
    }


@pytest.fixture()
def cfg(tmp_path):
    """复用项目配置，但把 processed 指到 tmp，避免往仓库里写裁剪图。"""
    c = load_config("configs/default.yaml")
    c.set("paths.processed", str(tmp_path / "processed"))
    c.set("eval.gate.max_tokens", 128)
    return c


@pytest.fixture()
def three_items(tmp_path, cfg):
    """3 条 closed/short 小样本（2 正样本 + 1 负样本），都带金标。"""
    specs = []
    for i, (ans, atype, need) in enumerate(
            [("nodule", "short", True),
             ("yes", "closed", True),
             ("no", "closed", False)]):
        img = _make_png(str(tmp_path / f"img{i}.png"))
        specs.append(_spec(f"q{i}", img, f"question {i}?", ans,
                           answer_type=atype, need_crop=need,
                           with_box=need))
    qpath = _write_questions(tmp_path, specs)
    items = G.build_gate_items(qpath, None, cfg)
    return items


# ---------------------------------------------------------------- build_gate_items


def test_build_gate_items_filters_long_and_goldless(tmp_path, cfg):
    """long 题、空金标题、未标注 answer_type 的题都不能进闸门。"""
    img = _make_png(str(tmp_path / "a.png"))
    specs = [
        _spec("ok", img, "q?", "nodule"),
        _spec("long", img, "q?", "some long free text", answer_type="long"),
        _spec("empty", img, "q?", ""),
        _spec("weird", img, "q?", "nodule", answer_type="unknown"),
    ]
    qpath = _write_questions(tmp_path, specs)
    items = G.build_gate_items(qpath, None, cfg)
    assert [it["id"] for it in items] == ["ok"]
    it = items[0]
    assert it["sample_id"] == "s:ok"
    assert it["gold"] == "nodule"
    assert it["needs_tool"] is True
    assert it["dataset"] == "Mock"
    assert it["gold_crop"] and os.path.exists(it["gold_crop"])


def test_build_gate_items_skips_crop_for_negative(tmp_path, cfg):
    """need_crop=False 的负样本：不画裁剪图，needs_tool=False。"""
    img = _make_png(str(tmp_path / "n.png"))
    qpath = _write_questions(tmp_path, [_spec("neg", img, "q?", "no",
                                              answer_type="closed",
                                              need_crop=False, with_box=False)])
    items = G.build_gate_items(qpath, None, cfg)
    assert len(items) == 1
    assert items[0]["needs_tool"] is False
    assert items[0]["gold_crop"] is None


def test_build_gate_items_limit_and_traj_filter(tmp_path, cfg):
    """limit 先过滤后截断；trajs 给了就只保留覆盖到的题。"""
    specs = []
    for i in range(4):
        img = _make_png(str(tmp_path / f"l{i}.png"))
        specs.append(_spec(f"l{i}", img, "q?", "nodule"))
    qpath = _write_questions(tmp_path, specs)

    assert len(G.build_gate_items(qpath, None, cfg)) == 4
    assert len(G.build_gate_items(qpath, None, cfg, limit=2)) == 2

    tpath = str(tmp_path / "traj.jsonl")
    with open(tpath, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "traj-l1", "traj": {"id": "traj-l1",
                                                      "sample_id": "l1"}}) + "\n")
    got = G.build_gate_items(qpath, tpath, cfg)
    assert [it["id"] for it in got] == ["l1"]

    # 不存在的 trajs 文件 = 不过滤（而不是返回空集）
    assert len(G.build_gate_items(qpath, str(tmp_path / "nope.jsonl"), cfg)) == 4


def test_crop_boxes_for_context_uses_pad(tmp_path, cfg):
    """金标框走 crop_boxes_for 的 context 角色，外扩 0.25。"""
    img = _make_png(str(tmp_path / "p.png"))
    qpath = _write_questions(tmp_path, [_spec("pad", img, "q?", "nodule")])
    items = G.build_gate_items(qpath, None, cfg)
    box, pad = items[0]["_gold_plan"]
    assert pad == pytest.approx(G.DEFAULT_PAD)
    assert box.as_tuple() == (20.0, 20.0, 50.0, 50.0)


def test_g2_gold_plan_picks_inner_not_outer(tmp_path, cfg):
    """G2 的 plan 是 [outer(最外层大框), inner(金标病灶)]，必须挑 inner。

    取 plan[0] 会拿到 outer：真实数据里 outer/gold 面积比中位 4x、最大 16x，
    极端样本的 outer 外扩后就是整图，gold_box 条件退化成 full_view。
    现成裁图路径（_crop_from_meta）挑的是 inner，两条路径必须同框。
    """
    img = _make_png(str(tmp_path / "g2.png"), size=128)
    spec = _spec("g2", img, "q?", "nodule")
    # 最外层近乎整图，金标是里面 40-80 的小框
    spec["boxes"] = [[40.0, 40.0, 80.0, 80.0, "nodule"]]
    spec["meta"]["generator"] = "G2"
    spec["meta"]["scale_boxes"] = [[5.0, 5.0, 95.0, 95.0, "lung"],
                                   [40.0, 40.0, 80.0, 80.0, "nodule"]]
    # 现成裁图：outer 与 inner 两张都真实存在，_crop_from_meta 必须挑 inner
    outer_png = _make_png(str(tmp_path / "g2_outer.png"), size=128)
    inner_png = _make_png(str(tmp_path / "g2_inner.png"), size=64)
    spec["meta"]["crops"] = [
        {"path": outer_png, "box": [5.0, 5.0, 95.0, 95.0, "lung"], "role": "outer"},
        {"path": inner_png, "box": [40.0, 40.0, 80.0, 80.0, "nodule"], "role": "inner"},
    ]
    qpath = _write_questions(tmp_path, [spec])
    items = G.build_gate_items(qpath, None, cfg)
    assert len(items) == 1
    box, pad = items[0]["_gold_plan"]
    assert box.as_tuple() == (40.0, 40.0, 80.0, 80.0)
    assert pad == 0.0  # inner 不外扩

    # 两条取图路径同框：现成裁图也要挑 inner
    assert items[0]["gold_crop"] == inner_png

    # 渲染出来的 crop 调用坐标必须是 inner，而不是 outer
    r = G.render_condition(items[0], "gold_box", cfg)
    call_text = "".join(str(b.get("text", "")) for b in r["messages"][-2]["content"])
    assert "[40.0, 40.0, 80.0, 80.0]" in call_text
    assert "[5.0, 5.0, 95.0, 95.0]" not in call_text


def test_gold_role_order_is_shared(tmp_path, cfg):
    """_gold_plan 与 _crop_from_meta 必须共用同一个 role 优先级常量。"""
    assert G.GOLD_ROLE_ORDER.index("inner") < G.GOLD_ROLE_ORDER.index("outer")
    # 渲染路径与现场重裁用的是同一份优先级：改常量即同时生效
    assert G.GOLD_ROLE_ORDER[0] in ("context", "tight")


def test_build_gate_items_drops_need_crop_without_gold(tmp_path, cfg, capsys):
    """need_crop=True 却没有金标区域的题在造题阶段就剔掉，并报告条数。

    这种题是数据缺陷：模型在 gold_box 条件下一次都调不到工具，留着只会写出一条
    n_calls=1 / format_ok=False 的记录，把数据问题记成模型问题。
    """
    img = _make_png(str(tmp_path / "nb.png"))
    bad = _spec("nogold", img, "q?", "nodule")
    bad["boxes"] = []                       # 没有框
    bad["meta"]["need_crop"] = True         # 却标了需要裁剪
    bad["meta"].pop("scale_boxes", None)
    good = _spec("good", img, "q?", "nodule")
    qpath = _write_questions(tmp_path, [bad, good])

    items = G.build_gate_items(qpath, None, cfg)
    assert [it["id"] for it in items] == ["good"]
    err = capsys.readouterr().err
    assert "剔除 need_crop=True 但无金标区域的题 1 条" in err


def test_run_condition_no_gold_box_does_not_fake_a_call(tmp_path, cfg):
    """条件不成立（no_gold_box）时不能写 n_calls=1 / tool_oks=[False]。"""
    item = {"id": "x", "sample": Sample(id="x", image="", question="q", answer="a"),
            "needs_tool": True, "gold_crop": None, "_gold_plan": None, "dataset": "Mock"}
    called = []

    def fn(m, i, mt, t):
        called.append(1)
        return "<answer>a</answer>"

    rec = G.run_condition(fn, item, "gold_box", cfg)
    assert called == []                      # 模型一次都没被调用
    assert rec["n_calls"] == 0               # 不能记成"调用了一次且失败"
    assert rec["tool_oks"] == []
    assert rec["invalid_item"] is True
    assert rec["correct"] is None            # 缺测，不是答错
    # 闸门侧不能把它算成一次工具失败（没有调用 → nan，而不是 0.0）
    eff = C.metrics.tool_success_rate([rec])
    assert eff != 0.0

    # 手工构造的 items 也不能让这条记录流进产物
    grader = DeterministicGrader(cfg)
    records = G.run_gate([item], cfg, fn, conditions=("gold_box",), grader=grader)
    assert records == []


def test_no_tool_self_initiated_call_counts_as_over_call(tmp_path, cfg):
    """no_tool 条件下模型自发发 <tool_call> → over_call_rate > 0。

    原实现里 no_tool 的 n_calls 硬编码 0，这道闸门结构上永远 PASS。
    """
    img = _make_png(str(tmp_path / "oc.png"))
    qpath = _write_questions(tmp_path, [
        _spec("pos", img, "q?", "nodule"),
        _spec("neg", img, "q?", "no", answer_type="closed", need_crop=False,
              with_box=False)])
    items = G.build_gate_items(qpath, None, cfg)
    assert [it["needs_tool"] for it in items] == [True, False]  # 一正一负

    def fn(m, i, mt, t):
        # 自己在没有工具的条件下"调"了一次
        return "<tool_call>crop(box=[1,1,5,5]):</tool_call><answer>no</answer>"

    recs = G.run_gate(items, cfg, fn, conditions=("no_tool",), workers=1)
    assert len(recs) == 2
    neg = [r for r in recs if not r["needs_tool"]]
    assert neg and all(r["n_calls"] == 1 for r in neg)
    assert all(r["tool_oks"] == [] for r in neg)  # 没有真实执行，不伪造成功

    oc = C.over_call_rate(recs)
    assert oc["n"] == 1 and oc["n_called"] == 1
    assert oc["rate"] == pytest.approx(1.0)


def test_limit_non_positive_is_rejected(tmp_path, capsys):
    """--limit 0 / 负数曾经被静默当成"不限量"，必须直接报错。"""
    img = _make_png(str(tmp_path / "lm.png"))
    qpath = _write_questions(tmp_path, [_spec("a", img, "q?", "nodule")])
    cfgp = _write_cfg(tmp_path)
    for bad in ("0", "-1"):
        code = G.main(["--config", cfgp, "--questions", qpath,
                       "--trajs", str(tmp_path / "none.jsonl"),
                       "--limit", bad, "--dry-run"])
        assert code == 2
        assert "--limit" in capsys.readouterr().err


def test_match_traj_ids_uses_sample_id_only(tmp_path):
    """轨迹覆盖只认 sample_id，不拿 traj id / 剥 traj- 前缀的短串放行。"""
    tpath = str(tmp_path / "traj.jsonl")
    with open(tpath, "w", encoding="utf-8") as f:
        # sample id 本身以 traj- 开头：旧实现会把剥掉前缀的 "x" 也放进集合
        f.write(json.dumps({"id": "traj-traj-x", "sample_id": "traj-x",
                            "traj": {"id": "traj-traj-x", "sample_id": "traj-x"}}) + "\n")
    ids = G._match_traj_ids(tpath)
    assert ids == {"traj-x"}
    assert "x" not in ids
    assert "traj-traj-x" not in ids


# ---------------------------------------------------------------- 渲染结构


def test_conditions_have_different_structure(three_items, cfg):
    """gold_box 比 no_tool 多一次调用、多一条观测、多一张图。"""
    item = next(it for it in three_items if it["needs_tool"])
    a = G.render_condition(item, "gold_box", cfg)
    b = G.render_condition(item, "no_tool", cfg)

    assert len(a["images"]) == 2
    assert len(b["images"]) == 1
    assert len(a["messages"]) >= 3
    assert len(b["messages"]) == 1
    assert a["images"][0] == b["images"][0] == item["sample"].image
    assert a["images"][1] == item["gold_crop"]
    assert a["need_continue"] is True and b["need_continue"] is False

    # 观测消息必须是独立的 user 轮，且带首尾标记（training_set 的约定）
    obs = a["messages"][-1]
    assert obs["role"] == "user"
    texts = "".join(b.get("text", "") for b in obs["content"] if b.get("type") == "text")
    assert texts.startswith(G.OBS_HEADER) and texts.endswith(G.OBS_TAIL)

    # 工具调用渲染成 <tool_call>crop(...)</tool_call>
    call_texts = "".join(
        "".join(str(x.get("text", "")) for x in m["content"])
        for m in a["messages"] if m["role"] == "assistant")
    assert "<tool_call>crop(" in call_texts
    assert b["messages"][0]["content"][-1]["text"] == a["messages"][0]["content"][-1]["text"]


def test_render_unknown_condition_raises(three_items, cfg):
    with pytest.raises(ValueError):
        G.render_condition(three_items[0], "full_view", cfg)


def test_gold_box_without_box_is_not_renderable(tmp_path, cfg):
    """没有金标区域的题，gold_box 条件不成立（need_continue=False）。"""
    item = {"id": "x", "sample": Sample(id="x", image="", question="q", answer="a"),
            "needs_tool": True, "gold_crop": None, "_gold_plan": None, "dataset": "Mock"}
    r = G.render_condition(item, "gold_box", cfg)
    assert r["need_continue"] is False
    assert r["error"] == "no_gold_box"


# ---------------------------------------------------------------- 端到端闸门


def _fake_model(gold_box_text: str, no_tool_text: str):
    """假模型：gold_box 条件按顺序给对答案，no_tool 恒给错；记录收到的入参。"""
    calls: List[Dict[str, Any]] = []

    def fn(messages: List[Dict[str, Any]], images: List[str],
           max_tokens: int, temperature: float) -> str:
        calls.append({"n_msg": len(messages), "n_img": len(images)})
        # 带观测的条件 = 入参里出现工具返回标记
        has_obs = "【工具返回】" in json.dumps(messages, ensure_ascii=False)
        return gold_box_text if has_obs else no_tool_text

    fn.calls = calls  # type: ignore[attr-defined]
    return fn


def test_run_gate_end_to_end(three_items, cfg):
    """gold_box 答对、no_tool 答错 → answer_change_rate=1.0，correct 逐条正确。"""
    grader = DeterministicGrader(cfg)
    # gold_box：按题给金标；no_tool：统一给一个错答案
    golds = {it["id"]: it["gold"] for it in three_items}

    def fn(messages, images, max_tokens, temperature):
        blob = json.dumps(messages, ensure_ascii=False)
        if "【工具返回】" in blob:
            # 从原题消息里取回题面，找到对应金标
            q = json.loads(blob)[0]["content"][-1]["text"]
            sid = q.split("question ")[1].split("?")[0]
            return f"思考中。<answer>{golds['q' + sid]}</answer>"
        return "我猜不出。<answer>definitely_wrong</answer>"

    records = G.run_gate(three_items, cfg, fn, workers=2, grader=grader)

    # 2 条正样本跑两条件，1 条负样本只跑 no_tool
    assert len(records) == 2 * 2 + 1
    for r in records:
        assert r["format_ok"] is True
        if r["condition"] == "gold_box":
            assert r["n_calls"] == 1 and r["tool_oks"] == [True]
            assert r["correct"] is True
        else:
            assert r["n_calls"] == 0 and r["tool_oks"] == []
            assert r["correct"] is False

    chg = C.answer_change_rate(records)
    assert chg["n_paired"] == 2
    assert chg["rate"] == pytest.approx(1.0)


def test_format_ok_is_strict(three_items, cfg):
    """raw 里没有 <answer> 一律 format_ok=False；空标签也不行。"""
    fn = lambda m, i, mt, t: "答案就是 nodule（但没有标签）"  # noqa: E731
    item = three_items[0]
    rec = G.run_condition(fn, item, "no_tool", cfg)
    assert rec["format_ok"] is False
    assert rec["answer"] == ""

    fn2 = lambda m, i, mt, t: "<answer>   </answer>"  # noqa: E731
    rec2 = G.run_condition(fn2, item, "no_tool", cfg)
    assert rec2["format_ok"] is False

    # 对照：带标签才算合法（防止上面两条是因为别的原因失败）
    fn3 = lambda m, i, mt, t: "<answer>nodule</answer>"  # noqa: E731
    rec3 = G.run_condition(fn3, item, "no_tool", cfg)
    assert rec3["format_ok"] is True
    assert rec3["answer"] == "nodule"


def test_format_ok_ignores_grader_fallback(three_items, cfg):
    """grader.parse 会把裸文本当答案，但 format_ok 不能跟着它放水。"""
    grader = DeterministicGrader(cfg)
    item = three_items[0]
    rec = G.run_condition(lambda m, i, mt, t: "first line\nnodule", item,
                          "no_tool", cfg)
    # grader 自己解析得出 nodule（退化路径），但闸门的格式判定必须是 False
    assert grader.parse("first line\nnodule")[0] == "nodule"
    assert rec["format_ok"] is False
    graded = G.grade_record(dict(rec), item, grader)
    assert graded["correct"] is False


def test_tool_failure_recorded_as_false(tmp_path, cfg):
    """裁剪失败时 gold_box 仍记 n_calls=1 且 tool_oks=[False]（不伪装成没调用）。"""
    img = _make_png(str(tmp_path / "f.png"))
    qpath = _write_questions(tmp_path, [_spec("f", img, "q?", "nodule")])
    items = G.build_gate_items(qpath, None, cfg)
    item = items[0]
    # 把现成裁剪图删掉，再把原图挪走，令现场重裁失败
    if item.get("gold_crop") and os.path.exists(item["gold_crop"]):
        os.remove(item["gold_crop"])
    item["gold_crop"] = None
    os.remove(img)

    rendered = G.render_condition(item, "gold_box", cfg)
    assert rendered["crop_ok"] is False
    rec = G.run_condition(lambda m, i, mt, t: "<answer>nodule</answer>", item,
                          "gold_box", cfg)
    assert rec["n_calls"] == 1
    assert rec["tool_oks"] == [False]

    # 闸门侧：这次调用按失败计数
    eff = C.metrics.tool_success_rate([rec])
    assert eff == 0.0


def test_record_is_check_gate_compatible(three_items, cfg):
    """记录喂给 check_gate 时必须落在它认的字段上（不是 nan）。"""
    fn = _fake_model("<answer>nodule</answer>", "<answer>wrong</answer>")
    records = G.run_gate(three_items, cfg, fn, workers=1)
    flat = [C._flat(r) for r in records]
    assert all(f.get("condition") in ("gold_box", "no_tool") for f in flat)
    assert all(f.get("needs_tool") is not None for f in flat)
    assert C.format_valid_rate(records) == pytest.approx(1.0)
    oc = C.over_call_rate(records)
    # 负样本只跑了 no_tool（n_calls=0），所以 over-call 为 0
    assert oc["n"] == 1 and oc["n_called"] == 0


def test_run_gate_skips_gold_box_for_negative(three_items, cfg):
    """needs_tool=False 的题不跑 gold_box：否则注入的调用会把自己算成 over-call。"""
    fn = lambda m, i, mt, t: "<answer>no</answer>"  # noqa: E731
    records = G.run_gate(three_items, cfg, fn, conditions=("gold_box", "no_tool"))
    gold_ids = {r["id"] for r in records if r["condition"] == "gold_box"}
    neg_ids = {it["id"] for it in three_items if not it["needs_tool"]}
    assert gold_ids.isdisjoint(neg_ids)


def _write_cfg(tmp_path) -> str:
    """写一份最小配置，把 processed 指到 tmp：CLI 用例不能往仓库 data/ 里写裁剪图。"""
    path = str(tmp_path / "cfg.yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"paths:\n  processed: {tmp_path / 'processed'}\n"
                f"  logs: {tmp_path / 'logs'}\n"
                f"  questions: {tmp_path}\n"
                f"  trajectories: {tmp_path}\n")
    return path


def test_dry_run_cli(tmp_path, capsys):
    """--dry-run 全离线、不要求 endpoint，退出码 0，且两个条件都报告 1/1 可渲染。"""
    img = _make_png(str(tmp_path / "d.png"))
    qpath = _write_questions(tmp_path, [_spec("d", img, "q?", "nodule")])
    code = G.main(["--config", _write_cfg(tmp_path), "--questions", qpath,
                   "--trajs", str(tmp_path / "none.jsonl"), "--dry-run"])
    assert code == 0
    out = capsys.readouterr().out
    # 回归防线：no_tool 是单段对话，曾经误用 need_continue 统计导致报 0 条。
    assert "dry-run gold_box: 可渲染 1/1 条" in out
    assert "dry-run no_tool: 可渲染 1/1 条" in out


def test_main_requires_endpoint_without_dry_run(tmp_path):
    img = _make_png(str(tmp_path / "e.png"))
    qpath = _write_questions(tmp_path, [_spec("e", img, "q?", "nodule")])
    code = G.main(["--config", _write_cfg(tmp_path), "--questions", qpath,
                   "--trajs", str(tmp_path / "none.jsonl")])
    assert code == 2


def test_missing_questions_returns_2(tmp_path):
    code = G.main(["--config", "configs/default.yaml",
                   "--questions", str(tmp_path / "nope.jsonl"), "--dry-run"])
    assert code == 2


def _four_items(tmp_path, cfg):
    """3 正 1 负：正样本跑两条件（6 条 + 负样本 1 条 = 7 条记录）。"""
    specs = []
    for i, (ans, atype, need) in enumerate(
            [("nodule", "short", True), ("yes", "closed", True),
             ("mass", "short", True), ("no", "closed", False)]):
        img = _make_png(str(tmp_path / f"e{i}.png"))
        specs.append(_spec(f"e{i}", img, f"question {i}?", ans,
                           answer_type=atype, need_crop=need, with_box=need))
    return _write_questions(tmp_path, specs)


def test_records_each_get_their_own_line(tmp_path, cfg):
    """同一题的两条记录 id 相同，但 uid 不同；每条 write 都必须真的写进去。

    JsonlWriter 默认按 id 去重：gold_box 与 no_tool 两条 id 都是题号，第二条会被
    静默跳过（write 返回 False），产物条件分布退化成 3:1，answer_change 永远 nan。
    """
    from medtaco.utils.io import JsonlWriter, iter_jsonl

    qpath = _four_items(tmp_path, cfg)
    items = G.build_gate_items(qpath, None, cfg)
    fn = lambda m, i, mt, t: "<answer>nodule</answer>"  # noqa: E731
    records = G.run_gate(items, cfg, fn, workers=2)
    assert len(records) == 7
    uids = [r["uid"] for r in records]
    assert len(set(uids)) == len(uids) == 7
    assert all(r["uid"] == f"{r['id']}|{r['condition']}" for r in records)

    out = str(tmp_path / "rec.jsonl")
    with JsonlWriter(out, id_key="uid", resume=False) as w:
        assert [w.write(r) for r in records] == [True] * len(records)
    persisted = list(iter_jsonl(out))
    assert len(persisted) == len(records)
    conds = [p["condition"] for p in persisted]
    assert conds.count("gold_box") == 3 and conds.count("no_tool") == 4


def test_main_persisted_file_is_gate_judgable(tmp_path, monkeypatch, cfg):
    """端到端：main 落盘后把产物喂给 evaluate_gate 必须配对成功并通过。

    这是唯一覆盖 main 落盘路径的用例。修前：文件只有 4 行（条件分布 3:1），
    answer_change_rate=nan → passed=False，而 main 仍返回 0。
    """
    qpath = _four_items(tmp_path, cfg)
    golds = {f"e{i}": ans for i, ans in
             enumerate(["nodule", "yes", "mass", "no"])}

    def fake_fn(messages, images, max_tokens, temperature):
        blob = json.dumps(messages, ensure_ascii=False)
        q = json.loads(blob)[0]["content"][-1]["text"]
        sid = "e" + q.split("question ")[1].split("?")[0]
        if "【工具返回】" in blob:
            return f"看到了。<answer>{golds[sid]}</answer>"
        return "猜的。<answer>definitely_wrong</answer>"

    # main 走 make_http_model_fn 构造真实端点客户端，这里替换成假模型
    monkeypatch.setattr(G, "make_http_model_fn", lambda *a, **k: fake_fn)
    out = str(tmp_path / "sft_gate.jsonl")
    code = G.main(["--config", _write_cfg(tmp_path), "--questions", qpath,
                   "--trajs", str(tmp_path / "none.jsonl"),
                   "--model-endpoint", "http://127.0.0.1:1/v1",
                   "--out", out, "--workers", "2"])
    assert code == 0

    from medtaco.utils.io import iter_jsonl

    persisted = list(iter_jsonl(out))
    assert len(persisted) == 7
    conds = [p["condition"] for p in persisted]
    assert conds.count("gold_box") == 3 and conds.count("no_tool") == 4

    report = C.evaluate_gate(persisted, {})
    assert report["details"]["answer_change"]["n_paired"] == 3
    assert report["details"]["answer_change"]["rate"] == pytest.approx(1.0)
    assert report["passed"] is True

