"""training_set 单测：轨迹渲染、观测重注入顺序、loss mask 位置。"""
import pytest

from medtaco.data.training_set import (
    IMAGE_PLACEHOLDER,
    OBS_HEADER,
    render_trajectory,
    to_rl_record,
    to_sft_record,
)
from medtaco.utils.schema import Box, Sample, Step, ToolCall, ToolResult, Trajectory


def _sample():
    return Sample(id="s1", image="/img/full.png", question="左上肺有结节吗？",
                  answer="有", tier="A", answer_type="closed",
                  boxes=[Box(10, 10, 60, 60, "nodule")])


def _traj_one_call():
    return Trajectory(
        id="t1", sample_id="s1",
        steps=[
            Step(text="先看全图，左上肺似乎有密度增高影。"),
            Step(text="放大看一下", call=ToolCall("crop", {"box": [5, 5, 70, 70]}),
                 result=ToolResult(ok=True, image="/img/crop1.png",
                                   meta={"crop_box": [5, 5, 70, 70]})),
            Step(text="裁剪后确认有结节。<answer>有</answer>"),
        ],
        probes=[],
    )


def test_render_message_order():
    """关键契约：工具返回的图必须是独立 user message，不能塞进 tool message。"""
    r = render_trajectory(_sample(), _traj_one_call())
    roles = [m["role"] for m in r.messages]
    assert roles[0] == "user"
    # 依次: user(原图) / assistant(T1) / assistant(toolcall) / user(观测) / assistant(T2+answer)
    assert roles == ["user", "assistant", "assistant", "user", "assistant"]
    obs_msg = r.messages[3]
    assert any(b["type"] == "image" and b["image"] == "/img/crop1.png"
               for b in obs_msg["content"])
    assert r.images == ["/img/full.png", "/img/crop1.png"]


def test_observation_spans_marked():
    r = render_trajectory(_sample(), _traj_one_call())
    assert len(r.observation_spans) == 1
    s, e = r.observation_spans[0]
    assert r.text[s:e].startswith(OBS_HEADER)
    assert IMAGE_PLACEHOLDER in r.text[s:e]


def test_segments_consistent():
    r = render_trajectory(_sample(), _traj_one_call())
    seg = r.segments
    assert seg.t1[0] >= 0 and seg.t1[1] > seg.t1[0]
    assert len(seg.calls) == 1
    obs_s, obs_e, t2_s, t2_e = seg.calls[0]
    assert obs_s == r.observation_spans[0][0]
    assert t2_s >= obs_e
    # T1 在观测之前
    assert seg.t1[1] <= obs_s


def test_multi_call_segments():
    t = Trajectory(id="t2", sample_id="s1", steps=[
        Step(text="先粗看。"),
        Step(text="c1", call=ToolCall("crop", {"box": [1, 1, 50, 50]}),
             result=ToolResult(ok=True, image="/a.png")),
        Step(text="再放大。"),
        Step(text="c2", call=ToolCall("zoom", {"box": [10, 10, 30, 30]}),
             result=ToolResult(ok=True, image="/b.png")),
        Step(text="<answer>有</answer>"),
    ])
    r = render_trajectory(_sample(), t)
    assert len(r.segments.calls) == 2
    assert r.meta["n_calls"] == 2
    # 两段观测互不重叠且递增
    (a1, b1, _, _), (a2, b2, _, _) = r.segments.calls
    assert b1 <= a2


def test_failed_tool_result_rendered_as_text():
    t = Trajectory(id="t3", sample_id="s1", steps=[
        Step(text="试着裁一下"),
        Step(text="", call=ToolCall("measure", {"box": [0, 0, 10, 10]}),
             result=ToolResult(ok=False, error="missing pixel spacing")),
        Step(text="<answer>无法测量</answer>"),
    ])
    r = render_trajectory(_sample(), t)
    obs = [m for m in r.messages if m["role"] == "user"][1]
    text = " ".join(b.get("text", "") for b in obs["content"])
    assert "missing pixel spacing" in text
    # 失败时不应有任何图被注入
    assert r.images == ["/img/full.png"]


def test_no_tool_trajectory():
    t = Trajectory(id="t4", sample_id="s1",
                   steps=[Step(text="这是全局知识题。<answer>不知道</answer>")])
    r = render_trajectory(_sample(), t)
    assert r.meta["n_calls"] == 0
    assert r.segments.calls == []
    assert r.observation_spans == []
    assert [m["role"] for m in r.messages] == ["user", "assistant"]


def test_export_shapes():
    r = render_trajectory(_sample(), _traj_one_call())
    tr = _traj_one_call()
    sft = to_sft_record(r)
    assert "messages" in sft and "images" in sft and len(sft["images"]) == 2
    rl = to_rl_record(r, tr)
    assert rl["id"] == "t1" and "segments_char" in rl
    assert isinstance(rl["probe_scores"], list)


def _arrow_mangled(msgs):
    """模拟 HF Dataset 的 Arrow 往返：同列 dict 被补成并集 schema。"""
    keys = set()
    for m in msgs:
        for b in m["content"]:
            keys.update(b)
    return [{"role": m["role"],
             "content": [{k: b.get(k) for k in keys} for b in m["content"]]}
            for m in msgs]


def test_obs_message_has_no_literal_image_placeholder():
    """观测消息里不能写字面量占位符：image 块会被模板自己展开，重复写就多算一张图。"""
    r = render_trajectory(_sample(), _traj_one_call())
    obs = r.messages[3]
    texts = [b["text"] for b in obs["content"] if b["type"] == "text"]
    assert texts and all("<|image_pad|>" not in t for t in texts)
    # 文本块不能带 image 键（值哪怕是 None，模板也按「键存在」判图）
    assert all("image" not in b for b in obs["content"] if b["type"] == "text")


def test_clean_messages_repairs_arrow_none_keys():
    """Arrow 补出的 {"image": None} 会让模板把每个文本块当成一张图，必须剥掉。"""
    from medtaco.data.training_set import clean_messages

    r = render_trajectory(_sample(), _traj_one_call())
    dirty = _arrow_mangled(r.messages)
    n_text = sum(1 for m in dirty for b in m["content"] if b.get("type") == "text")
    assert n_text > 0

    # 脏数据里每个文本块都有 image 键 —— 这正是越界的成因
    assert all("image" in b for m in dirty for b in m["content"])

    fixed = clean_messages(dirty)
    assert all(not any(v is None for v in b.values())
               for m in fixed for b in m["content"])
    # 图片块一个不少，文本块一个不多
    assert sum(1 for m in fixed for b in m["content"]
               if b.get("type") == "image") == len(r.images)
    assert sum(1 for m in fixed for b in m["content"]
               if b.get("type") == "text") == n_text


def test_count_dirty_blocks():
    from medtaco.data.training_set import count_dirty_blocks

    r = render_trajectory(_sample(), _traj_one_call())
    assert count_dirty_blocks(r.messages) == 0
    assert count_dirty_blocks(_arrow_mangled(r.messages)) > 0
