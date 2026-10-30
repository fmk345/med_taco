"""tests/test_teacher.py —— medtaco/synth/teacher.py 的单测。

不联网、不读大数据文件：图片全部用 PIL 在 tmp_path 里现造 64x64。
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List

import pytest

from medtaco.synth import teacher as T
from medtaco.utils.config import load_config
from medtaco.utils.schema import Box, Probe, Sample, Step, Trajectory

CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml")


# ---------------------------------------------------------------- 夹具


@pytest.fixture(scope="module")
def cfg():
    return load_config(CFG_PATH)


def _make_img(path: str, size=(64, 64), color=(128, 60, 200)) -> str:
    from PIL import Image

    Image.new("RGB", size, color).save(path)
    return path


@pytest.fixture()
def sample(tmp_path, cfg):
    img = _make_img(str(tmp_path / "orig.png"), (512, 384))
    c1 = _make_img(str(tmp_path / "crop1.png"), (256, 256), (10, 200, 30))
    c2 = _make_img(str(tmp_path / "crop2.jpg"), (200, 100), (240, 240, 10))
    return Sample(
        id="MS-CXR/0001",
        image=img,
        question="Is there a pneumothorax in the right upper lobe?",
        answer="yes",
        dataset="MS-CXR",
        tier="A",
        answer_type="closed",
        boxes=[Box(10, 20, 200, 250, "pneumothorax")],
        meta={"crop_paths": [c1, c2], "source": "MS-CXR"},
    )


def _payload(refusal: bool = False) -> Dict[str, Any]:
    """构造一份符合 schema 的「已解码」结构化输出。"""
    return {
        "probe_no_tool": {
            "answer": "no",
            "confidence": 0.35,
            "rationale": "原图上右上肺野呈弥漫透亮，但分辨率不足以确认胸膜线。",
        },
        "steps": [
            {
                "think": "假设右上肺野存在少量气胸。需要放大该区域确认胸膜线。",
                "tool_name": "crop",
                "tool_x0": 300.0, "tool_y0": 20.0, "tool_x1": 480.0, "tool_y1": 200.0,
                "tool_out_side": None, "tool_center": None, "tool_width": None, "tool_mode": None,
                "result_ok": True,
                "result_text": "",
                "result_image_ref": "CROP_1",
            },
            {
                "think": "裁剪里可见清晰的脏层胸膜线，确认气胸。再用 measure 量宽度。",
                "tool_name": "measure",
                "tool_x0": 310.0, "tool_y0": 40.0, "tool_x1": 360.0, "tool_y1": 90.0,
                "tool_out_side": None, "tool_center": None, "tool_width": None, "tool_mode": "long",
                "result_ok": True,
                "result_text": "12.3 mm",
                "result_image_ref": "",
            },
            {
                "think": "已有足够证据，不需要第三次调用。",
                "tool_name": "none",
                "tool_x0": None, "tool_y0": None, "tool_x1": None, "tool_y1": None,
                "tool_out_side": None, "tool_center": None, "tool_width": None, "tool_mode": None,
                "result_ok": True,
                "result_text": "",
                "result_image_ref": "",
            },
        ],
        "final_answer": "yes",
        "answer_type": "closed",
        "final_rationale": "裁剪后可见胸膜线，初始判断被推翻。",
    }


def _message(payload: Dict[str, Any], stop_reason: str = "end_turn") -> Dict[str, Any]:
    return {
        "id": "msg_x",
        "model": "claude-opus-5",
        "stop_reason": stop_reason,
        "stop_details": None,
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
        "usage": {"input_tokens": 4500, "output_tokens": 4200},
    }


# ---------------------------------------------------------------- schema


def test_schema_is_flat_strict_and_complete(cfg):
    s = T.build_schema(cfg)
    assert s["type"] == "object"
    assert s["additionalProperties"] is False
    assert set(s["required"]) == {"probe_no_tool", "steps", "final_answer", "answer_type", "final_rationale"}

    steps = s["properties"]["steps"]
    assert steps["minItems"] == steps["maxItems"] == 3

    def assert_strict(node: Dict[str, Any], where: str) -> None:
        if node.get("type") == "object":
            assert node["additionalProperties"] is False, where
            assert set(node["required"]) == set(node["properties"]), where
            for k, v in node["properties"].items():
                assert_strict(v, f"{where}.{k}")
        elif node.get("type") == "array":
            assert_strict(node["items"], f"{where}[]")
        elif "anyOf" in node:
            for i, alt in enumerate(node["anyOf"]):
                assert_strict(alt, f"{where}|{i}")

    assert_strict(s, "$")
    # 拍平：3 个 step 共用同一份 properties，没有 oneOf/if-then 分支
    step = steps["items"]
    assert "oneOf" not in step and "anyOf" not in step
    assert "tool_x0" in step["properties"] and "tool_y1" in step["properties"]
    assert set(step["properties"]["tool_name"]["enum"]) == set(T.TOOL_NAMES)


def test_schema_is_deterministic(cfg):
    a = json.dumps(T.build_schema(cfg), sort_keys=True)
    b = json.dumps(T.build_schema(cfg), sort_keys=True)
    assert a == b


# ---------------------------------------------------------------- 视觉 token


@pytest.mark.parametrize(
    "w,h,expect",
    [
        (28, 28, 1),
        (29, 28, 2),
        (512, 384, 19 * 14),      # ceil(512/28)=19, ceil(384/28)=14
        (1024, 1024, 37 * 37),
        (4096, 4096, 147 * 147),  # 21609 → 需要缩
    ],
)
def test_visual_tokens_formula(w, h, expect):
    assert T.visual_tokens(w, h) == expect


def test_fit_size_respects_both_caps():
    nw, nh, tok = T.fit_size(4096, 4096)
    assert max(nw, nh) <= 2576
    assert tok <= T.MAX_VISION_TOKENS
    assert T.visual_tokens(nw, nh) == tok

    # 长边不超但 token 超（例如 2576 x 2576 = 92*92 = 8464）也要被压下来
    nw2, nh2, tok2 = T.fit_size(2576, 2576)
    assert tok2 <= T.MAX_VISION_TOKENS
    assert max(nw2, nh2) <= 2576

    # 小图原样返回
    assert T.fit_size(256, 300) == (256, 300, T.visual_tokens(256, 300))


def test_fit_size_never_returns_zero():
    nw, nh, tok = T.fit_size(100000, 100, max_side_px=2576)
    assert nw >= T.PATCH_PX and nh >= T.PATCH_PX and tok >= 1


# ---------------------------------------------------------------- build_request


def test_build_request_shape(sample, cfg):
    req = T.build_request(sample, cfg)

    assert req["custom_id"] == "traj-MS-CXR_0001"
    p = req["params"]
    assert p["model"] == "claude-opus-5"
    assert p["max_tokens"] == 8192

    content = p["messages"][0]["content"]
    # 3 个 image block + 1 段文本
    imgs = [b for b in content if b["type"] == "image"]
    texts = [b for b in content if b["type"] == "text"]
    assert len(imgs) == 3 and len(texts) == 1
    assert [r["source"]["type"] for r in [*imgs]] == ["base64"] * 3
    assert imgs[0]["source"]["media_type"] == "image/png"
    assert imgs[2]["source"]["media_type"] == "image/jpeg"

    # 结构化输出 + 无 prefill（最后一条消息必须是 user）
    fmt = p["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["additionalProperties"] is False
    assert [m["role"] for m in p["messages"]] == ["user"]
    assert "output_format" not in p
    assert "assistant" not in json.dumps(p["messages"])

    # 该代模型不接受采样参数
    assert "temperature" not in p and "top_p" not in p and "top_k" not in p

    assert req["n_images"] == 3
    assert req["roles"] == ["ORIGINAL", "CROP_1", "CROP_2"]
    assert req["input_tokens"] > req["vision_tokens"] > 0


def test_build_request_input_tokens_is_vision_plus_text(sample, cfg):
    req = T.build_request(sample, cfg)
    expect_vision = sum(
        T.visual_tokens(*sz) for sz in [(512, 384), (256, 256), (200, 100)]
    )
    assert req["vision_tokens"] == expect_vision
    text = [b for b in req["params"]["messages"][0]["content"] if b["type"] == "text"][0]["text"]
    assert req["input_tokens"] == expect_vision + math.ceil(T.approx_text_tokens(text)) + 4 * 4


def test_build_request_omits_temperature_for_legacy_model(sample):
    from medtaco.utils.config import Cfg

    legacy = Cfg({"teacher": dict(T.TEACHER_DEFAULTS, model="claude-sonnet-4-6")})
    req = T.build_request(sample, legacy)
    assert req["params"]["temperature"] == 0.7


def test_build_request_requires_two_crops_fallback(tmp_path, cfg):
    img = _make_img(str(tmp_path / "only.png"))
    s = Sample(id="a", image=img, question="q?", answer="a", answer_type="short", meta={})
    req = T.build_request(s, cfg)
    assert req["roles"] == ["ORIGINAL", "CROP_1", "CROP_2"]
    assert req["n_images"] == 3
    assert all(os.path.exists(p) for p in [s.image])


def test_build_request_raises_on_missing_image(tmp_path, cfg):
    s = Sample(id="b", image=str(tmp_path / "nope.png"), question="q", answer="a")
    with pytest.raises(T.TeacherError, match="不存在"):
        T.build_request(s, cfg)


def test_build_request_raises_on_bad_extension(tmp_path, cfg):
    bad = tmp_path / "x.bmp"
    bad.write_bytes(b"BM")
    s = Sample(id="c", image=str(bad), question="q", answer="a")
    with pytest.raises(T.TeacherError, match="后缀"):
        T.build_request(s, cfg)


def test_build_request_rejects_too_many_images(tmp_path, cfg):
    img = _make_img(str(tmp_path / "o.png"))
    many = [_make_img(str(tmp_path / f"c{i}.png")) for i in range(25)]
    s = Sample(id="d", image=img, question="q", answer="a", meta={"crop_paths": many})
    with pytest.raises(T.TeacherError, match="超过上限"):
        T.build_request(s, cfg)


def test_crop_paths_accepts_dict_form(tmp_path, cfg):
    img = _make_img(str(tmp_path / "o.png"))
    c1 = _make_img(str(tmp_path / "c1.png"))
    s = Sample(id="e", image=img, question="q", answer="a",
               meta={"crops": [{"path": c1, "tool": "crop", "box": [0, 0, 10, 10]},
                               {"image": c1}]})
    assert T.crop_paths_of(s) == [c1, c1]


def test_large_image_gets_resized_metadata(tmp_path, cfg):
    big = _make_img(str(tmp_path / "big.png"), (4096, 3072))
    s = Sample(id="f", image=big, question="q", answer="a", meta={"crop_paths": [big, big]})
    req = T.build_request(s, cfg)
    assert req["resized"], "4096 宽必须触发缩放"
    assert req["resized"][0]["from"] == [4096, 3072]
    assert max(req["resized"][0]["to"]) <= 2576
    # 缩放后单图 token 仍受 4784 限制
    assert req["vision_tokens"] == 3 * T.visual_tokens(*req["resized"][0]["to"])


def test_file_ids_switch_block_source(sample, cfg):
    req = T.build_request(sample, cfg, file_ids={sample.image: "file_abc"})
    first = req["params"]["messages"][0]["content"][0]
    assert first["source"] == {"type": "file", "file_id": "file_abc"}


# ---------------------------------------------------------------- 指令层反 bypassing


def test_instruction_constraints_present(sample, cfg):
    text = T.build_instruction(sample, ["ORIGINAL", "CROP_1", "CROP_2"])
    for cid in T.CONSTRAINT_IDS:
        assert f"[{cid}]" in text, f"缺少约束 {cid}"
    low = text.lower()
    # 先给初始判断，再看裁剪
    assert "before you consider crop_1" in low
    assert "probe_no_tool" in text
    # 明确禁止「先猜答案再补装饰性裁剪」
    assert "answer-first" in low
    assert "decorative crop" in low
    # 问题本身要出现
    assert sample.question in text


# ---------------------------------------------------------------- 成本


def test_estimate_cost_matches_proposal_magnitude(cfg):
    r = T.estimate_cost(50000, cfg, calls_per_traj=1.0)

    # 单条实时：4500*5/1e6 + 4200*25/1e6 = 0.0225 + 0.105 = 0.1275
    assert r["per_traj_realtime_usd"] == pytest.approx(0.1275, rel=1e-6)
    assert r["per_traj_batch_usd"] == pytest.approx(0.06375, rel=1e-6)

    # 实时 0.11–0.13/条 → 5.5k–6.5k
    assert 5500 <= r["realtime_usd"] <= 6500
    # Batch（−50%）：PROPOSAL §5.2 说 2.8k–3.3k
    assert 2800 <= r["batch_usd"] <= 3300
    # 拆成 3 次串行调用 ≈ 0.19/条 → 约 9.3k，作为反例对照
    serial = T.estimate_cost(50000, cfg, calls_per_traj=3.0)
    assert serial["realtime_usd"] > r["realtime_usd"] * 2.5


def test_estimate_cost_scales_linearly(cfg):
    a = T.estimate_cost(1000, cfg)
    b = T.estimate_cost(2000, cfg)
    assert b["realtime_usd"] == pytest.approx(a["realtime_usd"] * 2)
    assert T.estimate_cost(0, cfg)["realtime_usd"] == 0


def test_estimate_cost_rejects_bad_args(cfg):
    with pytest.raises(ValueError):
        T.estimate_cost(-1, cfg)
    with pytest.raises(ValueError):
        T.estimate_cost(10, cfg, calls_per_traj=0)


# ---------------------------------------------------------------- parse_response


def test_parse_response_fields(sample, cfg):
    traj = T.parse_response(_message(_payload()), sample)

    assert isinstance(traj, Trajectory)
    assert traj.id == "traj-MS-CXR_0001"
    assert traj.sample_id == "MS-CXR/0001"
    assert traj.source == "teacher"
    assert traj.final_answer == "yes"
    assert len(traj.steps) == 3

    s0, s1, s2 = traj.steps
    assert s0.is_tool_turn and s0.call.name == "crop"
    assert s0.call.args == {"x0": 300.0, "y0": 20.0, "x1": 480.0, "y1": 200.0}
    assert s0.result.ok is True
    # 观测图指向真实裁剪路径（tools/ops 现场生成的）
    assert s0.result.image == T.crop_paths_of(sample)[0]
    # 执行结果只是「计划」，必须由沙箱重放
    assert s0.result.meta["executed"] is False
    assert s0.result.meta["source"] == "teacher_planned"
    assert s0.result.meta["crop_box"] == [300.0, 20.0, 480.0, 200.0, ""]

    assert s1.is_tool_turn and s1.call.name == "measure"
    assert s1.call.args["mode"] == "long"
    assert s1.result.text == "12.3 mm"

    assert not s2.is_tool_turn and s2.call is None and s2.result is None
    assert "证据" in s2.text

    assert traj.n_calls == 2
    assert len(traj.probes) == 1
    assert isinstance(traj.probes[0], Probe)
    assert traj.probes[0].k == 0 and traj.probes[0].answer == "no"
    assert traj.meta["probes_pending"] is True
    assert traj.meta["probe_contaminated"] is True
    assert traj.meta["probe_confidence"] == 0.35
    assert traj.meta["usage"]["output_tokens"] == 4200
    assert traj.meta["stop_reason"] == "end_turn"


def test_parse_response_roundtrips_through_json(sample):
    traj = T.parse_response(_message(_payload()), sample)
    again = Trajectory.from_json(json.loads(json.dumps(traj.to_json())))
    assert again.to_json() == traj.to_json()


def test_parse_response_accepts_already_decoded_payload(sample):
    traj = T.parse_response(_payload(), sample)
    assert traj.final_answer == "yes"


def test_parse_response_accepts_batch_result(sample):
    batch = {
        "custom_id": "traj-MS-CXR_0001",
        "result": {"type": "succeeded", "message": _message(_payload())},
    }
    assert T.parse_response(batch, sample).final_answer == "yes"


def test_parse_response_handles_fenced_json(sample):
    msg = _message(_payload())
    msg["content"][0]["text"] = "```json\n" + msg["content"][0]["text"] + "\n```"
    assert T.parse_response(msg, sample).final_answer == "yes"


def test_parse_response_refusal_raises(sample):
    msg = _message(_payload(), stop_reason="refusal")
    with pytest.raises(T.TeacherError, match="拒答"):
        T.parse_response(msg, sample)


def test_parse_response_wrong_step_count_raises(sample):
    p = _payload()
    p["steps"] = p["steps"][:2]
    with pytest.raises(T.TeacherError, match="长度 3"):
        T.parse_response(_message(p), sample)


def test_parse_response_missing_probe_raises(sample):
    p = _payload()
    p["probe_no_tool"]["answer"] = ""
    with pytest.raises(T.TeacherError, match="a_1 探针"):
        T.parse_response(_message(p), sample)


def test_parse_response_partial_box_raises(sample):
    p = _payload()
    p["steps"][0]["tool_y1"] = None
    with pytest.raises(T.TeacherError, match="框坐标不完整"):
        T.parse_response(_message(p), sample)


def test_parse_response_bad_json_raises(sample):
    msg = _message(_payload())
    msg["content"][0]["text"] = "{not json"
    with pytest.raises(T.TeacherError, match="不是合法 JSON"):
        T.parse_response(msg, sample)


def test_parse_response_window_tool(sample):
    p = _payload()
    p["steps"][0].update(
        tool_name="window", tool_center=40.0, tool_width=400.0,
        tool_x0=None, tool_y0=None, tool_x1=None, tool_y1=None,
    )
    traj = T.parse_response(_message(p), sample)
    assert traj.steps[0].call.args == {"center": 40.0, "width": 400.0}


def test_parse_response_two_crops_use_call_order_not_step_index(sample):
    """[crop, measure, crop]：第 3 步要拿第 2 张裁剪图，而不是越界退回标签。

    老代码按 step 下标取 ``crop_paths[k]``，第 3 步 k=2 越界 → image 变成
    教师写的 'CROP_2' 字符串，导出时这一串会被当成图片文件路径。
    """
    p = _payload()
    p["steps"][1].update(tool_name="crop", result_image_ref="CROP_2")
    p["steps"][2].update(
        tool_name="crop",
        tool_x0=100.0, tool_y0=100.0, tool_x1=300.0, tool_y1=300.0,
        result_image_ref="CROP_2", result_text="",
    )
    traj = T.parse_response(_message(p), sample)
    c1, c2 = T.crop_paths_of(sample)
    assert traj.steps[0].result.image == c1
    assert traj.steps[1].result.image == c2
    assert traj.steps[2].result.image is None, "只有两张裁剪图，第三次裁剪无图可指"
    assert all(not str(st.result.image or "").startswith("CROP_") for st in traj.steps)


def test_parse_response_never_treats_label_as_image(sample):
    """window/measure 的 result_image_ref 是标签，不能落进 ToolResult.image。"""
    p = _payload()
    p["steps"][0].update(
        tool_name="window", tool_center=40.0, tool_width=400.0,
        tool_x0=None, tool_y0=None, tool_x1=None, tool_y1=None,
        result_image_ref="CROP_1",
    )
    traj = T.parse_response(_message(p), sample)
    assert traj.steps[0].result.image is None
    assert traj.steps[0].result.meta["result_ref"] == "CROP_1"


def test_sanitize_images_repairs_legacy_labels(sample):
    """已经落盘的老轨迹：把 'CROP_1' 改回真实路径，且幂等。"""
    traj = T.parse_response(_message(_payload()), sample)
    c1, c2 = T.crop_paths_of(sample)
    traj.steps[0].result.image = "CROP_1"          # 老数据的坏值
    traj.steps[1].result.image = "CROP_2"          # measure 不该有图
    assert T.sanitize_images(traj, sample) == 2
    assert traj.steps[0].result.image == c1
    assert traj.steps[1].result.image is None
    assert T.sanitize_images(traj, sample) == 0, "幂等"
    assert c2  # 夹具里确实有两张裁剪图，只是这条轨迹只用了一张


# ---------------------------------------------------------------- mock client / run_batch


class _FakeBatches:
    """模拟 messages.batches 的 create/retrieve/results。"""

    def __init__(self, owner: "_FakeClient") -> None:
        self.owner = owner

    def create(self, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.owner.create_calls.append([r["custom_id"] for r in requests])
        self.owner.submitted.extend(r["custom_id"] for r in requests)
        bid = f"batch_{len(self.owner.create_calls)}"

        def _fail_chunk(attempt: int) -> Exception:
            return RuntimeError(f"transient {attempt}")

        if self.owner.fail_times:
            n = self.owner.fail_times.pop(0)
            if n > 0:
                raise _fail_chunk(n)

        self.owner.batches[bid] = {
            "id": bid,
            "processing_status": "in_progress",
            "requests": requests,
        }
        return {"id": bid, "processing_status": "in_progress"}

    def retrieve(self, bid: str) -> Dict[str, Any]:
        b = self.owner.batches[bid]
        self.owner.poll_count += 1
        b["processing_status"] = "ended"
        return b

    def results(self, bid: str):
        b = self.owner.batches[bid]
        items = []
        for r in b["requests"]:
            cid = r["custom_id"]
            if cid in self.owner.always_fail:
                items.append({"custom_id": cid,
                              "result": {"type": "errored", "error": "server_overloaded"}})
            else:
                items.append({"custom_id": cid,
                              "result": {"type": "succeeded", "message": _message(_payload())}})
        return iter(items)


class _FakeClient:
    def __init__(self, fail_times: List[int] | None = None, always_fail: set | None = None):
        self.batches: Dict[str, Any] = {}
        self.create_calls: List[List[str]] = []
        self.submitted: List[str] = []
        self.poll_count = 0
        self.fail_times = list(fail_times or [])
        self.always_fail = always_fail or set()
        self.messages = type("M", (), {})()
        self.messages.batches = _FakeBatches(self)


def _samples(sample, n=3):
    out = []
    for i in range(n):
        s = Sample.from_json(json.loads(json.dumps(sample.to_json())))
        s.id = f"{sample.id}-{i}"
        out.append(s)
    return out


def test_run_batch_happy_path(sample, cfg):
    client = _FakeClient()
    samples = _samples(sample, 3)
    trajs = T.run_batch(samples, cfg, client=client, sleep_fn=lambda s: None, poll_interval=0)

    assert len(trajs) == 3
    assert [t.sample_id for t in trajs] == [s.id for s in samples]
    assert client.create_calls == [[f"traj-MS-CXR_0001-{i}" for i in range(3)]]
    assert all(isinstance(t, Trajectory) for t in trajs)


def test_run_batch_retries_transient_create_errors(sample, cfg):
    cfg2 = load_config(CFG_PATH, {"teacher.max_retries": 3})
    client = _FakeClient(fail_times=[2])  # 前两次抛，第三次成功
    trajs = T.run_batch(_samples(sample, 1), cfg2, client=client,
                        sleep_fn=lambda s: None, poll_interval=0)
    assert len(trajs) == 1
    assert client.poll_count == 1


def test_run_batch_gives_up_after_max_retries(sample, cfg):
    cfg2 = load_config(CFG_PATH, {"teacher.max_retries": 2})
    client = _FakeClient(fail_times=[99] * 10)  # 每次 create 都抛
    sink: List[Dict[str, Any]] = []
    trajs = T.run_batch(_samples(sample, 1), cfg2, client=client, failure_sink=sink,
                        sleep_fn=lambda s: None, poll_interval=0)
    assert trajs == []
    assert sink and sink[0]["stage"] == "batch"


def test_run_batch_is_idempotent_on_existing_ids(sample, cfg):
    client = _FakeClient()
    samples = _samples(sample, 3)

    # 第一次：只提交 2 条
    first = T.run_batch(samples[:2], cfg, client=client, sleep_fn=lambda s: None, poll_interval=0)
    assert len(first) == 2

    # 第二次：传 existing_ids（sample.id 形式），已完成的 2 条不得重复提交
    n_before = len(client.submitted)
    second = T.run_batch(samples, cfg, client=client,
                         existing_ids=[s.id for s in samples[:2]],
                         sleep_fn=lambda s: None, poll_interval=0)
    assert len(second) == 1
    newly = client.submitted[n_before:]
    assert newly == ["traj-MS-CXR_0001-2"]

    # custom_id 形式同样生效
    n_before2 = len(client.submitted)
    third = T.run_batch(samples, cfg, client=client,
                        existing_ids=[t.id for t in first] + [second[0].id],
                        sleep_fn=lambda s: None, poll_interval=0)
    assert third == []
    assert len(client.submitted) == n_before2


def test_run_batch_dedups_duplicate_inputs(sample, cfg):
    client = _FakeClient()
    samples = _samples(sample, 2)
    trajs = T.run_batch(samples + samples, cfg, client=client,
                        sleep_fn=lambda s: None, poll_interval=0)
    assert client.create_calls == [["traj-MS-CXR_0001-0", "traj-MS-CXR_0001-1"]]
    assert len(trajs) == 2


def test_run_batch_records_failed_result(sample, cfg):
    cfg2 = load_config(CFG_PATH, {"teacher.max_retries": 1})
    client = _FakeClient(always_fail={"traj-MS-CXR_0001-1"})
    sink: List[Dict[str, Any]] = []
    trajs = T.run_batch(_samples(sample, 3), cfg2, client=client, failure_sink=sink,
                        sleep_fn=lambda s: None, poll_interval=0)
    assert [t.sample_id for t in trajs] == ["MS-CXR/0001-0", "MS-CXR/0001-2"]
    assert any(x["custom_id"] == "traj-MS-CXR_0001-1" for x in sink)


def test_run_batch_bad_sample_does_not_kill_batch(tmp_path, sample, cfg):
    good1, good2 = _samples(sample, 2)
    broken = Sample(id="broken", image=str(tmp_path / "missing.png"),
                    question="q", answer="a", meta={})
    client = _FakeClient()
    sink: List[Dict[str, Any]] = []
    trajs = T.run_batch([good1, broken, good2], cfg, client=client, failure_sink=sink,
                        sleep_fn=lambda s: None, poll_interval=0)
    assert len(trajs) == 2
    assert sink[0]["stage"] == "build"

    # 坏样本没进 batch
    assert "traj-broken" not in client.submitted


def test_run_batch_chunks_by_request_limit(sample, cfg):
    cfg2 = load_config(CFG_PATH, {"teacher.batch_max_requests": 2})
    client = _FakeClient()
    trajs = T.run_batch(_samples(sample, 5), cfg2, client=client,
                        sleep_fn=lambda s: None, poll_interval=0)
    assert len(trajs) == 5
    assert [len(c) for c in client.create_calls] == [2, 2, 1]


def test_run_batch_empty_input(sample, cfg):
    client = _FakeClient()
    assert T.run_batch([], cfg, client=client, sleep_fn=lambda s: None) == []
    assert client.create_calls == []


def test_ensure_client_without_sdk_raises():
    try:
        import anthropic  # noqa: F401
    except ImportError:
        with pytest.raises(T.TeacherError, match="pip install anthropic"):
            T._ensure_client(None)
    else:  # pragma: no cover - 环境里装了 SDK 时跳过
        pytest.skip("环境已安装 anthropic")
