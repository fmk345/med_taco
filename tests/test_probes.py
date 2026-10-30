"""tests/test_probes.py —— medtaco/synth/probes.py 的单测。

不联网、不读大数据文件：原图与裁剪图都用 PIL 在 tmp_path 里现造 64x64。
Batch 客户端用本文件里的假实现（只实现 create/retrieve/results 三个方法）。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import pytest

from medtaco.rl.advantage import Segment, ogar_masks, staged_deltas, validate_probes
from medtaco.rl.grader import DeterministicGrader
from medtaco.synth import probes as P
from medtaco.synth.teacher import _Obj
from medtaco.utils.config import load_config
from medtaco.utils.schema import Box, Probe, Sample, Step, ToolCall, ToolResult, Trajectory

CFG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml"
)


# ---------------------------------------------------------------- 夹具


@pytest.fixture(scope="module")
def cfg():
    return load_config(CFG_PATH)


def _make_img(path: str, size=(64, 64), color=(128, 60, 200)) -> str:
    from PIL import Image

    Image.new("RGB", size, color).save(path)
    return path


class FakeBatchClient:
    """离线替身：``messages.batches.{create,retrieve,results}``。

    ``answer_fn(custom_id) -> str`` 决定每个探针请求返回的文本；
    返回 None 表示这一条要失败（errored），用于测重试与失败记录。
    """

    def __init__(self, answer_fn) -> None:
        self.answer_fn = answer_fn
        self.requests: List[Dict[str, Any]] = []   # 全部收到过的请求（跨 batch）
        self._batches: Dict[str, List[Dict[str, Any]]] = {}
        self._n = 0
        self.messages = _Obj(batches=_Obj(
            create=self._create, retrieve=self._retrieve, results=self._results))

    def _create(self, requests: Sequence[Dict[str, Any]], **_: Any) -> Any:
        self._n += 1
        bid = f"fake_batch_{self._n}"
        self._batches[bid] = list(requests)
        self.requests.extend(list(requests))
        return _Obj(id=bid, processing_status="ended")

    def _retrieve(self, batch_id: str, **_: Any) -> Any:
        assert batch_id in self._batches
        return _Obj(id=batch_id, processing_status="ended")

    def _results(self, batch_id: str, **_: Any):
        for req in self._batches.get(batch_id, []):
            cid = str(req["custom_id"])
            text = self.answer_fn(cid)
            if text is None:
                yield _Obj(custom_id=cid, result=_Obj(
                    type="errored", error=_Obj(type="api_error", message="boom")))
                continue
            msg = _Obj(
                model="fake",
                stop_reason="end_turn",
                content=[_Obj(type="text", text=text)],
                usage=_Obj(input_tokens=0, output_tokens=0),
            )
            yield _Obj(custom_id=cid, result=_Obj(type="succeeded", message=msg))


def _k_of(custom_id: str) -> int:
    """从 ``...-k<k>`` 尾巴里取探针编号。"""
    return int(str(custom_id).rsplit("-k", 1)[1])


def _answer_by_k(custom_id: str) -> str:
    """k=0 答错（no），k>=1 答对（yes），便于制造非零 Δ。"""
    return "no" if _k_of(custom_id) == 0 else "yes"


@pytest.fixture()
def images(tmp_path):
    """原图 + 2 张裁剪图（64x64）。"""
    return {
        "orig": _make_img(str(tmp_path / "orig.png"), (64, 64), (10, 10, 10)),
        "c1": _make_img(str(tmp_path / "c1.png"), (64, 64), (200, 10, 10)),
        "c2": _make_img(str(tmp_path / "c2.png"), (64, 64), (10, 200, 10)),
    }


def _make_sample(images, sid: str = "MS-CXR/0001") -> Sample:
    return Sample(
        id=sid,
        image=images["orig"],
        question="Is there a pneumothorax?",
        answer="yes",
        dataset="MS-CXR",
        tier="A",
        answer_type="closed",
        boxes=[Box(5, 5, 40, 40, "pneumothorax")],
        meta={"crop_paths": [images["c1"], images["c2"]]},
    )


def _make_traj(n_calls: int, images, tid: str = "traj-probe-demo",
               final_answer: str = "yes") -> Trajectory:
    """造一条 n_calls 次调用的轨迹；result.image 故意先写坏标签，测 sanitize。"""
    steps: List[Step] = []
    for i in range(n_calls):
        steps.append(Step(
            text=f"[test] 第 {i + 1} 轮裁剪",
            call=ToolCall(name="crop", args={"x0": 5.0, "y0": 5.0, "x1": 40.0, "y1": 40.0}),
            result=ToolResult(ok=True, image=f"CROP_{i + 1}", text=""),
        ))
    while len(steps) < 3:  # 末步无 call（与真实数据一致）
        steps.append(Step(text="[test] 给出答案"))
    return Trajectory(
        id=tid,
        sample_id="MS-CXR/0001",
        steps=steps,
        final_answer=final_answer,
        probes=[Probe(k=0, answer="placeholder", score=0, parse_ok=True)],
        delta_k=[],
        delta_total=0,
        source="teacher",
        meta={"probes_pending": True, "n_calls": n_calls},
    )


def _segments(n_calls: int) -> Segment:
    """与 n_calls 匹配的分段（每段 10 个 token，含 obs 区间）。"""
    calls = []
    pos = 10
    for _ in range(n_calls):
        calls.append((pos, pos + 4, pos + 4, pos + 10))  # obs 4 token + t2 6 token
        pos += 10
    return Segment(t1=(0, 10), calls=calls)


# ---------------------------------------------------------------- custom_id


def test_custom_id_charset_and_uniqueness():
    cids = [P.probe_custom_id("traj-chestx_ray14:abc#G1-en-10", k) for k in range(4)]
    assert len(set(cids)) == 4
    for cid in cids:
        assert len(cid) <= 64
        assert all(c.isalnum() or c in "-_" for c in cid), cid
        assert cid.startswith("probe-") and cid.endswith(tuple(f"-k{k}" for k in range(4)))
    # 消毒后可能撞车的两个 id 也必须区分开（哈希后缀）
    a = P.probe_custom_id("traj:a", 0)
    b = P.probe_custom_id("traj#a", 0)
    assert a != b


def test_custom_id_truncates_long_ids():
    cid = P.probe_custom_id("x" * 300, 7)
    assert len(cid) <= 64 and cid.endswith("-k7")
    with pytest.raises(P.ProbeError):
        P.probe_custom_id("", 0)
    with pytest.raises(P.ProbeError):
        P.probe_custom_id("t", -1)


# ---------------------------------------------------------------- 请求构造


@pytest.mark.parametrize("n_calls", [1, 2])
def test_build_probe_request_shape(cfg, images, n_calls):
    sample = _make_sample(images)
    traj = _make_traj(n_calls, images)
    for k in range(n_calls):
        req = P.build_probe_request(traj, sample, k, cfg)
        assert req["custom_id"] == P.probe_custom_id(traj.id, k)
        params = req["params"]
        # 上限来自 cfg.rl.probe.max_tokens（default.yaml = 32），不再是写死的 64
        assert P.PROBE_MAX_TOKENS == 32
        assert params["max_tokens"] == P.probe_max_tokens(cfg) == 32
        # 禁止 prefill：只有一条 user 消息
        assert len(params["messages"]) == 1
        assert params["messages"][0]["role"] == "user"
        blocks = params["messages"][0]["content"]
        imgs = [b for b in blocks if b["type"] == "image"]
        assert len(imgs) == k + 1  # 原图 + 前 k 张裁剪图
        assert blocks[-1]["type"] == "text"
        # 探针要自由文本，不能带 json_schema
        assert "output_config" not in params
        # 至少第一块是 base64 图，且指令不许模型解释
        assert imgs[0]["source"]["type"] == "base64"
        assert "no explanation" in blocks[-1]["text"].lower()
        # 图片路径已被 sanitize 修正成真实文件
        seen = P._prefix_images(traj, k)
        assert len(seen) == k
        assert all(os.path.exists(p) for p in seen)


def test_build_probe_request_rejects_k_equals_n_calls(cfg, images):
    sample = _make_sample(images)
    traj = _make_traj(2, images)
    with pytest.raises(P.ProbeError):
        P.build_probe_request(traj, sample, 2, cfg)  # a_K 复用 final_answer，不发请求


# ---------------------------------------------------------------- max_tokens


def test_probe_max_tokens_reads_cfg_and_clamps_to_32(images):
    """64 是写死的旧值；正确口径是读 ``rl.probe.max_tokens`` 且钳在 §7.1 的 32。"""
    sample = _make_sample(images)
    traj = _make_traj(1, images)

    def _mt(cfg_obj):
        return P.build_probe_request(traj, sample, 0, cfg_obj)["params"]["max_tokens"]

    # 配置缺省 -> 32
    assert P.probe_max_tokens(None) == 32
    assert P.probe_max_tokens({}) == 32
    # 显式小值原样生效
    assert _mt({"rl": {"probe": {"max_tokens": 8}}}) == 8
    assert _mt({"probe": {"max_tokens": 16}}) == 16
    # 超过 32 一律钳到 32，并 warn（探针长度上限是 PROPOSAL 硬约束）
    with pytest.warns(RuntimeWarning):
        assert _mt({"rl": {"probe": {"max_tokens": 128}}}) == 32
    assert P.probe_max_tokens({"rl": {"probe": {"max_tokens": 999}}}) == 32
    # 非法值回落 32，不抛
    with pytest.warns(RuntimeWarning):
        assert P.probe_max_tokens({"rl": {"probe": {"max_tokens": "abc"}}}) == 32
    with pytest.warns(RuntimeWarning):
        assert P.probe_max_tokens({"rl": {"probe": {"max_tokens": 0}}}) == 32


def test_effective_max_tokens_is_recorded_in_meta(cfg, images):
    """探针 meta 要记真正生效的 max_tokens，便于复盘是否被长度上限截断。"""
    sample = _make_sample(images)
    traj = _make_traj(1, images, tid="traj-mt")
    client = FakeBatchClient(lambda _cid: "yes")
    P.collect_probes([traj], [sample], cfg, client,
                     sleep_fn=lambda _s: None, now_fn=lambda: 0.0)
    assert traj.meta["probe_max_tokens"] == P.probe_max_tokens(cfg) == 32
    assert client.requests[0]["params"]["max_tokens"] == 32


def test_prefix_images_skips_empty_and_does_not_leak_later_calls(cfg, images):
    """第 1 次调用没有返回图时，a_1 不能拿第 2 次的图来凑。"""
    sample = _make_sample(images)
    traj = _make_traj(2, images)
    # measure 类调用只回数值、没有返回图（非 crop/zoom，sanitize 也不会给它补图）
    traj.steps[0].call = ToolCall(name="measure", args={"x0": 5.0, "y0": 5.0, "x1": 40.0, "y1": 40.0})
    traj.steps[0].result = ToolResult(ok=True, image=None, text="12.3 mm")
    P.build_probe_request(traj, sample, 1, cfg)
    assert P._prefix_images(traj, 1) == []
    assert len(P._prefix_images(traj, 2)) == 1


def test_measure_turn_before_crop_keeps_instruction_in_sync(cfg, images):
    """[measure, crop] 的轨迹：图片编号必须与实际附件一一对应，文本观测不能丢。

    修前行为：指令按 k 生成（"IMAGE 2 = CROP_1 (returned by tool call 1)"），
    而附件按"返回了图的工具轮"生成——k=1 时只有原图、却宣称有第二张图，
    且 MEASURE_RESULT 这样的文本观测根本不进提示词。
    """
    sample = _make_sample(images)
    traj = _make_traj(1, images)
    traj.steps[0].call = ToolCall(name="measure", args={"x0": 5.0, "y0": 5.0, "x1": 40.0, "y1": 40.0})
    # [measure, crop, crop]
    traj = _make_traj(3, images)
    traj.steps[0].call = ToolCall(name="measure", args={"x0": 5.0, "y0": 5.0, "x1": 40.0, "y1": 40.0})
    traj.steps[0].result = ToolResult(ok=True, image=None, text="12.3 mm")
    traj.steps[1].call = ToolCall(name="crop", args={"x0": 5.0, "y0": 5.0, "x1": 40.0, "y1": 40.0})
    traj.steps[1].result = ToolResult(ok=True, image="CROP_1", text="")
    traj.steps[2].call = ToolCall(name="crop", args={"x0": 10.0, "y0": 10.0, "x1": 30.0, "y1": 30.0})
    traj.steps[2].result = ToolResult(ok=True, image="CROP_2", text="")

    # k=1：只执行了 measure，只有原图
    req = P.build_probe_request(traj, sample, 1, cfg)
    blocks = req["params"]["messages"][0]["content"]
    assert len([b for b in blocks if b["type"] == "image"]) == 1
    text = blocks[-1]["text"]
    assert "IMAGE 2" not in text                      # 没有第二张图，就不能宣称有
    assert "CROP_1" not in text
    assert "12.3 mm" in text                          # measure 的文本观测必须进提示词
    assert "tool call 1 (measure)" in text
    assert "ORIGINAL image only" in text              # 附件描述与实际一致

    # k=2：measure + crop，附件 = 原图 + 1 张；编号必须指向第 2 次调用
    req2 = P.build_probe_request(traj, sample, 2, cfg)
    blocks2 = req2["params"]["messages"][0]["content"]
    imgs2 = [b for b in blocks2 if b["type"] == "image"]
    text2 = blocks2[-1]["text"]
    assert len(imgs2) == 2
    assert "IMAGE 2 = return image of tool call 2 (crop)" in text2
    assert "12.3 mm" in text2
    # 实际附件数与指令里的图片数一致（这正是缺陷 3 的核心）
    assert "IMAGE 3" not in text2


# ---------------------------------------------------------------- 主流程


@pytest.mark.parametrize("n_calls", [0, 1, 2])
def test_collect_probes_end_to_end(cfg, images, n_calls):
    sample = _make_sample(images)
    traj = _make_traj(n_calls, images)
    client = FakeBatchClient(_answer_by_k)

    out = P.collect_probes([traj], [sample], cfg, client,
                           sleep_fn=lambda _s: None, now_fn=lambda: 0.0)

    K = n_calls
    # 硬不变量：探针数 = K+1（K=0 时只有 a_0，由 final_answer 判分得到）
    assert len(traj.probes) == K + 1
    assert [p.k for p in traj.probes] == list(range(K + 1))
    assert len(traj.delta_k) == K
    assert traj.delta_k == staged_deltas([p.score for p in traj.probes])
    assert traj.delta_total == traj.probes[-1].score - traj.probes[0].score
    assert traj.meta["probes_pending"] is False
    assert traj.meta["probe_source"] == "recollected"
    assert all(p.score in (-1, 0, 1) for p in traj.probes)

    # a_0 = "no"（错）；k>=1 与 a_K 都是 "yes"（对）
    if K >= 1:
        assert traj.probes[0].score == 0
        assert traj.probes[1].score == 1
        assert traj.probes[-1].answer == traj.final_answer
    else:
        # K=0：a_0 就是 final_answer 本身（"yes" → 1）
        assert traj.probes[0].answer == traj.final_answer
        assert traj.probes[0].score == 1
        assert traj.delta_total == 0

    # 实际发出的请求数 = K，custom_id 唯一，图片数 = k+1
    assert len(client.requests) == K
    cids = [r["custom_id"] for r in client.requests]
    assert len(set(cids)) == K
    for r in client.requests:
        k = _k_of(r["custom_id"])
        assert len([b for b in r["params"]["messages"][0]["content"] if b["type"] == "image"]) == k + 1

    # ogar_masks 的硬不变量必须满足（不抛异常且掩码长度正确）
    res = ogar_masks([p.score for p in traj.probes], _segments(K))
    assert len(res.token_mask) > 0
    assert len(res.call_indices) == K
    ok, errs = validate_probes(traj)
    assert ok, errs

    assert out["n_completed"] == 1
    assert out["n_requests"] == K
    assert out["n_failed"] == 0
    assert out["completed_ids"] == [traj.id]


def test_collect_probes_idempotent(cfg, images):
    sample = _make_sample(images)
    traj = _make_traj(2, images)
    client = FakeBatchClient(_answer_by_k)

    first = P.collect_probes([traj], [sample], cfg, client,
                             sleep_fn=lambda _s: None, now_fn=lambda: 0.0)
    assert first["n_requests"] == 2
    snapshot = ([p.to_json() for p in traj.probes], list(traj.delta_k), traj.delta_total)

    # 第二次：已经完成，一条请求都不该发
    second = P.collect_probes([traj], [sample], cfg, client,
                              sleep_fn=lambda _s: None, now_fn=lambda: 0.0)
    assert second["n_requests"] == 0
    assert second["n_already"] == 1
    assert second["n_completed"] == 0
    assert ([p.to_json() for p in traj.probes], list(traj.delta_k), traj.delta_total) == snapshot

    # 换一条新轨迹但把 custom_id 全放进 existing_ids：同样跳过
    fresh = _make_traj(2, images, tid="traj-other")
    skip = [P.probe_custom_id("traj-other", k) for k in range(2)]
    third = P.collect_probes([fresh], [sample], cfg, client, existing_ids=skip,
                             sleep_fn=lambda _s: None, now_fn=lambda: 0.0)
    assert third["n_requests"] == 0
    assert third["n_incomplete"] == 1
    assert fresh.meta["probes_pending"] is True


def test_existing_ids_can_carry_answers(cfg, images):
    """``existing_ids`` 传 {custom_id: 答案} 时直接复用，不再提交。"""
    sample = _make_sample(images)
    traj = _make_traj(1, images, tid="traj-reuse")
    client = FakeBatchClient(lambda _cid: "yes")
    prior = {P.probe_custom_id("traj-reuse", 0): "yes"}
    out = P.collect_probes([traj], [sample], cfg, client, existing_ids=prior,
                           sleep_fn=lambda _s: None, now_fn=lambda: 0.0)
    assert out["n_requests"] == 0
    assert out["n_completed"] == 1
    assert [p.score for p in traj.probes] == [1, 1]


def test_collect_probes_records_failures_and_retries(cfg, images):
    sample = _make_sample(images)
    traj = _make_traj(2, images)
    failed: set = set()

    def flaky(cid: str) -> Optional[str]:
        if _k_of(cid) == 1 and cid not in failed:
            failed.add(cid)
            return None  # 第一次失败，第二轮重试成功
        return _answer_by_k(cid)

    client = FakeBatchClient(flaky)
    sink: List[Dict[str, Any]] = []
    out = P.collect_probes([traj], [sample], cfg, client, failure_sink=sink,
                           sleep_fn=lambda _s: None, now_fn=lambda: 0.0)
    assert out["n_completed"] == 1
    assert out["n_failed"] == 0  # 重试成功后不再记失败
    assert len(traj.probes) == 3
    assert failed  # 的确有一条被重试过
    assert len(client.requests) == 3  # 2 条正常 + 1 条重试


def test_collect_probes_failure_keeps_pending(cfg, images):
    sample = _make_sample(images)
    traj = _make_traj(1, images, tid="traj-broken")
    client = FakeBatchClient(lambda _cid: None)  # 全部失败
    sink: List[Dict[str, Any]] = []
    out = P.collect_probes([traj], [sample], cfg, client, failure_sink=sink,
                           sleep_fn=lambda _s: None, now_fn=lambda: 0.0)
    assert out["n_completed"] == 0
    assert out["n_incomplete"] == 1
    assert traj.meta["probes_pending"] is True
    assert traj.probes == [Probe(k=0, answer="placeholder", score=0, parse_ok=True)]
    assert sink and sink[0]["traj_id"] == "traj-broken"


def test_out_path_writes_and_resumes(cfg, images, tmp_path):
    sample = _make_sample(images)
    out_path = str(tmp_path / "probes.jsonl")
    traj = _make_traj(2, images, tid="traj-disk")
    client = FakeBatchClient(_answer_by_k)

    P.collect_probes([traj], [sample], cfg, client, out_path=out_path,
                     sleep_fn=lambda _s: None, now_fn=lambda: 0.0)
    with open(out_path, "r", encoding="utf-8") as f:
        lines = [json.loads(x) for x in f if x.strip()]
    assert len(lines) == 1
    rec = lines[0]
    assert rec["id"] == "traj-disk"
    assert len(rec["probes"]) == 3 and len(rec["delta_k"]) == 2
    assert rec["traj"]["meta"]["probes_pending"] is False

    # 新进程/新对象 + 空 client：靠 out_path 里已写全的行直接补齐
    fresh = _make_traj(2, images, tid="traj-disk")
    empty = FakeBatchClient(lambda _cid: None)
    out = P.collect_probes([fresh], [sample], cfg, empty, out_path=out_path,
                           sleep_fn=lambda _s: None, now_fn=lambda: 0.0)
    assert out["n_requests"] == 0
    assert out["n_adopted"] == 1
    assert [p.score for p in fresh.probes] == [p.score for p in traj.probes]
    assert fresh.delta_k == traj.delta_k
    # 落盘幂等：没有产生第二行
    with open(out_path, "r", encoding="utf-8") as f:
        assert len([x for x in f if x.strip()]) == 1


def test_out_path_overwrites_incomplete_same_id_row(cfg, images, tmp_path):
    """文件里已有同 id **但不完整**的旧行时，新结果必须真的落到磁盘上。

    修前行为：``collect_probes`` 用 ``utils/io.JsonlWriter`` 落盘，而它按 id 去重，
    撞到已见 id 时静默返回 False——``_finalize`` 在内存里修好了，磁盘上那条
    老口径（K=0、只有 2 个探针）的旧行却永远不变，``n_completed`` 还报告成功。
    """
    sample = _make_sample(images)
    out_path = str(tmp_path / "probes.jsonl")
    traj = _make_traj(2, images, tid="traj-dup")

    # 手工造一条同 id 的旧行：老口径，probes 只有 2 条、delta_k 缺项、待采
    stale = _make_traj(2, images, tid="traj-dup", final_answer="")
    stale.probes = [Probe(k=0, answer="stale", score=0, parse_ok=True),
                    Probe(k=1, answer="stale", score=0, parse_ok=True)]
    stale.delta_k = []
    stale.delta_total = 0
    stale.meta = {"probes_pending": True, "n_calls": 2}
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "traj-dup", "traj": stale.to_json(),
                            "probes": [p.to_json() for p in stale.probes],
                            "delta_k": [], "delta_total": 0}, ensure_ascii=False) + "\n")

    client = FakeBatchClient(_answer_by_k)
    out = P.collect_probes([traj], [sample], cfg, client, out_path=out_path,
                           sleep_fn=lambda _s: None, now_fn=lambda: 0.0)
    assert out["n_completed"] == 1
    assert out["n_adopted"] == 0          # 旧行不完整，不能被采纳

    with open(out_path, "r", encoding="utf-8") as f:
        lines = [json.loads(x) for x in f if x.strip()]
    # 同 id 只剩最后一行（压缩过），且内容是新结果而不是旧行
    assert len(lines) == 1
    rec = lines[0]
    assert rec["id"] == "traj-dup"
    assert len(rec["probes"]) == 3 and len(rec["delta_k"]) == 2
    assert rec["traj"]["meta"]["probes_pending"] is False
    assert [p["score"] for p in rec["probes"]] == [p.score for p in traj.probes]
    assert rec["probe_meta"]["probe_max_tokens"] == 32

    # 再续跑一次：这次应当能采纳磁盘结果，且不产生第二行
    fresh = _make_traj(2, images, tid="traj-dup")
    empty = FakeBatchClient(lambda _cid: None)
    out2 = P.collect_probes([fresh], [sample], cfg, empty, out_path=out_path,
                            sleep_fn=lambda _s: None, now_fn=lambda: 0.0)
    assert out2["n_requests"] == 0 and out2["n_adopted"] == 1
    assert [p.score for p in fresh.probes] == [p.score for p in traj.probes]
    with open(out_path, "r", encoding="utf-8") as f:
        assert len([x for x in f if x.strip()]) == 1


# ---------------------------------------------------------------- 完成度判定


def test_missing_probes_pending_key_is_not_complete(images):
    """K=0 且 meta 为空（键缺失）时不能判成"已完成"，否则占位会被冻结成 a_0。"""
    traj = _make_traj(0, images, tid="t-nokey")
    traj.meta = {}
    traj.probes = [Probe(k=0, answer="placeholder", score=0, parse_ok=True)]
    traj.delta_k = []
    assert P.probes_complete(traj) is False


def test_probe_source_is_required_for_completion(images):
    """缺 ``probe_source`` 的探针无法区分"重采过"与"教师自述"，一律算未完成。"""
    traj = _make_traj(1, images, tid="t-nosrc")
    traj.probes = [Probe(k=0, answer="no", score=0), Probe(k=1, answer="yes", score=1)]
    traj.delta_k = [1]
    traj.meta = {"probes_pending": False}
    assert P.probes_complete(traj) is False
    traj.meta["probe_source"] = "recollected"
    assert P.probes_complete(traj) is True


def test_adopt_copies_all_probe_meta_keys(cfg, images, tmp_path):
    """采纳与写入必须搬同一份 meta 字段集，否则内存重采与磁盘采纳产出不可复现。"""
    sample = _make_sample(images)
    out_path = str(tmp_path / "probes.jsonl")
    src = _make_traj(1, images, tid="traj-meta", final_answer="")
    client = FakeBatchClient(lambda _cid: "")
    P.collect_probes([src], [sample], cfg, client, out_path=out_path,
                     sleep_fn=lambda _s: None, now_fn=lambda: 0.0)
    written = {k: src.meta[k] for k in P.PROBE_META_KEYS if k in src.meta}
    # 采完的轨迹一定带上这些字段（probe_skip_reason 只属于"跳过"状态，这不算）
    assert {"probes_pending", "probe_source", "probe_model",
            "probe_max_tokens", "probe_empty_ks"} <= set(written)
    with open(out_path, "r", encoding="utf-8") as f:
        rec = json.loads(f.readline())
    assert rec["probe_meta"] == written

    dst = _make_traj(1, images, tid="traj-meta")
    adopted = P.collect_probes([dst], [sample], cfg, FakeBatchClient(lambda _c: None),
                               out_path=out_path,
                               sleep_fn=lambda _s: None, now_fn=lambda: 0.0)
    assert adopted["n_adopted"] == 1
    for key in written:
        assert dst.meta.get(key) == written[key], key


def test_duplicate_traj_ids_do_not_submit_duplicate_custom_ids(cfg, images):
    """两条轨迹撞同一个 id 时，绝不能提交重复 custom_id（Batch API 会整批拒收）。"""
    sample = _make_sample(images)
    a = _make_traj(1, images, tid="dup", final_answer="yes")
    b = _make_traj(1, images, tid="dup", final_answer="yes")
    client = FakeBatchClient(_answer_by_k)
    sink: List[Dict[str, Any]] = []
    out = P.collect_probes([a, b], [sample], cfg, client, failure_sink=sink,
                           sleep_fn=lambda _s: None, now_fn=lambda: 0.0)

    cids = [r["custom_id"] for r in client.requests]
    assert len(cids) == len(set(cids)) == 1        # 只发一条，没有重复
    assert out["n_completed"] == 1
    assert out["n_incomplete"] == 1
    dup = [s for s in sink if s["stage"] == "duplicate_id"]
    assert len(dup) == 1 and dup[0]["traj_id"] == "dup"
    # 后到的那条保持待采，不被错误地判成完成
    assert b.meta["probes_pending"] is True
    assert a.meta["probes_pending"] is False


def test_same_traj_object_twice_does_not_resubmit(cfg, images):
    """同一个 Trajectory 对象在列表里出现两次时，第二次不能再发一遍请求。

    与上一条的区别：这里撞车的是**同一个对象**，owner == id(traj)，
    只用 custom_id 判重会被『是自己』绕过（缺陷 9）。去重要按 (cid, id(traj)) 做。
    """
    sample = _make_sample(images)
    a = _make_traj(2, images, tid="same", final_answer="yes")
    client = FakeBatchClient(_answer_by_k)
    sink: List[Dict[str, Any]] = []
    out = P.collect_probes([a, a], [sample], cfg, client, failure_sink=sink,
                           sleep_fn=lambda _s: None, now_fn=lambda: 0.0)

    cids = [r["custom_id"] for r in client.requests]
    assert cids == [P.probe_custom_id("same", 0),
                    P.probe_custom_id("same", 1)]   # 每个 k 只发一次
    assert out["n_completed"] == 1
    assert out["n_incomplete"] == 0                # 没有多余副本需要待采
    assert a.meta["probes_pending"] is False
    # 重复出现不算 id 撞车，不该被报成 duplicate_id
    assert not [s for s in sink if s["stage"] == "duplicate_id"]


# ---------------------------------------------------------------- 待采清单
def test_pending_probe_keys(cfg, images):
    sample = _make_sample(images)
    trajs = [
        _make_traj(0, images, tid="t0"),
        _make_traj(2, images, tid="t1"),
        _make_traj(1, images, tid="t2"),
    ]
    keys = P.pending_probe_keys(trajs, [sample], cfg)
    # K=0 的 t0 不发请求（a_0 = final_answer），K>=1 的发 0..K-1
    assert keys == [("t1", "MS-CXR/0001", 0), ("t1", "MS-CXR/0001", 1),
                    ("t2", "MS-CXR/0001", 0)]

    # 采完之后清单里不再出现
    client = FakeBatchClient(_answer_by_k)
    P.collect_probes(trajs, [sample], cfg, client,
                     sleep_fn=lambda _s: None, now_fn=lambda: 0.0)
    assert P.pending_probe_keys(trajs, [sample], cfg) == []


def test_pending_probe_keys_skips_trajs_without_sample(cfg, images):
    trajs = [_make_traj(1, images, tid="orphan")]
    assert P.pending_probe_keys(trajs, [], cfg) == []


# ---------------------------------------------------------------- 判分口径


def test_empty_probe_answers_score_zero_and_are_not_in_delta(cfg, images):
    """空/纯空白答案 = 解析失败：记 0 且不进 Δ（PROPOSAL §7.1），绝不记 -1。

    修前行为：空串被 grader 判成 ``empty_pred -> -1``，再配上一条答对的 a_K
    就造出 ``delta_k=[2]`` 的极端 Δ。修法是 ``parse_ok=False`` + score 0，
    并让 ``advantage.validate_probes`` 的 ``parse_ok`` 检查真正可达。
    """
    sample = _make_sample(images)
    traj = _make_traj(1, images, final_answer="")  # 最终答案也为空
    client = FakeBatchClient(lambda _cid: "")
    P.collect_probes([traj], [sample], cfg, client,
                     sleep_fn=lambda _s: None, now_fn=lambda: 0.0)
    assert [p.score for p in traj.probes] == [0, 0]
    assert [p.parse_ok for p in traj.probes] == [False, False]
    assert traj.delta_k == [0]          # 解析失败不进 Δ
    assert traj.delta_total == 0
    assert traj.meta["probe_empty_ks"] == [0, 1]
    # 解析失败的探针必须被 validate_probes 拦下，而不是伪造 parse_ok=True 放行
    ok, errs = validate_probes(traj)
    assert not ok and any("parse" in str(e).lower() for e in errs), errs


def test_blank_but_nonempty_answer_is_parse_failure(cfg, images):
    """纯空白 / "n/a" 这类"没答"同样是解析失败，不是答错。"""
    sample = _make_sample(images)
    traj = _make_traj(1, images, final_answer="   \n\t ")
    client = FakeBatchClient(lambda _cid: "  ")
    P.collect_probes([traj], [sample], cfg, client,
                     sleep_fn=lambda _s: None, now_fn=lambda: 0.0)
    assert [p.score for p in traj.probes] == [0, 0]
    assert [p.parse_ok for p in traj.probes] == [False, False]
    assert traj.meta["probe_empty_ks"] == [0, 1]


def test_long_samples_are_not_gradable(cfg, images):
    """long 题型进不了 Δ 通道：不发请求、不写 probes_pending=False，单列计数。"""
    sample = _make_sample(images)
    sample.answer_type = "long"
    sample.answer = "focal opacity with well-defined borders"
    traj = _make_traj(1, images, final_answer="focal opacity")
    client = FakeBatchClient(lambda _cid: "focal opacity")
    out = P.collect_probes([traj], [sample], cfg, client,
                           sleep_fn=lambda _s: None, now_fn=lambda: 0.0)
    assert out["n_requests"] == 0 and client.requests == []
    assert out["n_completed"] == 0
    assert out["n_not_gradable"] == 1
    assert out["not_gradable_ids"] == [traj.id]
    # 不清洗：仍然是待采的教师占位，不会被当成可用轨迹
    assert traj.meta["probes_pending"] is True
    assert traj.meta["probe_skip_reason"] == "not_gradable"
    assert "probe_source" not in traj.meta
    assert traj.probes == [Probe(k=0, answer="placeholder", score=0, parse_ok=True)]


def test_grader_instance_is_used_for_scoring(cfg, images):
    """确认判分走的是 DeterministicGrader（closed 题的 yes/no 极性）。"""
    sample = _make_sample(images)
    traj = _make_traj(1, images)
    client = FakeBatchClient(lambda _cid: "Yes.")
    P.collect_probes([traj], [sample], cfg, client,
                     sleep_fn=lambda _s: None, now_fn=lambda: 0.0)
    assert traj.probes[0].answer == "Yes."
    assert traj.probes[0].score == DeterministicGrader(cfg).score(sample, "Yes.")
