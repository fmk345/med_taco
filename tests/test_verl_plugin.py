"""medtaco/rl/verl_plugin 的离线单测。

只测**不需要 verl / GPU / 真实 tokenizer** 的部分：
  - 模块能 import（verl 缺失时走空装饰器）
  - adv_est 调 advantage.py 的 ogar/staged（用假模块注入，同伴模块可能还没写完）
  - rollout 的探针 prompt 拼接
  - 工具返回后「独立 user message 重注入」的消息顺序
  - 观测 token 打标 / 两段 mask 的宽度与置零 / 零方差组跳过
"""
from __future__ import annotations

import hashlib
import importlib
import types

import numpy as np
import pytest

from medtaco.rl.verl_plugin import adv_est, rollout

REPO = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/fengmingkuan/workspace/med_taco"


# ------------------------------------------------------------------ 假 tokenizer / 假模型


def fake_tokenize(text: str) -> list[int]:
    """确定性、可复现的假 tokenizer：每个词一个稳定 id。"""
    out = []
    for w in str(text).split():
        h = int(hashlib.md5(w.encode("utf-8")).hexdigest()[:8], 16)
        out.append(h % 32000)
    return out


class FakeGenerator:
    """主生成与探针生成**分两条脚本队列**：探针 temperature=0，主生成 temperature=1。"""

    def __init__(self, main: list[str], probe: list[str] | None = None) -> None:
        self.main = list(main)
        self.probe = list(probe or [])
        self.calls: list[dict] = []

    def __call__(self, input_ids, max_tokens=64, temperature=1.0, **kw):
        self.calls.append(
            {"input_ids": list(input_ids), "max_tokens": max_tokens, "temperature": temperature, **kw}
        )
        is_probe = temperature == 0.0
        queue = self.probe if is_probe else self.main
        text = queue.pop(0) if queue else ""
        return text, fake_tokenize(text)

    def probe_calls(self) -> list[dict]:
        return [c for c in self.calls if c["temperature"] == 0.0]


def fake_run_tool(call):
    from medtaco.utils.schema import ToolResult

    return ToolResult(ok=True, image="/tmp/crop_0.jpg", text="cropped 512x512", meta={"crop_box": [1, 2, 3, 4]})


def make_loop(main: list[str], probe: list[str] | None = None, scores=None, **cfg_kw):
    gen = FakeGenerator(main, probe)
    queue = list(scores or [])

    def grade(answer: str, k: int):
        return (queue.pop(0) if queue else 0), True

    cfg = rollout.RolloutCfg(
        max_turns=cfg_kw.pop("max_turns", 3),
        image_token_count=4,
        probe_max_tokens=32,
        **cfg_kw,
    )
    loop = rollout.TacoAgentLoop(
        cfg=cfg,
        tokenize=fake_tokenize,
        generate=gen,
        run_tool=fake_run_tool,
        grade=grade,
    )
    return loop, gen


# ------------------------------------------------------------------ 1. import 不崩


def test_import_without_verl_does_not_crash():
    assert adv_est.HAS_VERL is False
    assert callable(adv_est.register_adv_est)
    assert adv_est.ADV_NAME == "taco_staged"

    @adv_est.register_adv_est("my_est")
    def _f(x):
        return x

    assert _f(1) == 1

    @adv_est.register_adv_est  # 无参形式
    def _g(x):
        return x

    assert _g(2) == 2


def test_estimator_constructs_from_cfg():
    from medtaco.utils.config import load_config

    cfg = load_config(f"{REPO}/configs/default.yaml")
    est = adv_est.TacoStagedAdvantageEstimator.from_cfg(cfg)
    assert est.zero_var_policy == "skip_group"
    assert est.std_eps == pytest.approx(1.0e-6)
    assert est.answer_tokens_mode == "paper"
    assert "taco_staged" in est.describe()


def test_rollout_cfg_from_cfg():
    from medtaco.utils.config import load_config

    cfg = load_config(f"{REPO}/configs/default.yaml")
    rc = rollout.RolloutCfg.from_cfg(cfg)
    assert rc.max_turns == 6
    assert rc.max_len == 32768
    assert rc.probe_max_tokens == 32
    assert rc.probe_temperature == 0.0
    assert rc.probe_delimiter == "</think>\n<answer>"
    assert rc.repeat_ratio == 0.5


# ------------------------------------------------------------------ 2. 假 advantage 模块


def make_fake_advantage_module():
    """按 INTERFACES.md 的签名造最小实现，供同伴模块缺席时测本适配层。"""
    mod = types.ModuleType("fake_advantage")

    class Segment:
        def __init__(self, t1, calls):
            self.t1 = t1
            self.calls = calls

    class TurnMasks:
        def __init__(self, token_mask, call_indices):
            self.token_mask = token_mask
            self.call_indices = call_indices

    mod.Segment = Segment
    mod.TurnMasks = TurnMasks
    mod.staged_deltas = lambda scores: [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]

    def ogar_masks(probe_scores, segments, mode="paper"):
        seg = segments[0]
        mask = [1] * 40
        for obs_s, obs_e, _t2_s, _t2_e in seg.calls:
            for i in range(obs_s, obs_e):
                if 0 <= i < len(mask):
                    mask[i] = 0
        return TurnMasks(token_mask=mask, call_indices=[c[0] for c in seg.calls])

    mod.ogar_masks = ogar_masks
    mod.validate_probes = lambda scores: len(scores) >= 1

    def group_advantage(values, std_eps, policy):
        arr = np.asarray(values, dtype=float)
        if np.allclose(arr, arr[0]):
            return None if policy == "skip_group" else [0.0] * len(arr)
        sd = max(float(arr.std()), std_eps)
        return list((arr - arr.mean()) / sd)

    mod.group_advantage = group_advantage
    return mod


class FakeDataProto:
    """最小 DataProto 语义：batch 是 tensor-like dict，non_tensor_batch 是 object dict。"""

    def __init__(self, bs: int, width: int, probes, segs, uids, outcome: list[float] | None = None):
        vals = outcome if outcome is not None else [1.0] * bs
        scores = np.zeros((bs, width), dtype=np.float32)
        for i, v in enumerate(vals):
            scores[i, 0] = float(v)  # 行和即结果奖励
        self.batch = {
            "responses": np.zeros((bs, width), dtype=np.int64),
            "response_mask": np.ones((bs, width), dtype=np.int64),
            "token_level_scores": scores,
        }
        self.non_tensor_batch = {
            "probe_scores": np.array(probes, dtype=object),
            "segments": np.array(segs, dtype=object),
            "uid": np.array(uids, dtype=object),
        }


def test_adv_est_calls_ogar_and_staged(monkeypatch):
    fake = make_fake_advantage_module()
    captured: dict = {}
    real_ogar = fake.ogar_masks
    real_group = fake.group_advantage

    def spy_ogar(scores, segments, mode="paper"):
        captured.setdefault("ogar_calls", []).append(list(scores))
        captured["scores"] = list(scores)
        captured["mode"] = mode
        captured["t1"] = tuple(segments[0].t1)
        captured["calls"] = list(segments[0].calls)
        return real_ogar(scores, segments, mode)

    def spy_group(values, std_eps, policy):
        captured.setdefault("groups", []).append((list(values), policy))
        return real_group(values, std_eps, policy)

    fake.ogar_masks = spy_ogar
    fake.group_advantage = spy_group
    monkeypatch.setattr(adv_est, "_ADV_MODULE", fake, raising=False)

    seg = {"t1": [0, 5], "calls": [[5, 12, 12, 20]]}
    data = FakeDataProto(
        bs=2, width=24, probes=[[0, 1], [1, 0]], segs=[seg, seg], uids=["g", "g"], outcome=[1.0, 0.0]
    )
    out = adv_est.TacoStagedAdvantageEstimator().compute_advantage(data)

    assert captured["ogar_calls"] == [[0, 1], [1, 0]]
    assert captured["mode"] == "paper"
    assert captured["t1"] == (0, 5)
    assert captured["calls"] == [(5, 12, 12, 20)]
    assert out["advantages"].shape == (2, 24)
    assert np.asarray(data.batch["loss_mask_1"]).shape == (2, 24)
    assert np.asarray(data.batch["adv_2"]).shape == (2, 24)
    assert list(data.non_tensor_batch["taco_delta_total"]) == [1, -1]
    # 两条通道各调一次 group_advantage（不是拼成一条）
    assert len(captured["groups"]) == 2
    assert all(p == "skip_group" for _v, p in captured["groups"])


def test_adv_est_masks_observation_tokens_and_gate(monkeypatch):
    fake = make_fake_advantage_module()
    monkeypatch.setattr(adv_est, "_ADV_MODULE", fake, raising=False)

    # 同一组两条轨迹：outcome 有差异（避免整组跳过），Δ_total 分别 -1 / +1
    seg = {"t1": [0, 5], "calls": [[5, 12, 12, 20]]}
    data = FakeDataProto(
        bs=2, width=24, probes=[[1, 0], [0, 1]], segs=[seg, seg], uids=["g", "g"], outcome=[1.0, 0.0]
    )
    adv_est.TacoStagedAdvantageEstimator().compute_advantage(data)

    mask1 = np.asarray(data.batch["loss_mask_1"])
    mask2 = np.asarray(data.batch["loss_mask_2"])
    adv2 = np.asarray(data.batch["adv_2"])
    assert not np.asarray(data.non_tensor_batch["taco_skip_uid"]).any()
    # 观测区间 [5,12) 恒为 0
    assert (mask1[:, 5:12] == 0).all()
    assert (mask2[:, 5:12] == 0).all()
    # Δ_total<0 的那条：轨迹门 g=0 → process 通道整体关闭
    assert (mask2[0] == 0).all() and (adv2[0] == 0).all()
    # Δ_total>0 的那条：process 通道有非零段
    assert mask2[1].sum() > 0


def test_zero_var_group_is_skipped(monkeypatch):
    fake = make_fake_advantage_module()
    monkeypatch.setattr(adv_est, "_ADV_MODULE", fake, raising=False)
    seg = {"t1": [0, 4], "calls": []}
    # 两条轨迹 outcome 与 Δ_total 都相同 → 两条通道零方差 → 整组跳过
    data = FakeDataProto(bs=2, width=16, probes=[[1, 1], [1, 1]], segs=[seg, seg], uids=["g", "g"])
    adv_est.TacoStagedAdvantageEstimator().compute_advantage(data)

    assert [bool(v) for v in data.non_tensor_batch["taco_skip_uid"]] == [True, True]
    assert not np.isnan(np.asarray(data.batch["advantages"])).any()
    assert (np.asarray(data.batch["loss_mask_1"]) == 0).all()
    assert (np.asarray(data.batch["loss_mask_2"]) == 0).all()


def test_mixed_groups_skip_only_zero_var_one(monkeypatch):
    fake = make_fake_advantage_module()
    monkeypatch.setattr(adv_est, "_ADV_MODULE", fake, raising=False)
    seg = {"t1": [0, 4], "calls": []}
    # g1 两条完全相同 → 跳过；g2 两条 outcome 与 Δ 都有差异 → 保留
    data = FakeDataProto(
        bs=4,
        width=16,
        probes=[[0, 0], [0, 0], [0, 1], [1, 0]],
        segs=[seg] * 4,
        uids=["g1", "g1", "g2", "g2"],
        outcome=[1.0, 1.0, 1.0, 0.0],
    )
    adv_est.TacoStagedAdvantageEstimator().compute_advantage(data)
    skipped = [bool(v) for v in data.non_tensor_batch["taco_skip_uid"]]
    assert skipped == [True, True, False, False]


def test_invalid_probe_trajectory_is_dropped(monkeypatch):
    fake = make_fake_advantage_module()
    fake.validate_probes = lambda scores: False
    monkeypatch.setattr(adv_est, "_ADV_MODULE", fake, raising=False)
    data = FakeDataProto(bs=1, width=8, probes=[[0, 1]], segs=[{"t1": [0, 2], "calls": []}], uids=["u"])
    adv_est.TacoStagedAdvantageEstimator().compute_advantage(data)
    assert (np.asarray(data.batch["loss_mask_1"]) == 0).all()


def test_against_real_advantage_module(monkeypatch):
    """若是同伴模块已落地，用**真实** advantage.py 再跑一遍适配层（缺席则跳过）。

    这条不替代假模块测试：假模块保证本适配层的契约，真模块保证集成不漂移。
    """
    real = pytest.importorskip("medtaco.rl.advantage")
    monkeypatch.setattr(adv_est, "_ADV_MODULE", real, raising=False)

    # t1=[0,4) + obs=[4,10) + t2=[10,16)；Δ_total = a1 - a0 = 1
    seg = {"t1": [0, 4], "calls": [[4, 10, 10, 16]]}
    data = FakeDataProto(
        bs=2,
        width=20,
        probes=[[0, 1], [1, 0]],
        segs=[seg, seg],
        uids=["g", "g"],
        outcome=[1.0, 0.0],
    )
    adv_est.TacoStagedAdvantageEstimator().compute_advantage(data)

    mask1 = np.asarray(data.batch["loss_mask_1"])
    mask2 = np.asarray(data.batch["loss_mask_2"])
    # 观测 token 不进任何通道
    assert (mask1[:, 4:10] == 0).all()
    assert (mask2[:, 4:10] == 0).all()
    # 样本 0：Δ_total=+1 → T1 拿到 outcome 信用，process 通道也有非零段
    assert mask1[0, :4].sum() == 4
    assert mask2[0].sum() > 0
    # 样本 1：Δ_total=-1 → 轨迹门 g=0，process 通道整体关闭
    assert (mask2[1] == 0).all()
    assert not np.isnan(np.asarray(data.batch["advantages"])).any()


def test_real_module_rejects_probe_count_mismatch(monkeypatch):
    """探针数 != K+1 时，真实 validate_probes 必须把它判非法并剔除。"""
    real = pytest.importorskip("medtaco.rl.advantage")
    monkeypatch.setattr(adv_est, "_ADV_MODULE", real, raising=False)

    seg = {"t1": [0, 4], "calls": [[4, 8, 8, 10]]}  # 只有 1 次调用
    # 3 个探针 → 与 K+1=2 不符
    data = FakeDataProto(bs=1, width=12, probes=[[0, 1, 1]], segs=[seg], uids=["u"], outcome=[1.0])
    adv_est.TacoStagedAdvantageEstimator().compute_advantage(data)
    assert (np.asarray(data.batch["loss_mask_1"]) == 0).all()
    assert (np.asarray(data.batch["loss_mask_2"]) == 0).all()


def test_compute_advantage_without_advantage_module_raises(monkeypatch):
    monkeypatch.setattr(adv_est, "_ADV_MODULE", None, raising=False)

    def boom(name, *a, **kw):
        if name == "medtaco.rl.advantage":
            raise ImportError("absent")
        return importlib.import_module(name)

    monkeypatch.setattr(adv_est.importlib, "import_module", boom)
    data = FakeDataProto(bs=1, width=8, probes=[[0, 1]], segs=[{"t1": [0, 2], "calls": []}], uids=["u"])
    with pytest.raises(RuntimeError, match="medtaco.rl.advantage"):
        adv_est.TacoStagedAdvantageEstimator().compute_advantage(data)


# ------------------------------------------------------------------ 3. 探针 prompt 拼接


def test_build_probe_ids_appends_delimiter_only():
    loop, _gen = make_loop([])
    prefix = [11, 22, 33]
    ids = loop.build_probe_ids(prefix)
    delim = fake_tokenize(loop.cfg.probe_delimiter)
    assert ids == prefix + delim
    assert ids[: len(prefix)] == prefix
    assert prefix == [11, 22, 33], "不得原地修改调用方的前缀"


def test_probe_uses_greedy_and_short_budget():
    loop, gen = make_loop([], probe=["no"])
    loop._probe([1, 2, 3])
    call = gen.calls[-1]
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == 32
    assert call["input_ids"] == [1, 2, 3] + fake_tokenize(loop.cfg.probe_delimiter)


def test_probe_returns_text_and_ok():
    loop, _gen = make_loop([], probe=["no"])
    text, ok = loop._probe([1])
    assert text == "no" and ok is True


# ------------------------------------------------------------------ 4. 重注入消息顺序


def test_observation_message_order_image_in_user_message():
    from medtaco.utils.schema import ToolCall, ToolResult

    loop, _gen = make_loop([])
    call = ToolCall(name="crop", args={"x0": 0, "y0": 0, "x1": 10, "y1": 10})
    res = ToolResult(ok=True, image="/tmp/crop.jpg", text="ok")
    msgs = loop.build_observation_messages(call, res)

    assert [m["role"] for m in msgs] == ["assistant", "tool", "user"]
    # tool 消息只能有文本 —— 图塞进这里模型看不见（本工程的关键坑）
    assert isinstance(msgs[1]["content"], str)
    assert any(b["type"] == "image" for b in msgs[2]["content"])


def test_observation_message_no_image_falls_back_to_text():
    from medtaco.utils.schema import ToolCall, ToolResult

    loop, _gen = make_loop([])
    msgs = loop.build_observation_messages(ToolCall(name="measure"), ToolResult(ok=True, image=None, text="32.5 mm"))
    assert [m["role"] for m in msgs] == ["assistant", "tool", "user"]
    assert all(b["type"] == "text" for b in msgs[2]["content"])
    assert "32.5 mm" in msgs[2]["content"][0]["text"]


def test_observation_messages_counted_consistently():
    """span 宽度估算与真正拼进序列的 token 数必须一致，否则 obs mask 会错位。"""
    from medtaco.utils.schema import ToolCall, ToolResult

    loop, _gen = make_loop([])
    msgs = loop.build_observation_messages(
        ToolCall(name="crop"), ToolResult(ok=True, image="/tmp/x.jpg", text="ok")
    )
    est = sum(loop.count_message_tokens(m) for m in msgs)
    real = len(loop.tokenize_observations(msgs))
    assert est == real


# ------------------------------------------------------------------ 5. 整条 rollout


def test_full_rollout_segments_and_obs_mask():
    """一条 1 次工具调用的轨迹：分段正确、观测 token 被 mask、探针数 = K+1。"""
    turn1 = 'thinking <tool_call>{"name": "crop", "args": {"x0": 0}}</tool_call>'
    turn2 = "now I see it <answer>pneumonia</answer>"
    loop, gen = make_loop([turn1, turn2], probe=["no", "yes"], scores=[0, 1], max_turns=2)
    res = loop.run_one(prompt_ids=[100, 200], sample_id="s1")

    assert res.segments["t1"][0] == 0
    assert len(res.segments["calls"]) == 1
    obs_s, obs_e, t2_s, t2_e = res.segments["calls"][0]
    assert obs_s == len(fake_tokenize(turn1))
    assert obs_e > obs_s
    assert t2_s == obs_e
    assert t2_e == obs_e + len(fake_tokenize(turn2))

    assert len(res.response_ids) == len(res.obs_mask)
    assert set(res.obs_mask[obs_s:obs_e]) == {0}
    assert all(v == 1 for i, v in enumerate(res.obs_mask) if not (obs_s <= i < obs_e))

    assert res.probe_scores == [0, 1]
    assert res.trajectory.delta_total == 1
    assert res.final_answer == "pneumonia"
    assert res.input_ids[:2] == [100, 200]
    assert len(res.input_ids) == 2 + len(res.response_ids)
    # 探针次数 = K+1 = 2，全部 greedy
    assert len(gen.probe_calls()) == 2


def test_full_rollout_two_calls():
    """两次工具调用：K=2，探针 3 次，两个 obs 区间不相交。"""
    turn1 = '<tool_call>{"name": "crop", "args": {}}</tool_call>'
    turn2 = '<tool_call>{"name": "zoom", "args": {}}</tool_call>'
    turn3 = "<answer>yes</answer>"
    loop, gen = make_loop([turn1, turn2, turn3], probe=["a", "a", "a"], scores=[0, 1, 1], max_turns=3)
    res = loop.run_one(prompt_ids=[7], sample_id="s2")
    calls = res.segments["calls"]
    assert len(calls) == 2
    assert calls[0][1] <= calls[1][0]
    assert len(res.probe_scores) == 3
    assert res.obs_mask.count(0) > 0
    assert len(gen.probe_calls()) == 3
    assert res.over_turn is False


def test_over_turn_marks_trajectory():
    """模型一直调工具、超过 max_turns：标记 over_turn（不惩罚预算耗尽）。"""
    turn = '<tool_call>{"name": "crop", "args": {}}</tool_call>'
    loop, _gen = make_loop([turn] * 4, probe=["a"] * 4, scores=[0] * 4, max_turns=1)
    res = loop.run_one(prompt_ids=[1], sample_id="s3")
    assert res.over_turn is True


def test_rollout_without_staged_probe_only_probes_k0():
    loop, gen = make_loop([], probe=["no"], scores=[1], collect_staged=False, max_turns=2)
    res = loop.run_one(prompt_ids=[1], sample_id="s4")
    assert len(gen.probe_calls()) == 1
    assert res.probe_scores == [1]


def test_repeated_suffix_ratio():
    assert rollout.TacoAgentLoop.repeated_suffix_ratio([1, 2, 3], 0.5) is False
    rep = [i % 7 for i in range(40)]
    assert rollout.TacoAgentLoop.repeated_suffix_ratio(rep * 3, 0.5) is True
    assert rollout.TacoAgentLoop.repeated_suffix_ratio(list(range(60)), 0.5) is False


def test_parse_tool_call_and_extract_answer():
    call = rollout.TacoAgentLoop.parse_tool_call(
        '<tool_call>{"name": "crop", "args": {"x0": 1}}</tool_call>'
    )
    assert call is not None and call.name == "crop" and call.args["x0"] == 1
    assert rollout.TacoAgentLoop.parse_tool_call("no tool here") is None
    assert rollout.TacoAgentLoop.extract_answer("blah <answer> hi </answer>") == "hi"
    assert rollout.TacoAgentLoop.extract_answer("nothing") == ""
