"""verl 多轮工具 rollout 脚手架：`TacoAgentLoop`。

⚠️ 诚实声明：**未在真实 verl 上验证**。本容器没有 verl、没有 GPU、没有真实 tokenizer，
   所以本文件是一个**参数名与数据结构都对齐 verl** 的可执行骨架：
   真实的 `generate` / `tokenize` / 工具执行全部靠注入，离线可以用假模型跑通，
   上机时必须把 `generate` 换成 verl 的 SGLang 调用（见每个 TODO 标注）。

四条必须守住的语义（对应 README 的四条）：
  1. **返回图必须作为独立 user message 重新注入**。塞进 tool message 里模型看不见——
     很多 chat template 对 role=tool 只渲染文本，image block 会被静默丢掉。
  2. **探针 = prompt_ids + tokenize(delimiter) 的二次短 decode**，temperature=0、≤32 token。
     不依赖 `continue_final_message`（verl 里不存在）。
  3. **观测 token 打标**：每次工具返回的图 / 沙箱输出，在 response 里的 token 区间记成
     obs span，两趟 loss 里都要 mask 掉。
  4. **分段**：T1 = 首轮思考，C_k = 第 k 次调用，T_{k+1} = 调用后的后续思考，
     按 schema 的 `(obs_s, obs_e, t2_s, t2_e)` 四元组落盘，交给 advantage.py 路由。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from medtaco.utils.schema import Probe, ToolCall, ToolResult, Trajectory

logger = logging.getLogger(__name__)

# 工具返回图的占位文本：真正喂给模型的是 image block，这段文字只是给人看的
OBS_IMAGE_HINT = "以下是工具返回的图像（已作为独立的用户消息重新注入，请对照回答）："
OBS_TEXT_TEMPLATE = "工具 {name} 的返回：{text}"


# ------------------------------------------------------------------ 配置


@dataclass
class RolloutCfg:
    """rollout 超参，字段名与 `configs/default.yaml` 的 `rl.*` 对齐。"""

    max_turns: int = 6
    max_len: int = 32768
    temperature: float = 1.0
    group_size: int = 8
    probe_max_tokens: int = 32
    probe_temperature: float = 0.0
    probe_delimiter: str = "</think>\n<answer>"
    collect_staged: bool = True
    # 多模态占位：一个 image block 在 token 序列里占多少 token（真实值由 processor 决定）
    image_token_count: int = 256
    # 早停：尾部子串重复覆盖超过该比例即停（Thyme / PROPOSAL §7.5）
    repeat_ratio: float = 0.5
    # 超出轮数的轨迹做 over-turn masking（不惩罚预算耗尽）
    over_turn_masking: bool = True

    @classmethod
    def from_cfg(cls, cfg: Any) -> "RolloutCfg":
        """从 medtaco Cfg 的 `rl.*` 段构造。"""
        rl = cfg.get("rl", {})
        rl = dict(rl) if isinstance(rl, dict) else {}
        probe = dict(rl.get("probe", {}) or {})
        stop = dict(rl.get("stop", {}) or {})
        return cls(
            max_turns=int(rl.get("max_turns", 6)),
            max_len=int(rl.get("max_len", 32768)),
            temperature=float(rl.get("temperature", 1.0)),
            group_size=int(rl.get("group_size", 8)),
            probe_max_tokens=int(probe.get("max_tokens", 32)),
            probe_temperature=float(probe.get("temperature", 0.0)),
            probe_delimiter=str(probe.get("delimiter", "</think>\n<answer>")),
            collect_staged=bool(probe.get("collect_staged", True)),
            repeat_ratio=float(stop.get("repeat_ratio", 0.5)),
            over_turn_masking=bool(rl.get("over_turn_masking", True)),
        )


# ------------------------------------------------------------------ 单条轨迹的记账


@dataclass
class TurnMasksBuilder:
    """一边生成 token 一边记分段。所有区间都是 **response 内** 的相对下标。

    response = 所有 assistant 生成 token + 所有注入的观测（图片 / 文本）token，
    与 `Trajectory` 的 token 编号约定一致（见 INTERFACES.md 的 CONTRACT）。
    """

    t1: Tuple[int, int] = (-1, -1)
    calls: List[Tuple[int, int, int, int]] = field(default_factory=list)
    obs_spans: List[Tuple[int, int]] = field(default_factory=list)
    _cursor: int = 0
    _pending_obs_start: int = -1
    _open_t2: bool = False   # 上一次 close_call 之后，后续 assistant token 要计入该轮 T2

    def add_assistant(self, n_tokens: int) -> Tuple[int, int]:
        start = self._cursor
        self._cursor += n_tokens
        span = (start, self._cursor)
        if self._open_t2 and self.calls:
            # 观测后的后续思考/调用 token → 扩到最近一次调用的 t2 末尾
            obs_s, obs_e, t2_s, _ = self.calls[-1]
            self.calls[-1] = (obs_s, obs_e, t2_s, self._cursor)
        elif self.t1 == (-1, -1):
            self.t1 = span  # 第一段思考 = 首轮 assistant 输出
        return span

    def add_observation(self, n_tokens: int) -> Tuple[int, int]:
        start = self._cursor
        self._cursor += n_tokens
        span = (start, self._cursor)
        self.obs_spans.append(span)
        self._pending_obs_start = start
        return span

    def close_call(self) -> None:
        """一次工具往返结束时登记 (obs_s, obs_e, t2_s, t2_e)。

        obs 区间是刚注入的观测 token；t2 先留空（`t2_s == t2_e`），
        等下一段 assistant token 到来时由 `add_assistant` 向右扩。
        """
        if self._pending_obs_start < 0:
            return
        obs_s = self._pending_obs_start
        obs_e = self.obs_spans[-1][1]
        self.calls.append((obs_s, obs_e, obs_e, obs_e))
        self._pending_obs_start = -1
        self._open_t2 = True

    @property
    def total(self) -> int:
        return self._cursor

    def as_segment(self) -> Dict[str, Any]:
        return {"t1": [int(self.t1[0]), int(self.t1[1])], "calls": [[int(v) for v in c] for c in self.calls]}

    def token_mask(self) -> List[int]:
        """1 = 参与 loss 的模型 token；观测 token 恒为 0。"""
        m = [1] * self._cursor
        for s, e in self.obs_spans:
            for i in range(max(0, s), min(e, self._cursor)):
                m[i] = 0
        return m


@dataclass
class RolloutResult:
    """一次 rollout 的产物。前 4 个字段直接落进 verl 的 DataProto。"""

    input_ids: List[int]              # prompt + response，绝对序列
    response_ids: List[int]           # 只有 response 段
    obs_mask: List[int]               # 与 response 等长，0 = 观测 token
    segments: Dict[str, Any]          # {"t1": [...], "calls": [[...]]}
    probe_scores: List[int]           # a_0..a_K
    trajectory: Optional[Trajectory] = None
    final_answer: str = ""
    over_turn: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ 主循环


class TacoAgentLoop:
    """多轮工具调用 rollout。

    真实 verl 里这个类要接到 `AgentLoopBase`（verl main 的 `verl/experimental/agent_loop`），
    本文件只给出**与框架解耦的骨架**：`generate` / `tokenize` / `run_tool` / `grade` 全部注入。

    Args:
        cfg: `RolloutCfg`，字段与 `rl.*` 对齐。
        tokenize: `(text) -> list[int]`，真实环境用 processor 的 tokenizer。
        decode: `(ids) -> str`，可选，仅用于日志。
        generate: `(input_ids, max_tokens, temperature, stop, **kw) -> (text, ids)`
            真实环境替换成 SGLang 的 `engine.generate`（见 TODO-1）。
        run_tool: `(ToolCall) -> ToolResult`，真实环境是 `medtaco.tools.ops.execute`。
        grade: `(answer_text, k) -> (score, parse_ok)`，真实环境是
            `DeterministicGrader.score` + `parse`（见 medtaco/rl/grader.py）。
    """

    def __init__(
        self,
        cfg: RolloutCfg,
        tokenize: Callable[[str], List[int]],
        generate: Callable[..., Tuple[str, List[int]]],
        run_tool: Callable[[ToolCall], ToolResult],
        grade: Callable[[str, int], Tuple[int, bool]],
        decode: Optional[Callable[[Sequence[int]], str]] = None,
    ) -> None:
        self.cfg = cfg
        self.tokenize = tokenize
        self.generate = generate
        self.run_tool = run_tool
        self.grade = grade
        self.decode = decode or (lambda ids: "")

    # -------------------------------------------------------------- 单条
    def run_one(
        self,
        prompt_ids: Sequence[int],
        sample_id: str = "",
        initial_image: Any = None,
        max_turns: Optional[int] = None,
    ) -> RolloutResult:
        """跑一条轨迹。

        `initial_image` 只用来构造原始 prompt（调用方已经拼好）；这里保留参数是为了
        在重注入时能拿到同一张图的引用。真实环境里它是 PIL.Image 或路径。
        """
        cfg = self.cfg
        limit = cfg.max_turns if max_turns is None else int(max_turns)
        base = [int(t) for t in prompt_ids]
        acc: List[int] = list(base)          # 绝对 input_ids，探针直接吃它
        resp: List[int] = []
        segs = TurnMasksBuilder()
        probes: List[Probe] = []
        steps: List[Any] = []
        img: Any = initial_image
        over_turn = False
        final_text = ""

        # ---- k=0 探针：未做任何工具调用时的答案 a_0
        a0, ok0 = self._probe(acc)
        s0, p_ok0 = self.grade(a0, 0)
        probes.append(Probe(k=0, answer=a0, score=int(s0), parse_ok=bool(p_ok0) and ok0))

        for turn in range(limit + 1):
            # ---- 1. 生成一段 assistant 文本（含 think 与可能的工具调用）
            text, ids = self.generate(
                acc,
                max_tokens=cfg.max_len - len(acc),
                temperature=cfg.temperature,
                stop=["</tool_call>", "</answer>"],
            )
            if not ids:
                break
            segs.add_assistant(len(ids))
            acc.extend(ids)
            resp.extend(ids)
            final_text = text

            if self.repeated_suffix_ratio(ids, cfg.repeat_ratio):
                logger.info("[taco] 轨迹 %s 触发重复早停", sample_id)
                break

            call = self.parse_tool_call(text)
            if call is None:
                break  # 模型给了最终答案
            if turn >= limit:
                # 超出轮数：over-turn masking，不惩罚预算耗尽
                over_turn = bool(cfg.over_turn_masking)
                break

            # ---- 2. 执行工具
            result = self.run_tool(call)
            steps.append({"turn": turn, "text": text, "call": call.to_json(), "result": result.to_json()})

            # ---- 3. 观测重注入：图作为**独立 user message**（README 第 3 条）
            obs_msgs = self.build_observation_messages(call, result)
            obs_tokens = sum(self.count_message_tokens(m) for m in obs_msgs)
            segs.add_observation(obs_tokens)
            obs_ids = self.tokenize_observations(obs_msgs)
            acc.extend(obs_ids)
            resp.extend(obs_ids)
            segs.close_call()

            # ---- 4. 第 k 次调用后的分级探针
            if cfg.collect_staged:
                ans, ok = self._probe(acc)
                score, p_ok = self.grade(ans, turn + 1)
                probes.append(Probe(k=turn + 1, answer=ans, score=int(score), parse_ok=bool(p_ok) and ok))
            # 后续轮次的"当前图"是工具返回图。
            # TODO-5（verl 对齐）：真实实现里下一轮 generate 必须把 `img` 通过
            #   `multi_modal_data={"image": [img]}` 一起交给 SGLang；只拼 input_ids
            #   而丢掉图，模型看到的是幻觉出来的像素位置。
            img = result.image or img

            if len(acc) >= cfg.max_len:
                over_turn = bool(cfg.over_turn_masking)
                break

        final = self.extract_answer(final_text) or (probes[-1].answer if probes else "")
        traj = Trajectory(
            id=f"{sample_id}#rollout",
            sample_id=sample_id,
            steps=[],  # Step 结构留给 synth 侧；rollout 只保证 token 级契约
            final_answer=final,
            probes=probes,
            delta_k=[],
            delta_total=(probes[-1].score - probes[0].score) if probes else 0,
            source="student",
            meta={"segments": segs.as_segment(), "over_turn": over_turn, "obs_spans": segs.obs_spans},
        )
        return RolloutResult(
            input_ids=acc,
            response_ids=resp,
            obs_mask=segs.token_mask(),
            segments=segs.as_segment(),
            probe_scores=[p.score for p in probes],
            trajectory=traj,
            final_answer=final,
            over_turn=over_turn,
            meta={"obs_spans": segs.obs_spans, "n_calls": len(segs.calls), "steps": steps},
        )

    # -------------------------------------------------------------- 探针
    def build_probe_ids(self, prompt_ids: Sequence[int]) -> List[int]:
        """探针的输入 = 当前完整前缀 + tokenize(delimiter)。

        ⚠️ 这里 **不** 用 `continue_final_message`（verl 不支持），而是把分隔符当成新的
        用户/续写 token 拼进去，再让 SGLang 以 temperature=0 直出 ≤32 token。
        """
        return [int(t) for t in prompt_ids] + [int(t) for t in self.tokenize(self.cfg.probe_delimiter)]

    def _probe(self, prompt_ids: Sequence[int]) -> Tuple[str, bool]:
        """返回 (答案文本, 是否成功 decode)。"""
        probe_ids = self.build_probe_ids(prompt_ids)
        try:
            text, ids = self.generate(
                probe_ids,
                max_tokens=self.cfg.probe_max_tokens,  # ≤32，防止探针成本与方差失控
                temperature=self.cfg.probe_temperature,  # 0.0，greedy
                stop=["</answer>"],
                probe=True,
            )
        except TypeError:
            # 假 generate / 老接口不支持 probe 关键字
            text, ids = self.generate(probe_ids, max_tokens=self.cfg.probe_max_tokens,
                                      temperature=self.cfg.probe_temperature)
        if len(ids) > self.cfg.probe_max_tokens:
            logger.warning("[taco] 探针输出 %d token 超上限 %d", len(ids), self.cfg.probe_max_tokens)
            ids = ids[: self.cfg.probe_max_tokens]
        return (text or self.decode(ids)), bool(text or ids)

    # -------------------------------------------------------------- 消息构造
    def build_observation_messages(self, call: ToolCall, result: ToolResult) -> List[Dict[str, Any]]:
        """工具返回后要追加的消息序列。**顺序在这里定死**，测试会断言它。

        1) assistant：带 tool_call 的字段（回显模型的选择，保证模板配对）
        2) tool：只有文本观测。**不能放图**——放进去模型看不见。
        3) user：独立消息，含 image block（若有返回图）+ 一行说明。
        """
        # TODO-2（verl 对齐）：真实 verl 的 SGLang 后端要求 image block 通过
        #   `multi_modal_data` 传，而不是塞在 messages 里。这里是 messages 语义的骨架，
        #   上机时保留**顺序**与**独立 user 消息**这两点，把 image 换成
        #   {"type": "image", "image": <PIL.Image|url>} 并按 sglang 的格式喂。
        msgs: List[Dict[str, Any]] = [
            {"role": "assistant", "tool_calls": [{"name": call.name, "arguments": call.args}]},
        ]
        if result.ok:
            msgs.append({"role": "tool", "name": call.name, "content": result.text})
        else:
            msgs.append({"role": "tool", "name": call.name, "content": f"ERROR: {result.error}"})

        content: List[Dict[str, Any]] = []
        if result.image:
            content.append({"type": "image", "image": result.image})
        content.append({"type": "text", "text": OBS_IMAGE_HINT if result.image else
                        OBS_TEXT_TEMPLATE.format(name=call.name, text=result.text)})
        msgs.append({"role": "user", "content": content})
        return msgs

    def count_message_tokens(self, msg: Dict[str, Any]) -> int:
        """估算一条注入消息占多少 token（决定 obs span 宽度）。"""
        content = msg.get("content", "")
        if isinstance(content, str):
            return len(self.tokenize(content))
        n = 0
        for block in content:
            if block.get("type") == "image":
                n += self.cfg.image_token_count
            else:
                n += len(self.tokenize(str(block.get("text", ""))))
        return n

    def tokenize_observations(self, msgs: Sequence[Dict[str, Any]]) -> List[int]:
        """把注入消息编成 token。真实环境应走 processor 的 chat template。

        TODO-3（verl 对齐）：真实实现里返回的是 `multi_modal_data` +
        `input_ids` 的组合，图 token 由 vision tower 动态决定；这里的拼接只保证
        **长度一致**（与 count_message_tokens 的估算口径一致），用于离线验证分段逻辑。
        """
        out: List[int] = []
        for m in msgs:
            content = m.get("content", "")
            if isinstance(content, str):
                out.extend(self.tokenize(content))
                continue
            for block in content:
                if block.get("type") == "image":
                    out.extend([0] * self.cfg.image_token_count)  # 占位 image token
                else:
                    out.extend(self.tokenize(str(block.get("text", ""))))
        return out

    # -------------------------------------------------------------- 早停 / 解析
    @staticmethod
    def repeated_suffix_ratio(ids: Sequence[int], threshold: float) -> bool:
        """Rabin-Karp 近似的尾部重复检测：尾部子串覆盖比例 > threshold 即停。

        只看尾部：从 `len//2` 往下找第一个"尾段在前文出现过至少两次"的长度。
        """
        n = len(ids)
        if n < 16:
            return False
        half = n // 2
        for L in range(half, 7, -1):  # L < 8 的重复没有意义（模板词常见）
            tail = list(ids[n - L :])
            cnt = 0
            for i in range(0, n - L + 1):
                if list(ids[i : i + L]) == tail:
                    cnt += 1
            if cnt >= 2 and (cnt * L) / max(1, n) > threshold:
                return True
        return False

    @staticmethod
    def extract_answer(text: str) -> str:
        """从最终 assistant 文本里取 `<answer>..</answer>`。真实环境用 grader.parse。"""
        if "<answer>" not in text:
            return ""
        body = text.split("<answer>", 1)[1]
        return body.split("</answer>", 1)[0].strip() if "</answer>" in body else body.strip()

    @staticmethod
    def parse_tool_call(text: str) -> Optional[ToolCall]:
        """从模型输出里抽工具调用。骨架只认一个极简 JSON 片段，真实环境用 Qwen 的模板解析。

        TODO-4（verl 对齐）：换成 verl `tool_agent_loop` 的 parser，或 medtaco 自己的
        `<tool_call>{...}</tool_call>` 解析器；这里保留最小实现是为了可单测。
        """
        if "<tool_call>" not in text:
            return None
        import json

        body = text.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
        try:
            d = json.loads(body)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(d, dict) or "name" not in d:
            return None
        return ToolCall(name=str(d["name"]), args=dict(d.get("args", {})))

    # -------------------------------------------------------------- 批量
    def run_group(self, prompts: Sequence[Tuple[str, Sequence[int], Any]]) -> List[RolloutResult]:
        """一组 rollout（G=8）。真实 verl 里由 worker 并发调度，这里串行。"""
        return [self.run_one(p, sample_id=sid, initial_image=img) for sid, p, img in prompts]


__all__ = ["RolloutCfg", "RolloutResult", "TurnMasksBuilder", "TacoAgentLoop"]
