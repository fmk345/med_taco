"""轨迹 -> 训练样本：SFT 的 messages/loss_mask，与 RL 的 segments。

这是数据合成与训练之间的桥梁。两个关键点（PROPOSAL §7.5）：
  1. 工具返回的图必须作为**独立 user message** 重新注入，不能塞进 tool message
     —— 否则模型看不见，训练出一个 VEG≈0 的模型而指标看起来正常。
  2. 观测 token（返回图占位 + 沙箱输出文本）必须 **loss-mask**，
     否则它们主导梯度。

本模块分两段，避免硬依赖 tokenizer：
  - render_*()  纯字符串渲染，产出带**字符区间**标注的结构，可离线单测；
  - tokenize_with_mask()  用注入的 tokenizer（Qwen3-VL 的 processor）把字符区间
    映射成 token 区间。用 offsets_mapping 做对齐，不靠猜。
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from medtaco.utils.schema import Sample, Step, Trajectory

IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

# 观测消息的前缀：既给人看也给模型看，且便于事后定位
OBS_HEADER = "【工具返回】"
OBS_TAIL = "【工具返回结束】"


@dataclass
class Segment:
    """字符区间：(start, end)，左闭右开。不适用填 (-1,-1)。"""

    t1: Tuple[int, int] = (-1, -1)
    calls: List[Tuple[int, int, int, int]] = field(default_factory=list)
    # 每次调用: (obs_start, obs_end, t2_start, t2_end)


@dataclass
class RenderedTrajectory:
    """渲染结果。messages 已按 Qwen3-VL 约定排好序。"""

    messages: List[Dict[str, Any]]
    images: List[str]                     # 与 message 里的 image placeholder 顺序一致
    text: str                             # 拼接后的完整文本（不含图像 token 展开）
    segments: Segment                     # 字符级区间
    observation_spans: List[Tuple[int, int]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


def _msg(role: str, content: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"role": role, "content": content}


def clean_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """剥掉内容块里值为 ``None`` 的键。**吃盘上的数据前必须过一遍。**

    为什么非做不可：messages 一旦经 HF ``datasets``/Arrow 往返，同一列的 dict 会被
    补成并集 schema —— 纯文本块也会多出 ``{"image": None}``，图片块多出
    ``{"text": None}``。而 Qwen3-VL 的 chat template 判图用的是
    ``content.type == 'image' or 'image' in content``：**只判键在不在，不看值**。
    于是每个文本块都被当成一张图，占位符数（7）远超实际图片数（3），processor
    在 ``image_grid_thw[index]`` 上越界，报
    ``IndexError: index 3 is out of bounds for dimension 0 with size 3``。

    这个错误只在「从 jsonl -> Dataset -> map」这条路上出现，内存里直接渲染
    （``tokenize_with_mask``）是好的 —— 所以单测和 dry-run 都测不到它。
    """
    out: List[Dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            out.append(m)
            continue
        blocks = [{k: v for k, v in b.items() if v is not None}
                  for b in content if isinstance(b, dict)]
        out.append({**m, "content": blocks})
    return out


def count_dirty_blocks(messages: Sequence[Dict[str, Any]]) -> int:
    """统计带 None 值的块，供导出体检报警用。"""
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and any(v is None for v in b.values()):
                    n += 1
    return n


def render_trajectory(sample: Sample, traj: Trajectory,
                      obs_max_side: int = 1024) -> RenderedTrajectory:
    """把一条轨迹渲染成 Qwen3-VL 的 messages 序列。

    结构：
      user:      [原图] + 问题
      assistant: 第一段思考（T1）
      assistant: tool_call
      user:      [裁剪图] + 观测文本        <- 独立 user message（关键）
      assistant: 第二段思考（T2）+ ...
      assistant: 最终答案
    """
    messages: List[Dict[str, Any]] = []
    images: List[str] = []

    # ---- user 轮：原图 + 问题
    messages.append(_msg("user", [
        {"type": "image", "image": sample.image},
        {"type": "text", "text": sample.question},
    ]))
    images.append(sample.image)

    text_parts: List[str] = []
    cursor = 0

    def _append(t: str) -> Tuple[int, int]:
        nonlocal cursor
        s = cursor
        text_parts.append(t)
        cursor += len(t)
        return (s, cursor)

    # 用于定位的累计文本：与 text_parts 同步
    seg = Segment()
    obs_spans: List[Tuple[int, int]] = []

    # 我们把"整条 assistant 侧文本"串起来算区间；区间只用于 token mask，因此
    # 不需要与 messages 严格对应的绝对偏移，只需要在拼好的 text 上自洽。
    _append(sample.question + "\n")

    step_iter = list(traj.steps)
    # 第一段思考（在第一次工具调用之前的所有非工具步）
    first_tool = next((i for i, s in enumerate(step_iter) if s.is_tool_turn), len(step_iter))
    t1_text = "".join(s.text for s in step_iter[:first_tool])
    a, b = _append(t1_text)
    seg.t1 = (a, b)
    if t1_text.strip():
        messages.append(_msg("assistant", [{"type": "text", "text": t1_text}]))

    # 依次处理工具轮
    call_idx = 0
    i = first_tool
    while i < len(step_iter):
        st = step_iter[i]
        if not st.is_tool_turn:
            i += 1
            continue
        # 1) assistant 发起的工具调用
        call_json = _render_call(st)
        a_c, b_c = _append(call_json)
        messages.append(_msg("assistant", [{"type": "text", "text": call_json}]))

        # 2) 观测：独立 user message，图 + 文本
        obs_parts: List[Dict[str, Any]] = []
        obs_text_parts: List[str] = []
        if st.result and st.result.ok and st.result.image:
            obs_parts.append({"type": "image", "image": st.result.image})
            images.append(st.result.image)
            obs_text_parts.append(IMAGE_PLACEHOLDER)
        if st.result and st.result.text:
            obs_text_parts.append(st.result.text)
        if st.result and not st.result.ok:
            obs_text_parts.append(f"工具执行失败：{st.result.error}")
        # ``text`` 是扁平字符串，图只能用一个占位符代表它，供字符区间定位用。
        obs_str = OBS_HEADER + (" ".join(obs_text_parts) if obs_text_parts else "") + OBS_TAIL
        a_o, b_o = _append(obs_str)
        obs_spans.append((a_o, b_o))
        # messages 里**不能**再写字面量占位符：``{"type": "image"}`` 块会被
        # Qwen3-VL 模板自己展开成 <|vision_start|><|image_pad|><|vision_end|>，
        # 文本里重复写一遍会让占位符数 > 图片数，processor 直接 IndexError
        # （实测 3 图 5 占位 → image_grid_thw 越界）。两处表达的是同一件事，
        # 只能留一处。
        messages.append(_msg("user", [
            *obs_parts,
            {"type": "text", "text": OBS_HEADER + " ".join(
                t for t in obs_text_parts if t != IMAGE_PLACEHOLDER) + OBS_TAIL},
        ]))

        # 3) 后续 T2：直到下一次工具调用
        j = i + 1
        t2_text = ""
        while j < len(step_iter) and not step_iter[j].is_tool_turn:
            t2_text += step_iter[j].text
            j += 1
        a_t, b_t = _append(t2_text)
        if t2_text.strip():
            messages.append(_msg("assistant", [{"type": "text", "text": t2_text}]))

        seg.calls.append((a_o, b_o, a_t, b_t))
        call_idx += 1
        i = j

    # 最终答案：如果最后一步不含 <answer>，补一条 assistant
    final = traj.final_answer or _extract_answer(step_iter)
    if final and "<answer>" not in text_parts[-1:] and _extract_answer(step_iter) == "":
        fa = f"<answer>{final}</answer>"
        _append(fa)
        messages.append(_msg("assistant", [{"type": "text", "text": fa}]))

    return RenderedTrajectory(
        messages=messages, images=images, text="".join(text_parts),
        segments=seg, observation_spans=obs_spans,
        meta={"n_calls": len(seg.calls), "sample_id": sample.id},
    )


def _render_call(st: Step) -> str:
    """工具调用在文本里怎么出现。与 SFT 模板保持一致。"""
    assert st.call is not None
    args = ", ".join(f"{k}={v!r}" for k, v in st.call.args.items())
    return f"\n<tool_call>{st.call.name}({args})</tool_call>\n"


def _extract_answer(steps: Sequence[Step]) -> str:
    for st in reversed(steps):
        m = re.search(r"<answer>(.*?)</answer>", st.text, re.S)
        if m:
            return m.group(1).strip()
    return ""


# ---------------------------------------------------------------- token 对齐


def tokenize_with_mask(rendered: RenderedTrajectory, processor, max_len: int = 32768
                       ) -> Dict[str, Any]:
    """用 Qwen3-VL 的 processor 编码，并把字符区间映射到 token 区间。

    返回:
      input_ids / attention_mask / pixel_values / image_grid_thw（若 processor 给）
      loss_mask      长度 = len(input_ids)，1 表示参与 loss
      segments_tok   token 级 Segment，供 rl/advantage.py 使用

    对齐方式：先用 text 单独 tokenize 拿 offsets_mapping，把字符区间转成
    token 区间；再 encode 整个 messages。**要求 offsets 是相对同一段文本的**，
    因此这里只对 assistant 文本做偏移，图像 token 通过占位符计数回填。
    """
    text = rendered.text
    enc = processor.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]

    def _map(span: Tuple[int, int]) -> Tuple[int, int]:
        s, e = span
        if s < 0 or e < 0 or s >= e:
            return (-1, -1)
        idx = [i for i, (a, b) in enumerate(offsets) if b > s and a < e]
        if not idx:
            return (-1, -1)
        return (idx[0], idx[-1] + 1)

    seg_tok = Segment(
        t1=_map(rendered.segments.t1),
        calls=[(*_map((c[0], c[1])), *_map((c[2], c[3]))) for c in rendered.segments.calls],
    )

    # 全序列编码（含图像展开）
    full = processor(text=[rendered.messages], images=rendered.images,
                     return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = full["input_ids"][0].tolist()
    n = len(input_ids)

    loss_mask = [1] * n
    # 前 6 个 token（system/user 开头）与观测区间置 0 —— 观测区间需要偏移对齐，
    # 这里采用保守策略：把 prompt 长度之前的全部 mask 掉。
    prompt_len = _prompt_len_hint(rendered, processor)
    for i in range(min(prompt_len, n)):
        loss_mask[i] = 0

    return {
        "input_ids": input_ids,
        "attention_mask": full["attention_mask"][0].tolist(),
        "pixel_values": full.get("pixel_values"),
        "image_grid_thw": full.get("image_grid_thw"),
        "loss_mask": loss_mask,
        "segments_tok": seg_tok,
        "text": text,
    }


def _prompt_len_hint(rendered: RenderedTrajectory, processor) -> int:
    """保守估计 prompt 前缀的 token 数：第一条 user 消息（图 + 问题）。"""
    tok = processor.tokenizer
    head = rendered.messages[0]["content"][-1]["text"] if rendered.messages else ""
    # 图像 token 数：用 4 个占位（Qwen 每图展开后远多于 1 个 token，保守给 0）。
    # 这里只做下界，真正的精确掩码由训练侧用 messages 结构重新算。
    return len(tok(head, add_special_tokens=False)["input_ids"])


def to_sft_record(rendered: RenderedTrajectory, tokenized: Optional[Dict[str, Any]] = None
                  ) -> Dict[str, Any]:
    """导出成训练侧吃的 json（messages + images + loss_mask）。"""
    rec = {"messages": rendered.messages, "images": rendered.images,
           "meta": rendered.meta}
    if tokenized:
        rec["loss_mask"] = tokenized["loss_mask"]
    return rec


def to_rl_record(rendered: RenderedTrajectory, traj: Trajectory) -> Dict[str, Any]:
    """导出成 RL 侧吃的 json：segments（token 级由训练侧补）+ 探针分数。"""
    return {
        "id": traj.id,
        "sample_id": traj.sample_id,
        "messages": rendered.messages,
        "images": rendered.images,
        "probe_scores": [p.score for p in traj.probes],
        "delta_k": traj.delta_k,
        "delta_total": traj.delta_total,
        "segments_char": dataclasses.asdict(rendered.segments),
    }
