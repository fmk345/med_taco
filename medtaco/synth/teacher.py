"""Claude Opus 5 轨迹合成（PROPOSAL §5）。

设计要点
--------
1. **一条请求 = 一条完整轨迹**（§5.1）。messages 里塞 3 个 image block
   （原图 + 2 个裁剪）+ 一段文本指令，输出用 ``output_config.format=json_schema``。
   **禁止 message prefill**：该代模型上 prefill 直接 400，JSON 只能靠 schema 约束。
2. **采样参数**：claude-opus-5 已移除 ``temperature``/``top_p``/``top_k``，
   传了就是 400。配置里的 ``teacher.temperature`` 只在旧模型上生效，
   本模块用 ``_sampling_params()`` 显式过滤，避免把 400 带进线上。
3. **视觉 token**：``ceil(w/28)·ceil(h/28)``，单图上限 4784 token，
   长边上限 ``teacher.max_side_px``（2576）。超了按比例缩（§5.1）。
4. **审计诚实点**：一条请求里 3 张图是**同时**给模型的，所以模型写下的
   ``probe_no_tool``（a_1 探针）在构造上就已经看过裁剪了——这是本形态的固有污染。
   指令层用 [C1]/[C3] 压制「先猜答案再补装饰性裁剪」，但**真正的判据在
   synth/validate.py 的三项构造性验证**（尤其 no_tool_ablation）。因此：
   - ``Trajectory.probes`` 只放**占位**探针，``meta['probes_pending']=True``，
     必须由 validate 阶段重新采集后再用于 OGAR；
   - ``Step.result.meta['executed']=False``：教师自述的执行结果不可信，
     必须由 tools/ops 在沙箱里重放（§5.3 第 1 项）。
5. **Batch API**：create → poll(retrieve) → results，50% 折扣；单 batch
   ≤100k 请求 / ≤256MB，按 custom_id 幂等（已成功的绝不重复提交）。
"""
from __future__ import annotations

import base64
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..utils.schema import Box, Probe, Sample, Step, ToolCall, ToolResult, Trajectory

# ---------------------------------------------------------------- 常量

#: 视觉 token 的 patch 边长（§5.1）
PATCH_PX: int = 28
#: 单图视觉 token 上限（§5.1）
MAX_VISION_TOKENS: int = 4784
#: 结构化输出允许的 answer_type 取值
ANSWER_TYPES: Tuple[str, ...] = ("closed", "short", "measure", "long")
#: 教师可调用的工具（与 tools/ops.execute 对齐）
TOOL_NAMES: Tuple[str, ...] = ("crop", "zoom", "window", "measure", "none")
#: 轨迹固定 3 个 step（3 loop 轮）
N_STEPS: int = 3
#: 该代模型已移除采样参数，传了返回 400
NO_SAMPLING_MODELS: Tuple[str, ...] = (
    "claude-opus-5",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-5",
    "claude-fable-5",
)
#: 指令里必须出现的约束标签，单测逐条断言（反 bypassing 的指令层约束）
CONSTRAINT_IDS: Tuple[str, ...] = ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")
#: 配置缺省值（configs/default.yaml 里的同名字段优先）
TEACHER_DEFAULTS: Dict[str, Any] = {
    "provider": "anthropic",
    "model": "claude-opus-5",
    "price_in_per_mtok": 5.0,
    "price_out_per_mtok": 25.0,
    "batch_discount": 0.5,
    "use_batch": True,
    "batch_max_requests": 100000,
    "batch_max_bytes": 268435456,
    "max_images_per_request": 20,
    "max_side_px": 2576,
    "max_tokens": 8192,
    "temperature": 0.7,
    "max_retries": 4,
    "concurrency": 8,
    "observation_max_side": 1024,
    "jpeg_quality": 90,
    "use_files_api": False,
    "poll_interval_s": 10.0,
    "max_wait_s": 86400.0,
}
#: 成本估算的缺省 token 画像（PROPOSAL §5.2：3 图块 / ~4.5k 输入 / ~4.2k 输出）
EST_INPUT_TOKENS: int = 4500
EST_OUTPUT_TOKENS: int = 4200

_MEDIA_TYPES: Dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


class TeacherError(RuntimeError):
    """教师合成链路的可预期失败（构造请求 / 解析响应 / 批处理重试耗尽）。"""


# ---------------------------------------------------------------- 配置读取


def _tcfg(cfg: Any) -> Any:
    """取 teacher 子配置；既接受整份 cfg，也接受已经取过 cfg.teacher 的子字典。"""
    if cfg is None:
        return dict(TEACHER_DEFAULTS)
    if isinstance(cfg, dict) and isinstance(cfg.get("teacher"), dict):
        return cfg["teacher"]
    return cfg


def _opt(cfg: Any, key: str, default: Any = None) -> Any:
    """读 teacher 配置项，缺省回落到 TEACHER_DEFAULTS。"""
    t = _tcfg(cfg)
    if t is None:
        return TEACHER_DEFAULTS.get(key, default)
    v = t.get(key, None)
    if v is None:
        return TEACHER_DEFAULTS.get(key, default)
    return v


# ---------------------------------------------------------------- 视觉 token


def visual_tokens(w: int, h: int) -> int:
    """视觉 token 数：ceil(w/28) · ceil(h/28)（§5.1）。"""
    if w <= 0 or h <= 0:
        raise ValueError(f"非法尺寸: {w}x{h}")
    return -(-int(w) // PATCH_PX) * -(-int(h) // PATCH_PX)


def fit_size(w: int, h: int, max_side_px: int = 2576,
             max_tokens: int = MAX_VISION_TOKENS) -> Tuple[int, int, int]:
    """把 (w,h) 缩到长边 ≤max_side_px 且视觉 token ≤max_tokens。

    返回 ``(新宽, 新高, 新视觉 token)``；不需要缩时原样返回。
    """
    nw, nh = int(w), int(h)
    if max(nw, nh) > max_side_px:
        s = max_side_px / float(max(nw, nh))
        nw, nh = max(PATCH_PX, int(nw * s)), max(PATCH_PX, int(nh * s))
    # 长边达标后按面积继续收缩；ceil 会放大，所以留一点余量并迭代
    for _ in range(64):
        tok = visual_tokens(nw, nh)
        if tok <= max_tokens:
            return nw, nh, tok
        s = math.sqrt(max_tokens / float(tok)) * 0.995
        nw = max(PATCH_PX, int(nw * s))
        nh = max(PATCH_PX, int(nh * s))
    raise TeacherError(f"无法把 {w}x{h} 缩到 {max_tokens} 视觉 token 以内")


def approx_text_tokens(text: str) -> float:
    """粗略文本 token 估算：CJK 约 1.6 字/token，其余约 3.8 字符/token。

    只用于 ``build_request`` 的 ``input_tokens`` 估算与成本预算，
    **不参与任何判分**；真实计量以 API 返回的 ``usage`` 为准。
    """
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿" or "぀" <= ch <= "ヿ")
    other = len(text) - cjk
    return cjk / 1.6 + other / 3.8


# ---------------------------------------------------------------- 图像块


def _media_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext not in _MEDIA_TYPES:
        raise TeacherError(f"不支持的图片后缀: {path}（只认 {sorted(_MEDIA_TYPES)}）")
    return _MEDIA_TYPES[ext]


def _read_image(path: str) -> Tuple[bytes, int, int]:
    """读图字节 + (w,h)。只用 PIL 读头部拿尺寸，不做解码重编码。"""
    if not os.path.exists(path):
        raise TeacherError(f"图片不存在: {path}")
    try:
        from PIL import Image
    except ImportError as e:  # pragma: no cover - 环境里必然有 PIL
        raise TeacherError("需要 Pillow 读取图片尺寸: pip install pillow") from e
    with Image.open(path) as im:
        w, h = im.size
    with open(path, "rb") as f:
        raw = f.read()
    if not raw:
        raise TeacherError(f"图片是空文件: {path}")
    return raw, int(w), int(h)


def _image_block(raw: bytes, media_type: str, file_id: Optional[str] = None) -> Dict[str, Any]:
    """构造一个 image content block；有 Files API 的 file_id 就走 file，否则 base64。"""
    if file_id:
        return {"type": "image", "source": {"type": "file", "file_id": file_id}}
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.b64encode(raw).decode("ascii"),
        },
    }


def crop_paths_of(sample: Sample) -> List[str]:
    """从 Sample.meta 里取裁剪图路径（由 tools/ops 现场生成后写回 meta）。

    兼容两种写法：``meta['crop_paths'] = [p1, p2]`` 或
    ``meta['crops'] = [{'path': p1, 'box': [...], 'tool': 'crop'}, ...]``。
    """
    meta = sample.meta or {}
    raw = meta.get("crop_paths")
    if raw is None:
        raw = meta.get("crops")
    out: List[str] = []
    for item in raw or []:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            p = item.get("path") or item.get("image") or item.get("file")
            if isinstance(p, str):
                out.append(p)
    return out


# ---------------------------------------------------------------- 输出 schema


def _nullable(t: str) -> Dict[str, Any]:
    """可空数值/字符串。structured outputs 支持 anyOf。"""
    return {"anyOf": [{"type": t}, {"type": "null"}]}


def _obj(props: Dict[str, Any], desc: str = "") -> Dict[str, Any]:
    """拍平的对象：全字段 required + additionalProperties=false（§5.1）。"""
    d: Dict[str, Any] = {
        "type": "object",
        "properties": props,
        "required": list(props.keys()),
        "additionalProperties": False,
    }
    if desc:
        d["description"] = desc
    return d


def build_schema(cfg: Any = None) -> Dict[str, Any]:
    """轨迹的 json_schema：扁平三层（probe / steps[3] / final）。"""
    box_fields = {
        "tool_x0": _nullable("number"),
        "tool_y0": _nullable("number"),
        "tool_x1": _nullable("number"),
        "tool_y1": _nullable("number"),
    }
    step_props: Dict[str, Any] = {
        "think": {
            "type": "string",
            "description": "本轮的推理：先写假设，再写上一轮观测确认或推翻了什么。",
        },
        "tool_name": {
            "type": "string",
            "enum": list(TOOL_NAMES),
            "description": "本轮调用的工具；不调用填 none。",
        },
        # 拍平的多工具参数：按 tool_name 解释，其余留 null
        "tool_out_side": _nullable("integer"),
        "tool_center": _nullable("number"),
        "tool_width": _nullable("number"),
        "tool_mode": _nullable("string"),
        "result_ok": {"type": "boolean", "description": "该调用是否产生了可用观测。"},
        "result_text": {"type": "string", "description": "观测的文本部分（measure 读数等）。"},
        "result_image_ref": {
            "type": "string",
            "description": "该调用对应的返回图标签，如 CROP_1；无图填空串。",
        },
    }
    step_props.update(box_fields)
    schema = _obj(
        {
            "probe_no_tool": _obj(
                {
                    "answer": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "rationale": {"type": "string"},
                },
                desc="[C1] 只看原图、不做任何工具调用时的初始判断（a_1 探针）。",
            ),
            "steps": {
                "type": "array",
                "minItems": N_STEPS,
                "maxItems": N_STEPS,
                "items": _obj(step_props, desc="一轮：think + 可选工具调用 + 观测。"),
            },
            "final_answer": {
                "type": "string",
                "description": "[C7] 可直接被确定性判分器匹配的短答案，不要含糊措辞。",
            },
            "answer_type": {"type": "string", "enum": list(ANSWER_TYPES)},
            "final_rationale": {"type": "string"},
        },
        desc="一条完整的工具增强医学影像问答轨迹。",
    )
    return schema


# ---------------------------------------------------------------- 指令


def build_instruction(sample: Sample, roles: Sequence[str]) -> str:
    """拼装文本指令。反 bypassing 的约束在 [C1]/[C3] 两条上。"""
    role_lines = "\n".join(f"  IMAGE {i + 1} = {r}" for i, r in enumerate(roles))
    crops = [r for r in roles if r.startswith("CROP_")]
    crop_note = (
        f"本轮提供了 {len(crops)} 张裁剪（{', '.join(crops)}），它们是**已经执行过**的"
        "工具调用的返回图。"
        if crops
        else "本轮没有提供裁剪图，所有 step 的 tool_name 应填 none。"
    )
    return f"""You are a senior radiologist reconstructing a supervised tool-use trajectory.

Images attached, in order:
{role_lines}

{crop_note}

Question: {sample.question}
{_answer_type_hint(sample)}

HARD CONSTRAINTS — violating any of them makes the trajectory useless for training:

[C1] FIRST, before you consider CROP_1 / CROP_2 in any way, fill `probe_no_tool` from the
     ORIGINAL image and the question alone: `answer`, `confidence` in [0,1], one-sentence
     `rationale`. This is the a_1 staged probe; it must reflect what you can conclude
     with zero tool calls. Write it as if the crops were not attached.

[C2] Only then decide, per step, whether a tool call was warranted. Every step whose
     `tool_name` is crop / zoom / window / measure must justify itself: `think` states the
     hypothesis FIRST, then the region follows from that hypothesis.

[C3] NO answer-first trajectories. Do not decide the final answer first and then attach a
     crop that merely decorates the reasoning. If your final answer is unchanged from your
     initial no-tool judgement and the crops added nothing, set every `tool_name` to "none"
     and say exactly that in `final_rationale`. A decorative crop is worse than no crop —
     we will detect and discard it.

[C4] Exactly {N_STEPS} steps, in chronological order. At most one tool call per step.

[C5] Each `think` must contain: (a) the current hypothesis, (b) what the previous
     observation changed or confirmed. No empty or restated-only reasoning.

[C6] `tool_x0/tool_y0/tool_x1/tool_y1` are pixel coordinates in the ORIGINAL image
     (origin top-left) delimiting the region shown in the corresponding CROP. Use null
     for window (it takes center/width) and for every field when `tool_name` is "none".
     Never emit a box wider or taller than the original image.

[C7] `final_answer` must be a single short span a deterministic grader can match:
     no hedging ("likely", "可能"), no units twice, no full sentence for closed questions.

[C8] Medical safety: report only findings visible in the supplied images. If the crops are
     not diagnostic, state that in `final_rationale` instead of inventing a finding.

`result_ok` / `result_text` / `result_image_ref` describe what you actually observe in the
supplied crop for that step (you can see it); they are recorded as *planned* observations
and will be re-executed in a sandbox afterwards."""


def _sampling_params(model: str, cfg: Any) -> Dict[str, Any]:
    """claude-opus-5 一类模型已移除 temperature/top_p/top_k，传了 400。"""
    if any(model.startswith(m) for m in NO_SAMPLING_MODELS):
        return {}
    temp = _opt(cfg, "temperature", 0.7)
    return {"temperature": float(temp)} if temp is not None else {}


# ---------------------------------------------------------------- 请求构造

#: 把 answer_type 翻译成"答案长什么样"的自然语言，而不是把枚举值摆进 prompt。
#: 直接写 "Answer type: closed (closed | short | measure | long)" 时，8B 教师有
#: 约三成概率把这个枚举字面量当成答案抄进 probe/final。
_AT_HINT: Dict[str, str] = {
    "closed": "closed question — answer with one of the offered options verbatim (e.g. yes / no).",
    "short": "short answer — a few words or a single phrase, no full sentence.",
    "measure": "measurement — a number with unit (e.g. 12.3 mm).",
    "long": "long answer — one to three sentences of clinical description.",
}


def _answer_type_hint(sample: Sample) -> str:
    at = str(getattr(sample, "answer_type", "") or "").strip().lower()
    hint = _AT_HINT.get(at) or "free-form answer; keep it short and judgeable."
    return f"Expected answer form: {hint}"



def build_request(sample: Sample, cfg: Any, file_ids: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """一条轨迹一次请求。

    返回::

        {"custom_id": ..., "params": {...}, "input_tokens": int,
         "n_images": int, "roles": [...], "resized": [...]}

    ``params`` 可直接丢给 ``client.messages.create(**params)``，
    也可作为 Batch API 的 ``Request.params``。
    """
    model = str(_opt(cfg, "model", "claude-opus-5"))
    max_tokens = int(_opt(cfg, "max_tokens", 8192))
    max_side = int(_opt(cfg, "max_side_px", 2576))
    max_images = int(_opt(cfg, "max_images_per_request", 20))

    paths = crop_paths_of(sample)
    if not paths:
        # 没有裁剪：退化成「原图 + 原图 + 原图」，让模型把所有 step 填 none。
        # 这样批次里不会出现结构不一致的请求。
        if not os.path.exists(sample.image):
            raise TeacherError(f"原图不存在: {sample.image}")
        paths = [sample.image, sample.image]

    roles: List[str] = ["ORIGINAL"] + [f"CROP_{i + 1}" for i in range(len(paths))]
    if len(roles) > max_images:
        raise TeacherError(
            f"单请求图片数 {len(roles)} 超过上限 {max_images}（§5.1：>20 张会触发更严的 2000px 限制）"
        )

    blocks: List[Dict[str, Any]] = []
    vision_tokens = 0
    resized: List[Dict[str, Any]] = []
    all_paths = [sample.image] + list(paths)
    for path in all_paths:
        if not os.path.exists(path):
            raise TeacherError(f"图片不存在: {path}")
        media = _media_type(path)  # 先校验后缀，别让 PIL 先报看不懂的错
        raw, w, h = _read_image(path)
        nw, nh, tok = fit_size(w, h, max_side, MAX_VISION_TOKENS)
        vision_tokens += tok
        if (nw, nh) != (w, h):
            resized.append({"path": path, "from": [w, h], "to": [nw, nh]})
        fid = (file_ids or {}).get(path)
        blocks.append(_image_block(raw, media, fid))

    text = build_instruction(sample, roles)
    blocks.append({"type": "text", "text": text})

    instruction_tokens = int(math.ceil(approx_text_tokens(text))) + 4 * len(blocks)
    input_tokens = vision_tokens + instruction_tokens

    params: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": blocks}],  # 只有 user：禁止 prefill
        "output_config": {"format": {"type": "json_schema", "schema": build_schema(cfg)}},
    }
    params.update(_sampling_params(model, cfg))

    return {
        "custom_id": custom_id_of(sample),
        "params": params,
        "input_tokens": int(input_tokens),
        "vision_tokens": int(vision_tokens),
        "n_images": len(blocks) - 1,
        "roles": roles,
        "resized": resized,
    }


def custom_id_of(sample: Sample) -> str:
    """Batch 的 custom_id。稳定、可逆，且只含 API 允许的字符。"""
    sid = str(sample.id).strip().replace(" ", "_").replace("/", "_")
    if not sid:
        raise TeacherError("sample.id 为空，无法生成 custom_id")
    return f"traj-{sid}"


# ---------------------------------------------------------------- 成本


def estimate_cost(n_samples: int, cfg: Any, calls_per_traj: float = 1.0) -> Dict[str, Any]:
    """按 cfg 单价与 batch 折扣估算成本，给出实时 / 批量两条路径。

    ``calls_per_traj``：每条轨迹折合几次请求（默认 1.0 = 一条请求一条轨迹，§5.1）。
    额外补一次「只看原图的探针调用」时传 2.0。
    """
    if n_samples < 0:
        raise ValueError("n_samples 不能为负")
    if calls_per_traj <= 0:
        raise ValueError("calls_per_traj 必须为正")

    p_in = float(_opt(cfg, "price_in_per_mtok", 5.0))
    p_out = float(_opt(cfg, "price_out_per_mtok", 25.0))
    disc = float(_opt(cfg, "batch_discount", 0.5))
    in_tok = int(_opt(cfg, "est_input_tokens", EST_INPUT_TOKENS))
    out_tok = int(_opt(cfg, "est_output_tokens", EST_OUTPUT_TOKENS))

    per_call = in_tok / 1e6 * p_in + out_tok / 1e6 * p_out
    n_calls = n_samples * calls_per_traj
    realtime = per_call * n_calls
    batch = realtime * disc

    return {
        "n_samples": int(n_samples),
        "calls_per_traj": float(calls_per_traj),
        "n_calls": float(n_calls),
        "input_tokens_per_call": in_tok,
        "output_tokens_per_call": out_tok,
        "price_in_per_mtok": p_in,
        "price_out_per_mtok": p_out,
        "batch_discount": disc,
        "per_traj_realtime_usd": per_call * calls_per_traj,
        "per_traj_batch_usd": per_call * calls_per_traj * disc,
        "realtime_usd": realtime,
        "batch_usd": batch,
        "assumptions": {
            "images_per_request": 3,
            "vision_token_formula": "ceil(w/28)*ceil(h/28), cap 4784/image",
            "note": "输入 4.5k / 输出 4.2k 来自 PROPOSAL §5.2 的画像；以实际账单为准。",
        },
    }


# ---------------------------------------------------------------- 响应解析


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """同时支持 dict 与 SDK 的 pydantic 对象。"""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _unwrap_message(resp: Any) -> Any:
    """把 Batch 结果 / Message 归一成 message 对象。"""
    if resp is None:
        raise TeacherError("响应为空")
    # Batch: {"custom_id":..., "result": {"type":"succeeded","message":{...}}}
    if isinstance(resp, dict) and "result" in resp and _get(resp["result"], "message") is not None:
        return _unwrap_message(resp["result"])
    if isinstance(resp, dict) and "message" in resp and "content" not in resp:
        return resp["message"]
    return resp


def _extract_json(resp: Any) -> Dict[str, Any]:
    """从 message / batch 结果 / 已解码 payload 里取结构化输出。"""
    if isinstance(resp, str):
        try:
            return json.loads(resp)
        except json.JSONDecodeError as e:
            raise TeacherError(f"响应不是合法 JSON: {e}") from e

    msg = _unwrap_message(resp)

    # 已经解码好的 payload（单测与重放走这条）
    if isinstance(msg, dict) and "steps" in msg and "content" not in msg:
        return msg
    if not isinstance(msg, dict) and hasattr(msg, "steps") and not hasattr(msg, "content"):
        return {k: getattr(msg, k) for k in ("probe_no_tool", "steps", "final_answer", "answer_type", "final_rationale")}

    stop_reason = _get(msg, "stop_reason")
    if stop_reason == "refusal":
        cat = _get(_get(msg, "stop_details"), "category")
        raise TeacherError(f"模型拒答（category={cat}），该样本需换问题或丢弃")

    content = _get(msg, "content")
    if content is None:
        raise TeacherError(f"响应里没有 content 字段: {type(msg).__name__}")

    texts: List[str] = []
    for b in content:
        if _get(b, "type") == "text":
            t = _get(b, "text") or ""
            if t:
                texts.append(t)
    if not texts:
        raise TeacherError("响应里没有 text block（structured outputs 应返回一段 JSON）")
    raw = "".join(texts).strip()
    # 兜底：个别情况下会被包进 ```json 代码块
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise TeacherError(f"结构化输出不是合法 JSON: {e}; 前 200 字: {raw[:200]}") from e
    if not isinstance(payload, dict) or "steps" not in payload:
        raise TeacherError("结构化输出缺少 steps 字段")
    return payload


def _step_box(step: Dict[str, Any]) -> Optional[Box]:
    vals = [step.get(k) for k in ("tool_x0", "tool_y0", "tool_x1", "tool_y1")]
    if all(v is None for v in vals):
        return None
    if any(v is None for v in vals):
        raise TeacherError(f"框坐标不完整（部分为 null）: {vals}")
    try:
        return Box(float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))
    except (TypeError, ValueError) as e:
        raise TeacherError(f"框坐标非法: {vals}") from e


def _call_args(name: str, step: Dict[str, Any], box: Optional[Box]) -> Dict[str, Any]:
    """按工具名把拍平的字段还原成 ToolCall.args；None 字段全丢。"""
    if name in ("crop", "zoom"):
        if box is None:
            raise TeacherError(f"{name} 必须给框")
        args: Dict[str, Any] = {
            "x0": box.x0, "y0": box.y0, "x1": box.x1, "y1": box.y1,
        }
        if name == "zoom" and step.get("tool_out_side") is not None:
            args["out_side"] = int(step["tool_out_side"])
        return args
    if name == "window":
        if step.get("tool_center") is None or step.get("tool_width") is None:
            raise TeacherError("window 必须给 center/width")
        return {"center": float(step["tool_center"]), "width": float(step["tool_width"])}
    if name == "measure":
        if box is None:
            raise TeacherError("measure 必须给框")
        out: Dict[str, Any] = {"x0": box.x0, "y0": box.y0, "x1": box.x1, "y1": box.y1}
        if step.get("tool_mode"):
            out["mode"] = str(step["tool_mode"])
        return out
    raise TeacherError(f"未知工具名: {name}")


def _crop_path_for(sample: Sample, k: int) -> Optional[str]:
    """第 k 次**裁剪类调用**对应的裁剪图路径（不是第 k 个 step）。"""
    paths = crop_paths_of(sample)
    return paths[k] if 0 <= k < len(paths) else None


def crop_paths_on_disk(sample: Sample, crops_dir: str) -> List[str]:
    """按 ``<safe(sample.id)>_<k>.png`` 的命名从盘上找回裁剪图。

    为什么需要这条：``prepare_crops`` 只把路径写进**内存里**的 ``sample.meta``，
    而 ``questions.jsonl`` 是它之前落盘的，读回来时 ``meta['crops']`` 根本不存在。
    于是下游按 meta 取路径必然为空 —— 一旦据此"修正"轨迹，就会把真实存在的
    裁剪图全抹成 None（实测把 60 条里 27 条的观测图抹掉了）。
    """
    if not crops_dir or not os.path.isdir(crops_dir):
        return []
    from .crops import _safe

    out: List[str] = []
    for k in range(8):  # 单样本最多两张，8 是防御上限
        p = os.path.join(crops_dir, f"{_safe(sample.id)}_{k}.png")
        if not os.path.exists(p):
            break
        out.append(p)
    return out


def _ref_as_path(ref: str) -> Optional[str]:
    """``result_image_ref`` 只有在真的是现存文件时才当图用。

    教师返回的是**标签**（``CROP_1``），不是路径。老代码把它兜底塞进
    ``ToolResult.image``，渲染时这一串就变成 messages 里的 image 块，
    导出体检立刻报 ``missing_img_file``（实测 60 条里 15 条中招）。
    """
    if not ref or not os.path.exists(ref):
        return None
    return ref


def sanitize_images(traj: Trajectory, sample: Sample,
                    crops_dir: Optional[str] = None) -> int:
    """按「第几次裁剪调用」重排 step.result.image，返回改动的条数。

    老数据里的 ``CROP_1``/``CROP_2`` 已经落盘，重跑合成不现实，所以在读侧
    （validate / export）统一修一遍。这条是幂等的：已经正确的轨迹改 0 处。
    非裁剪类调用（measure 无图、window 的图要由沙箱重放给出）一律置空。

    路径来源优先 ``sample.meta``，缺了再用 ``crops_dir`` 按命名回查。
    **该有裁剪却查不到图时保持原值不动**：把真路径抹成 None 比留个坏标签更糟——
    坏标签在导出体检里看得见，None 只会让观测轮静默地少一张图。反过来，
    本就不该有裁剪的样本（G3 负样本）被教师裁了一刀，就只能置空。
    """
    paths = crop_paths_of(sample)
    if not paths and crops_dir:
        paths = crop_paths_on_disk(sample, crops_dir)
    if not paths:
        from .crops import crop_boxes_for

        if crop_boxes_for(sample):
            # 应有裁剪却找不到图：别动，留给导出体检暴露
            return 0
        # 本就不该有裁剪（G3 负样本），教师却调了 crop：轨迹里没有图可指，
        # 保留 'CROP_1' 只会在导出时变成一个不存在的图片路径。
        # 这类轨迹会被 validate 判 unexpected_crop，这里只保证渲染不乱。

    n = 0
    crop_idx = 0
    for st in traj.steps:
        if not st.is_tool_turn or st.call is None:
            continue
        name = (st.call.name or "").lower()
        if name in ("crop", "zoom"):
            want = paths[crop_idx] if crop_idx < len(paths) else None
            crop_idx += 1
        else:
            want = None
        if st.result is None:
            continue
        if st.result.image != want:
            st.result.image = want
            n += 1
    return n


_PLACEHOLDER_FRAGMENTS: Tuple[str, ...] = (
    "probe_no_tool", "final_answer", "answer_type", "<answer>", "{{", "}}",
)


def _check_answer_field(name: str, value: str, sample: Sample) -> str:
    """拦住"看起来填了、其实是把字段名/枚举值抄回来"的答案。

    实测中 8B 教师在这三处反复出错：
      1. final_answer 直接写 answer_type 的枚举值（"closed"/"short"/"measure"/"long"）；
      2. final_answer 写成 schema 里的路径字符串（"probe_no_tool.answer"）；
      3. 二元题给个游离词（"heterogeneous"/"Good"）——这类不拦，它不是结构性错误。
    前两类进 SFT 就是教模型输出垃圾，必须当解析失败处理（落 failures 而不是训练集）。
    """
    v = (value or "").strip()
    if not v:
        raise TeacherError(f"{name} 为空")
    if v.lower() in ANSWER_TYPES:
        raise TeacherError(f"{name} 是 answer_type 枚举值而非答案: {v!r}")
    low = v.lower()
    for frag in _PLACEHOLDER_FRAGMENTS:
        if frag.lower() in low:
            raise TeacherError(f"{name} 含占位符/字段名 {frag!r}: {v[:60]!r}")
    return v


def parse_response(resp: Any, sample: Sample) -> Trajectory:
    """把一条响应解析成 Trajectory（3 个 Step + final_answer + 探针占位）。"""
    payload = _extract_json(resp)
    msg = _unwrap_message(resp)

    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or len(raw_steps) != N_STEPS:
        n = len(raw_steps) if isinstance(raw_steps, list) else "非 list"
        raise TeacherError(f"steps 必须是长度 {N_STEPS} 的数组，实际 {n}")

    steps: List[Step] = []
    # 裁剪图的序号跟着「裁剪类调用」走，而不是循环变量 k：教师常写
    # [crop, measure, crop]，此时第 3 个 step 用的是第 2 张裁剪图，按 k 取
    # 会越界（老代码就是这样退到 'CROP_2' 字符串上的）。
    crop_idx = 0
    for k, s in enumerate(raw_steps):
        if not isinstance(s, dict):
            raise TeacherError(f"step {k} 不是对象: {type(s).__name__}")
        name = str(s.get("tool_name", "none"))
        if name not in TOOL_NAMES:
            raise TeacherError(f"step {k} 工具名非法: {name}")
        text = str(s.get("think", "")).strip()
        if name == "none":
            steps.append(Step(text=text))
            continue

        box = _step_box(s)
        args = _call_args(name, s, box)
        ref = str(s.get("result_image_ref") or "").strip()
        if name in ("crop", "zoom"):
            image = _crop_path_for(sample, crop_idx) or _ref_as_path(ref)
            crop_idx += 1
        else:
            # measure 只回数值；window 的观测图必须由 tools/ops 在沙箱重放产出，
            # 教师给的 ref 不是图，不能塞进 messages。
            image = _ref_as_path(ref)
        result = ToolResult(
            ok=bool(s.get("result_ok", True)),
            image=image,
            text=str(s.get("result_text", "")),
            meta={
                "crop_box": box.to_json() if box else None,
                "result_ref": ref,
                # 教师自述的执行结果不可信，必须由 tools/ops 在沙箱里重放（§5.3 第 1 项）
                "executed": False,
                "source": "teacher_planned",
            },
        )
        steps.append(Step(text=text, call=ToolCall(name=name, args=args), result=result))

    probe_raw = payload.get("probe_no_tool") or {}
    # a_1 探针缺失（或写成字段名）时轨迹不可用于 OGAR，直接当解析失败
    try:
        probe_answer = _check_answer_field("probe_no_tool.answer",
                                           str(probe_raw.get("answer", "")), sample)
    except TeacherError as e:
        raise TeacherError(f"a_1 探针缺失（{e}）：轨迹不可用于 OGAR") from None

    final_answer = _check_answer_field("final_answer",
                                       str(payload.get("final_answer", "")), sample)

    usage = _get(msg, "usage")
    meta: Dict[str, Any] = {
        "model": str(_get(msg, "model") or _opt(None, "model", "claude-opus-5")),
        "stop_reason": _get(msg, "stop_reason"),
        "usage": {
            "input_tokens": _get(usage, "input_tokens"),
            "output_tokens": _get(usage, "output_tokens"),
        },
        "probe_confidence": probe_raw.get("confidence"),
        "probe_rationale": probe_raw.get("rationale"),
        "final_rationale": payload.get("final_rationale", ""),
        # answer_type 的权威来源是问题自身（qgen 依问法判定），教师自报只留作诊断：
        # 8B 教师在枚举上几乎恒定输出 closed，若拿它覆盖，long/short/measure 全被抹平。
        "answer_type": sample.answer_type,
        "teacher_answer_type": payload.get("answer_type"),
        "roles": ["ORIGINAL"] + [f"CROP_{i + 1}" for i in range(len(crop_paths_of(sample)))],
        # 单请求形态下模型已经同时看到裁剪，a_1 探针天然被污染（见模块 docstring 第 4 条）
        "probe_contaminated": True,
        "probes_pending": True,
        "n_calls": sum(1 for st in steps if st.is_tool_turn),
    }

    return Trajectory(
        id=custom_id_of(sample),
        sample_id=str(sample.id),
        steps=steps,
        final_answer=final_answer,
        # 占位：score=0 仅表示「未判分」，由 validate 阶段重采后覆盖
        probes=[Probe(k=0, answer=probe_answer, score=0, parse_ok=True)],
        delta_k=[],
        delta_total=0,
        source="teacher",
        meta=meta,
    )


# ---------------------------------------------------------------- client


def _ensure_client(client: Any) -> Any:
    """client 一律参数注入；没注入才尝试构造，缺依赖时给明确指引。"""
    if client is not None:
        return client
    try:
        import anthropic  # noqa: F401
    except ImportError as e:
        raise TeacherError(
            "未安装 anthropic SDK，无法构造真实 client。请 `pip install anthropic`，"
            "或显式注入 client=...（单测/离线重放用 mock）。"
        ) from e
    return anthropic.Anthropic()


# ---------------------------------------------------------------- mock client


class _Obj(dict):
    """同时支持 ``obj["x"]`` 与 ``obj.x`` 的最小对象（模拟 SDK 的 pydantic 模型）。"""

    def __getattr__(self, k: str) -> Any:
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


def mock_answer_for(sample: Sample) -> str:
    """mock 用的确定性"标准答案"。

    真答案优先；没有（G3 负样本是纯知识题，gold 由教师给）就用题面哈希派生一个
    稳定字符串。**必须确定性**：no_tool_ablation 会拿同一个 sample 问第二遍，
    两次答案不一致的话负样本会被误判成"裸答答错"。
    """
    a = str(sample.answer or "").strip()
    if a:
        return a
    import hashlib
    h = hashlib.md5(str(sample.question).encode("utf-8")).hexdigest()[:6]
    return f"[mock-answer-{h}]"


def _mock_payload(sample: Sample) -> Dict[str, Any]:
    """造一条结构合法的教师输出：有框就走 crop→zoom→none，没框就三步全 none。"""
    box = (sample.boxes or [None])[0]
    steps: List[Dict[str, Any]] = []

    if box is not None:
        steps.append({
            "think": "[mock] 先裁到标注区域，排除周围组织的干扰。",
            "tool_name": "crop",
            "tool_x0": box.x0, "tool_y0": box.y0, "tool_x1": box.x1, "tool_y1": box.y1,
            "result_ok": True, "result_text": "[mock] crop ok", "result_image_ref": "",
        })
        pad = max(2.0, 0.15 * max(box.w, box.h))
        zx0, zy0 = max(0.0, box.x0 - pad), max(0.0, box.y0 - pad)
        steps.append({
            "think": "[mock] 再放大一档看清边缘与密度。",
            "tool_name": "zoom",
            "tool_x0": zx0, "tool_y0": zy0, "tool_x1": box.x1 + pad, "tool_y1": box.y1 + pad,
            "tool_out_side": 1024,
            "result_ok": True, "result_text": "[mock] zoom ok", "result_image_ref": "",
        })
    while len(steps) < N_STEPS:
        steps.append({
            "think": "[mock] 结合裁剪结果给出答案。",
            "tool_name": "none",
            "result_ok": True, "result_text": "",
        })

    # 负样本（needs_tool=False）的语义正是"不用工具也能答对"，所以 mock 的 no-tool 探针
    # 必须给出正确答案，否则 validate 会把它们按 negative_sample_unsolved 全拒掉。
    needs_tool = bool(sample.meta.get("needs_tool", bool(sample.boxes)))
    if needs_tool:
        probe_answer = "[mock] 不给工具时看不准。"
        probe_conf = 0.2
        probe_why = "[mock] 分辨率不足。"
    else:
        probe_answer = mock_answer_for(sample)
        probe_conf = 0.9
        probe_why = "[mock] 全图信息已足够。"

    return {
        "steps": steps[:N_STEPS],
        "probe_no_tool": {
            "answer": probe_answer,
            "confidence": probe_conf,
            "rationale": probe_why,
        },
        "final_answer": mock_answer_for(sample),
        "answer_type": sample.answer_type or "closed",
        "final_rationale": "[mock] 依据裁剪区域作答。",
    }


class MockBatchClient:
    """离线的 anthropic 客户端替身：只实现 ``messages.batches`` 的三个方法。

    **产出的轨迹带 ``meta['mock']=True``，不能进训练集**——它只用来验证
    build→create→poll→collect→parse 这条链路本身是通的。
    """

    def __init__(self, model: str = "claude-opus-5") -> None:
        self.model = model
        self._batches: Dict[str, List[Dict[str, Any]]] = {}
        self._payloads: Dict[str, Dict[str, Any]] = {}
        self._n = 0
        self.messages = _Obj(batches=_Obj(
            create=self._create, retrieve=self._retrieve, results=self._results))

    def _create(self, requests: List[Dict[str, Any]], **_: Any) -> Any:
        self._n += 1
        bid = f"mock_batch_{self._n}"
        self._batches[bid] = list(requests)
        return _Obj(id=bid, processing_status="ended")

    def _retrieve(self, batch_id: str, **_: Any) -> Any:
        if batch_id not in self._batches:
            raise TeacherError(f"未知的 mock batch: {batch_id}")
        return _Obj(id=batch_id, processing_status="ended")

    def _results(self, batch_id: str, **_: Any) -> Any:
        for req in self._batches.get(batch_id, []):
            cid = str(req.get("custom_id"))
            payload = self._payloads.get(cid)
            if payload is None:
                yield _Obj(custom_id=cid, result=_Obj(
                    type="errored", error=_Obj(type="invalid_request_error", message="no payload")))
                continue
            msg = _Obj(
                model=self.model,
                stop_reason="end_turn",
                content=[_Obj(type="text", text=json.dumps(payload, ensure_ascii=False))],
                usage=_Obj(input_tokens=0, output_tokens=0),
            )
            yield _Obj(custom_id=cid, result=_Obj(type="succeeded", message=msg))

    # ---- 注册样本：run_batch 只把 custom_id 传进来，payload 由样本决定
    def register(self, samples: Sequence[Sample]) -> "MockBatchClient":
        self._payloads = {custom_id_of(s): _mock_payload(s) for s in samples}
        return self

    _payloads: Dict[str, Dict[str, Any]] = {}


def mock_client(samples: Sequence[Sample], model: str = "claude-opus-5") -> MockBatchClient:
    """给 ``run_batch`` 用的离线客户端。"""
    return MockBatchClient(model=model).register(samples)


def _sleep_default(sec: float) -> None:
    time.sleep(sec)


def _chunks(reqs: List[Dict[str, Any]], max_requests: int, max_bytes: int) -> List[List[Dict[str, Any]]]:
    """按条数与字节数双上限切分（Batch API 限 ≤100k 请求 / ≤256MB）。"""
    out: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    cur_bytes = 0
    for r in reqs:
        size = len(json.dumps(r, ensure_ascii=False).encode("utf-8"))
        if cur and (len(cur) >= max_requests or cur_bytes + size > max_bytes):
            out.append(cur)
            cur, cur_bytes = [], 0
        cur.append(r)
        cur_bytes += size
    if cur:
        out.append(cur)
    return out


def _create_with_retry(client: Any, batch: List[Dict[str, Any]], max_retries: int,
                       sleep_fn: Callable[[float], None]) -> Any:
    """``messages.batches.create`` 的重试外壳。错误类型不做特判（SDK 异常无法在此导入）。"""
    last: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return client.messages.batches.create(requests=batch)
        except Exception as e:  # noqa: BLE001 - 网络/SDK/限流都要重试
            last = e
            if attempt < max_retries:
                sleep_fn(min(60.0, 2.0 ** attempt))
    raise TeacherError(f"batch 创建失败，已重试 {max_retries} 次: {last}") from last


def _poll(client: Any, batch_id: str, interval: float, max_wait_s: float,
          sleep_fn: Callable[[float], None], now_fn: Callable[[], float]) -> None:
    """轮询直到 processing_status == 'ended'；超时抛 TeacherError。"""
    t0 = now_fn()
    while True:
        b = client.messages.batches.retrieve(batch_id)
        status = _get(b, "processing_status")
        if status in ("ended", "canceling", "canceled"):
            if status in ("canceling", "canceled"):
                raise TeacherError(f"batch {batch_id} 被取消（status={status}）")
            return
        if now_fn() - t0 > max_wait_s:
            raise TeacherError(f"batch {batch_id} 轮询超时（{max_wait_s:.0f}s，status={status}）")
        sleep_fn(interval)


def _collect(client: Any, batch_id: str) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """读一批结果，返回 (成功的 custom_id→message, 失败的 custom_id→原因)。"""
    ok: Dict[str, Any] = {}
    bad: Dict[str, Dict[str, Any]] = {}
    for item in client.messages.batches.results(batch_id):
        cid = str(_get(item, "custom_id"))
        res = _get(item, "result")
        rtype = _get(res, "type")
        if rtype == "succeeded":
            ok[cid] = _get(res, "message")
        else:
            bad[cid] = {"type": rtype, "error": str(_get(res, "error") or "")}
    return ok, bad


def run_batch(samples: Sequence[Sample], cfg: Any, client: Any = None, *,
              existing_ids: Optional[Iterable[str]] = None,
              failure_sink: Optional[List[Dict[str, Any]]] = None,
              on_traj: Optional[Callable[[Trajectory], None]] = None,
              sleep_fn: Optional[Callable[[float], None]] = None,
              now_fn: Optional[Callable[[], float]] = None,
              poll_interval: Optional[float] = None,
              max_wait_s: Optional[float] = None) -> List[Trajectory]:
    """Batch API 全流程：build → create → poll → results → parse，带重试与幂等。

    幂等：``custom_id`` 已出现在 ``existing_ids``（通常是上次落盘的 ``seen_ids``）
    或本轮已提交过的，一律不再提交。返回顺序与入参一致（缺失的跳过）。

    失败不抛，记进 ``failure_sink``（若提供）：``{'custom_id','sample_id','stage','error'}``。
    """
    client = _ensure_client(client)
    sleep = sleep_fn or _sleep_default
    now = now_fn or time.time
    interval = float(_opt(cfg, "poll_interval_s", 10.0) if poll_interval is None else poll_interval)
    wait_cap = float(_opt(cfg, "max_wait_s", 86400.0) if max_wait_s is None else max_wait_s)
    max_retries = int(_opt(cfg, "max_retries", 4))
    concurrency = max(1, int(_opt(cfg, "concurrency", 8)))
    max_req = int(_opt(cfg, "batch_max_requests", 100000))
    max_bytes = int(_opt(cfg, "batch_max_bytes", 268435456))

    # ---- 幂等去重：existing_ids 可以是 custom_id 也可以是 sample.id；
    # 二者任一命中即视为已产出，本轮不再提交。
    skip = {str(x) for x in (existing_ids or [])}
    pending: List[Sample] = []
    seen: set = set()
    for s in samples:
        cid = custom_id_of(s)
        if cid in skip or str(s.id) in skip or cid in seen:
            continue
        seen.add(cid)
        pending.append(s)

    def _fail(sample: Sample, stage: str, err: Exception) -> None:
        if failure_sink is not None:
            failure_sink.append(
                {"custom_id": custom_id_of(sample), "sample_id": str(sample.id),
                 "stage": stage, "error": f"{type(err).__name__}: {err}"}
            )

    if not pending:
        return []

    # ---- 构造请求（读图 + base64 是 I/O 密集，并发做）
    by_cid: Dict[str, Sample] = {custom_id_of(s): s for s in pending}
    built: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(build_request, s, cfg): s for s in pending}
        for fut, s in futs.items():
            try:
                built.append(fut.result())
            except Exception as e:  # noqa: BLE001 - 单条坏样本不拖垮整批
                _fail(s, "build", e)
    if not built:
        return []

    # ---- 提交 + 轮询 + 收集
    payloads: Dict[str, Any] = {}
    parsed_cids: set = set()      # 已解析并回调过的 custom_id（重试轮次不重复产出）
    emitted_out: List[Trajectory] = []
    remaining = list(built)
    for round_no in range(max_retries + 1):
        if not remaining:
            break
        for chunk in _chunks(remaining, max_req, max_bytes):
            try:
                batch = _create_with_retry(client, chunk, max_retries, sleep)
                bid = str(_get(batch, "id"))
                _poll(client, bid, interval, wait_cap, sleep, now)
                ok, bad = _collect(client, bid)
            except Exception as e:  # noqa: BLE001
                for r in chunk:
                    _fail(by_cid[r["custom_id"]], "batch", e)
                continue
            payloads.update(ok)
            # 逐块解析并回调落盘：全量任务动辄数小时，一次性返回意味着中途被杀
            # 就全丢。回调里抛异常会被当成该条的解析失败处理，不中断整批。
            block_out: List[Trajectory] = []
            for r in chunk:
                cid = r["custom_id"]
                if cid not in ok or cid in parsed_cids:
                    continue
                parsed_cids.add(cid)
                s = by_cid[cid]
                try:
                    t = parse_response(ok[cid], s)
                except Exception as e:  # noqa: BLE001
                    _fail(s, "parse", e)
                    continue
                block_out.append(t)
                if on_traj is not None:
                    try:
                        on_traj(t)
                    except Exception as e:  # noqa: BLE001 - 落盘失败不能吞掉整批
                        _fail(s, "sink", e)
            emitted_out.extend(block_out)
            # 只重试「可能自愈」的失败：canceled / expired / server 错误
            retry_cids = set(bad) - set(ok)
            remaining = [r for r in chunk if r["custom_id"] in retry_cids]
            for cid in retry_cids:
                if round_no >= max_retries:
                    _fail(by_cid[cid], "result", TeacherError(str(bad[cid])))

    # ---- 补齐：逐块循环里已解析并回调过的直接复用，剩下的（重复 id 或末轮重试）
    # 在这里按入参顺序解析一次，保证返回顺序与 samples 一致。
    by_traj_id: Dict[str, Trajectory] = {t.id: t for t in emitted_out}
    out: List[Trajectory] = []
    emitted: set = set()
    for s in samples:
        cid = custom_id_of(s)
        if cid in emitted:
            continue
        if cid in parsed_cids:
            emitted.add(cid)
            t = by_traj_id.get(cid)
            if t is not None:
                out.append(t)
            continue
        if cid not in payloads:
            continue
        emitted.add(cid)
        try:
            out.append(parse_response(payloads[cid], s))
        except Exception as e:  # noqa: BLE001
            _fail(s, "parse", e)
    return out
