"""分级前缀探针的重采（PROPOSAL §7.1 的 a_0..a_K 读数）。

为什么必须有这一步
------------------
``synth/teacher.py`` 用**一条请求**同时给出原图与所有裁剪，模型在写
``probe_no_tool`` 时已经看过裁剪了（见该模块 docstring 第 4 条）。所以那里落盘的
``Trajectory.probes`` 只是占位：``[Probe(k=0, ...)]``、``delta_k=[]``、
``meta['probes_pending']=True``。可 ``rl/advantage.py:ogar_masks`` 的硬不变量是
``len(probe_scores) == n_calls + 1``，占位数据（1 个探针）在 K≥1 的轨迹上必然违约。

本模块用**逐前缀重问**的方式把真读数补齐（一条 K 次调用的轨迹需要 K+1 个读数）：

  a_0  只看原图（不给任何裁剪）回答该问题；
  a_k  看原图 + 前 k 次工具调用的**返回图**回答该问题（1<=k<=K-1）；
  a_K  直接复用 ``traj.final_answer`` 判分——那正是模型自己的最终答案，不必再问。

因此需要发请求的下标只有 ``0..K-1``（K 条）。K=0（无工具调用）时一个请求都不发：
此时 a_0 就是 ``final_answer`` 本身（模型只看了原图就作答），``probes`` 只有一条，
``delta_k=[]``、``delta_total=0``——这正好满足硬不变量 ``len(probes)==n_calls+1==1``。

判分一律走 ``rl/grader.py:DeterministicGrader.score``（同输入同输出、零成本）。
空答案/纯空白答案按 PROPOSAL §7.1 的"解析失败记 0（不进 Δ）"处理：内存里记
``score=0, parse_ok=False`` 并写进 ``meta['probe_empty_ks']``——**不是** -1，
也绝不伪造 ``parse_ok=True``（那会让 advantage 的解析失败检查永远不可达）。
不可判分的样本（``long`` / 金标为空）直接跳过重采，不写 ``probes_pending=False``。

工程约定
--------
1. **幂等**：``custom_id = probe-<消毒后的 traj.id>-k<k>``，``existing_ids`` 里出现过的
   一律不再提交；``out_path`` 里已写全的轨迹在续跑时直接采纳，不再发请求。
   同批里两条轨迹撞出同一个 custom_id 时，后来的那条记失败并保持待采——
   Batch API 收到重复 custom_id 会整批报错。
2. **可续跑**：``out_path`` 给了就每采完一条轨迹立即落盘（本模块自己的
   ``_ProbeSink``：同 id 以最后一行为准，写完回读校验）。注意**不能**用
   ``utils/io.JsonlWriter``：它按 id 去重，撞到已见 id 时静默返回 False，
   磁盘上的旧版不完整行就永远不会被纠正。
3. **失败不抛**：单条请求/单条轨迹的失败记进 ``failure_sink``，该轨迹保持
   ``probes_pending=True`` 留给下一轮，不污染同批其他轨迹。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..rl.advantage import staged_deltas
from ..rl.grader import DeterministicGrader
from ..utils.io import iter_jsonl
from ..utils.schema import Probe, Sample, Trajectory
from .crops import crops_dir
from .teacher import (
    MAX_VISION_TOKENS,
    TeacherError,
    _answer_type_hint,
    _chunks,
    _collect,
    _create_with_retry,
    _get,
    _image_block,
    _media_type,
    _opt,
    _poll,
    _read_image,
    fit_size,
    sanitize_images,
)

__all__ = [
    "PROBE_MAX_TOKENS",
    "PROBE_META_KEYS",
    "ProbeError",
    "build_probe_instruction",
    "build_probe_request",
    "collect_probes",
    "pending_probe_keys",
    "probe_custom_id",
    "probe_max_tokens",
    "probes_complete",
]

#: 探针只需要一个短答案。PROPOSAL §7.1 的硬上限是 **≤32 token**
#: （"每次探针的答案长度上限设短（≤32 token），否则探针成本与方差都会失控"），
#: 配置只能往小调、不能往大调（``probe_max_tokens`` 会钳制）。
PROBE_MAX_TOKENS: int = 32
#: Batch API 只接受这些字符，且 custom_id 长度 <= 64
_CUSTOM_ID_BAD = re.compile(r"[^A-Za-z0-9_-]")
_MAX_CUSTOM_ID: int = 64
#: 探针结果的来源标记（写回 meta，便于下游区分"重采过"与"教师占位"）
PROBE_SOURCE: str = "recollected"
#: 探针相关的 meta 字段全集：``_finalize`` 写、``_adopt`` 搬，两边必须一致
#: （否则"当场重采"与"从磁盘采纳"的轨迹 meta 字段集不同，下游 checkpoint 不可复现）
PROBE_META_KEYS: Tuple[str, ...] = (
    "probes_pending",
    "probe_source",
    "probe_empty_ks",
    "probe_model",
    "probe_max_tokens",
    "probe_skip_reason",
)


class ProbeError(RuntimeError):
    """探针重采链路的可预期失败（编号越界 / 图片缺失 / custom_id 无法生成）。"""


# ---------------------------------------------------------------- custom_id


def probe_custom_id(traj_id: str, k: int) -> str:
    """``probe-<traj_id>-k<k>``，稳定、可逆，且只含 API 允许字符。

    轨迹 id 里常带 ``:`` / ``#`` / ``/``（如 ``traj-chestx_ray14:xxx#G1-en-10``），
    直接拼会被 Batch API 拒收。消毒本身**不保唯一**（``a:b`` 与 ``a#b`` 都会变成
    ``a_b``），所以一旦发生替换或截断就追加 8 位 sha1 前缀，把撞车概率压到可忽略。
    """
    raw = str(traj_id or "").strip()
    if not raw:
        raise ProbeError("traj.id 为空，无法生成 custom_id")
    kk = int(k)
    if kk < 0:
        raise ProbeError(f"探针编号必须 >= 0，收到 {kk}")

    head = "probe-"
    suffix = f"-k{kk}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    safe = _CUSTOM_ID_BAD.sub("_", raw)
    tagged = safe != raw
    budget = _MAX_CUSTOM_ID - len(head) - len(suffix) - (len(digest) + 1 if tagged else 0)
    if len(safe) > budget:
        # 截断同样会丢区分度，此时也把哈希带上
        tagged = True
        budget = _MAX_CUSTOM_ID - len(head) - len(suffix) - len(digest) - 1
        safe = safe[:budget]
    if budget < 1:
        raise ProbeError(f"traj.id 过长，无法在 64 字符内生成 custom_id: {raw[:80]!r}")
    cid = f"{head}{safe}{'-' + digest if tagged else ''}{suffix}"
    if len(cid) > _MAX_CUSTOM_ID:  # 防御：格式改动后仍不能越界
        raise ProbeError(f"custom_id 超长（{len(cid)} > {_MAX_CUSTOM_ID}）: {cid}")
    return cid


# ---------------------------------------------------------------- 小工具


def _crops_dir(cfg: Any) -> Optional[str]:
    """取裁剪图目录；cfg 缺 ``paths.processed`` 时返回 None（单测/离线重放）。"""
    try:
        return crops_dir(cfg)
    except Exception:  # noqa: BLE001 - 配置不完整不该阻断请求构造
        return None


def _cfg_node(cfg: Any, *path: str) -> Any:
    """按路径取任意深度的配置节点；``cfg`` 可能是 Cfg / dict / None。

    既接受整份 cfg（``cfg.rl.probe``），也接受已经被上层取好的子字典，
    所以先按路径走，走不通再把它自己当成待查节点。
    """
    cur = cfg
    for part in path:
        if cur is None:
            return None
        if isinstance(cur, Mapping):
            cur = cur.get(part)
            continue
        cur = getattr(cur, part, None)
    return cur


def _probe_cfg(cfg: Any) -> Any:
    """``cfg.rl.probe``；整份 cfg / rl 子字典 / probe 子字典三种写法都认。"""
    for path in (("rl", "probe"), ("probe",)):
        node = _cfg_node(cfg, *path)
        if isinstance(node, Mapping):
            return node
    # 已经是 probe 子字典本身（含 max_tokens 键）
    if isinstance(cfg, Mapping) and "max_tokens" in cfg:
        return cfg
    return None


def probe_max_tokens(cfg: Any) -> int:
    """本次探针请求真正生效的 ``max_tokens``。

    优先级：``cfg.rl.probe.max_tokens`` > 32（PROPOSAL §7.1 的硬上限）。
    配置写得比 32 大一律钳到 32 并 warn（探针成本与方差都随长度失控）；
    非法值（非整数 / <=0）回落到 32 并 warn。**不要**在这里抛异常——
    一个写错的配置不该让整轮重采挂掉。
    """
    node = _probe_cfg(cfg)
    raw = node.get("max_tokens") if isinstance(node, Mapping) else None
    if raw is None:
        return PROBE_MAX_TOKENS
    try:
        val = int(raw)
    except (TypeError, ValueError):
        warnings.warn(
            f"rl.probe.max_tokens 非法（{raw!r}），回落到 {PROBE_MAX_TOKENS}",
            RuntimeWarning, stacklevel=2,
        )
        return PROBE_MAX_TOKENS
    if val < 1:
        warnings.warn(
            f"rl.probe.max_tokens={val} 非正，回落到 {PROBE_MAX_TOKENS}",
            RuntimeWarning, stacklevel=2,
        )
        return PROBE_MAX_TOKENS
    if val > PROBE_MAX_TOKENS:
        warnings.warn(
            f"rl.probe.max_tokens={val} 超过 PROPOSAL §7.1 的上限，"
            f"已钳到 {PROBE_MAX_TOKENS}",
            RuntimeWarning, stacklevel=2,
        )
        return PROBE_MAX_TOKENS
    return val


def _sample_map(samples: Any) -> Dict[str, Sample]:
    """把 Sample 序列或 ``{id: Sample}`` 映射统一成 ``{str(id): Sample}``。"""
    items: Iterable[Sample]
    if isinstance(samples, Mapping):
        items = list(samples.values())
    else:
        items = list(samples)
    out: Dict[str, Sample] = {}
    for s in items:
        out[str(s.id)] = s
    return out


def _sample_for(traj: Trajectory, smap: Mapping[str, Sample]) -> Optional[Sample]:
    """按 ``traj.sample_id`` 找样本，缺失时退回 ``traj.id``（老数据两种写法都有）。"""
    sid = str(traj.sample_id or "")
    if sid and sid in smap:
        return smap[sid]
    return smap.get(str(traj.id))


def _tool_turns(traj: Trajectory) -> List[Any]:
    """按时间顺序取出所有工具调用轮（``Step.call`` 非空的那些）。"""
    return [st for st in (traj.steps or []) if st.is_tool_turn]


def _prefix_observations(traj: Trajectory, k: int) -> List[Dict[str, Any]]:
    """前 k 次工具调用"模型当时看到了什么"，按调用顺序返回。

    每个元素是一轮的观测：``call_index``（1 起）、``name``、``ok``、``image``
    （返回图路径或 None）、``text``、``error``。**这是唯一的事实来源**——
    提示词里的图片编号、文本观测都由它生成，不能自己按 k 数图。

    为什么要区分"调用了"与"返回了图"：measure / window 这类调用只回数值或失败文本，
    含这类轮次的轨迹上「第 k 个工具轮 = 第 k 张附件」根本不成立，按 k 数图会让
    指令宣称一张并不存在的图（见 build_probe_instruction）。
    """
    out: List[Dict[str, Any]] = []
    for i, st in enumerate(_tool_turns(traj)[:max(0, int(k))]):
        res = st.result
        img = getattr(res, "image", None) if res is not None else None
        out.append({
            "call_index": i + 1,
            "name": str(getattr(st.call, "name", "") or ""),
            "ok": bool(getattr(res, "ok", False)) if res is not None else False,
            "image": str(img) if img else None,
            "text": str(getattr(res, "text", "") or "") if res is not None else "",
            "error": str(getattr(res, "error", "") or "") if res is not None else "",
            "has_result": res is not None,
        })
    return out


def _prefix_images(traj: Trajectory, k: int) -> List[str]:
    """a_k 该看到的前缀图：前 k 次工具调用里**非空**的返回图路径。

    语义上就是"第 1..k 次调用已经执行完"，所以只扫前 k 个工具轮；某轮没有图
    （measure 只回数值）就自然少一张，而不是拿后面轮次的图来凑数——那会让 a_1
    提前看到第 2 次调用的观测，探针语义就失效了。
    """
    return [o["image"] for o in _prefix_observations(traj, k) if o["image"]]


def _message_text(msg: Any) -> Optional[str]:
    """把一条 Message 里所有 text block 拼起来；一个 text block 都没有时返回 None。

    返回 None 表示"这次请求根本没拿到回答"（API 侧截断 / 只给了 tool_use 之类），
    调用方按失败处理并重试；返回空串则是模型确实写了空内容——**也按解析失败处理**
    （PROPOSAL §7.1：R_ORM 解析失败记 0，不进 Δ），由 ``_finalize`` 统一记账，
    绝不当作"答错了"判 -1（那会造出 ±2 的极端 Δ）。
    """
    content = _get(msg, "content")
    if content is None:
        return None
    texts: List[str] = []
    for b in content:
        if _get(b, "type") == "text":
            t = _get(b, "text") or ""
            texts.append(str(t))
    if not texts:
        return None
    return "".join(texts).strip()


# ---------------------------------------------------------------- 指令


def _render_observation(o: Mapping[str, Any]) -> str:
    """把一轮工具观测渲染成一行提示词文本（图片单独进 image block，这里只写文本）。"""
    idx = o.get("call_index")
    name = o.get("name") or "unknown tool"
    head = f"  - tool call {idx} ({name})"
    if not o.get("has_result", True):
        return head + ": no observation was recorded for this call."
    if not o.get("ok", True):
        err = str(o.get("error") or "").strip() or "(empty error message)"
        return head + f" FAILED: ERROR: {err}"
    parts: List[str] = []
    if o.get("image"):
        parts.append("returned an image (see the corresponding IMAGE line above)")
    text = str(o.get("text") or "").strip()
    if text:
        parts.append(f'returned text: "{text}"')
    if not parts:
        parts.append("returned no observable content")
    return head + ": " + "; ".join(parts)


def build_probe_instruction(sample: Sample, k: int,
                            observations: Optional[Sequence[Mapping[str, Any]]] = None) -> str:
    """探针请求的文本指令。

    刻意**不**用 teacher 的 JSON schema 指令：探针要的是自由文本答案，由
    ``DeterministicGrader.parse`` 抽最后一行。也刻意只给 user 消息——该代模型禁止 prefill。

    ``observations`` 必须是 ``_prefix_observations(traj, k)`` 的结果：**图片编号与实际
    附件列表一一对应**，并把非图片的观测（measure 读数、工具失败文本等）一并渲染。
    绝不能按 k 去数图——contains measure/zoom 的轨迹上"第 k 个工具轮"与"第 k 张图"
    不是一回事，那样指令会宣称一张并不存在的 CROP，还会把数值观测整个丢掉。
    """
    kk = max(0, int(k))
    obs = list(observations or [])
    images = [o for o in obs if o.get("image")]

    role_lines = ["  IMAGE 1 = ORIGINAL"]
    for i, o in enumerate(images):
        role_lines.append(
            f"  IMAGE {i + 2} = return image of tool call {o.get('call_index')} "
            f"({o.get('name') or 'unknown tool'})"
        )
    role_block = "\n".join(role_lines)

    if kk == 0:
        crop_note = "Attached: the ORIGINAL image only; no tool call has been executed yet."
        obs_block = ""
    else:
        if obs:
            obs_block = "\nTool observations already available at this prefix:\n" + \
                "\n".join(_render_observation(o) for o in obs) + "\n"
        else:
            obs_block = ""
        if images:
            crop_note = (
                f"Attached: the ORIGINAL image plus the {len(images)} image(s) returned by "
                f"the first {kk} tool call(s), together with every text observation they "
                f"returned — i.e. what a model would have seen after executing those calls."
            )
        else:
            crop_note = (
                f"Attached: the ORIGINAL image only — none of the first {kk} tool call(s) "
                f"returned an image. Their text observations are listed below; treat them as "
                f"the only extra information a model would have had at this prefix."
            )

    return f"""You are a senior radiologist answering a question about a medical image.

Images attached, in order:
{role_block}

{crop_note}
{obs_block}
Question: {sample.question}
{_answer_type_hint(sample)}

Answer with the FINAL ANSWER ONLY: one single line, no explanation, no reasoning,
no preamble, no labels, no markdown. Do not restate the question."""


# ---------------------------------------------------------------- 请求构造


def build_probe_request(traj: Trajectory, sample: Sample, k: int, cfg: Any) -> Dict[str, Any]:
    """构造 a_k 探针的 Batch 请求（Anthropic 形状）。

    Args:
        traj:   待重采的轨迹（会就地 ``sanitize_images`` 修正 step.result.image）。
        sample: 对应的样本（原图 + 问题 + 金标）。
        k:      a_k 的下标，只允许 ``0 <= k < n_calls``；``a_K`` 复用 final_answer 不发请求。
                K=0 时没有任何请求可发（a_0 就是 final_answer 本身）。
        cfg:    teacher 子配置（模型名 / 长边上限 / 图片数上限）。

    Returns:
        ``{"custom_id": str, "params": {...}}``，``params`` 可直接作为 Batch 请求体，
        也可 ``client.messages.create(**params)``。**只有 user 消息**（禁止 prefill）。
    """
    n_calls = int(traj.n_calls)
    kk = int(k)
    # 需要发请求的下标是 0..K-1：a_K 复用 final_answer，不必再问。
    # K=0 时没有可发的请求——a_0 与 a_K 是同一条读数，直接由 final_answer 判分得到。
    if kk < 0 or kk >= n_calls:
        raise ProbeError(
            f"k 必须在 0..{n_calls - 1} 之间（a_K 直接复用 traj.final_answer，不发请求；"
            f"K=0 时无请求可发），收到 k={kk}（traj={traj.id!r}, n_calls={n_calls}）"
        )

    # 路径修正必须先做：老数据里 ToolResult.image 可能还是 'CROP_1' 这种标签
    cdir = _crops_dir(cfg)
    sanitize_images(traj, sample, cdir)

    crop_imgs = _prefix_images(traj, kk)
    if not sample.image:
        raise ProbeError(f"sample.image 为空: {sample.id}")
    paths: List[str] = [str(sample.image)] + crop_imgs

    max_images = int(_opt(cfg, "max_images_per_request", 20))
    if len(paths) > max_images:
        raise ProbeError(f"单请求图片数 {len(paths)} 超过上限 {max_images}（sample={sample.id}）")

    max_side = int(_opt(cfg, "max_side_px", 2576))
    blocks: List[Dict[str, Any]] = []
    for path in paths:
        media = _media_type(path)  # 先校验后缀，别让 PIL 先报看不懂的错
        raw, w, h = _read_image(path)
        fit_size(w, h, max_side, MAX_VISION_TOKENS)  # 只做尺寸审计，不改字节
        blocks.append(_image_block(raw, media))
    # 指令必须由**实际附件**生成：图片编号与 paths 一一对应，文本观测一并带上
    blocks.append({"type": "text",
                   "text": build_probe_instruction(sample, kk, _prefix_observations(traj, kk))})

    params: Dict[str, Any] = {
        "model": str(_opt(cfg, "model", "claude-opus-5")),
        "max_tokens": probe_max_tokens(cfg),
        # 只有 user 一条消息：禁止 message prefill
        "messages": [{"role": "user", "content": blocks}],
    }
    return {"custom_id": probe_custom_id(traj.id, kk), "params": params}


# ---------------------------------------------------------------- 完成度判定


def probes_complete(traj: Trajectory) -> bool:
    """该轨迹的探针是否已经可用来训练。

    四项同时成立才算完成：``probes_pending`` 不为真、带 ``probe_source``、
    长度 = K+1、``delta_k`` 与 ``staged_deltas`` 一致。任一不满足就视为"待重采"——
    宁可贵一点重采一次，也不能把占位探针送进 OGAR。

    注意 ``probes_pending`` 用 **默认 True** 取：键缺失的轨迹（老工具/手工构造的
    Trajectory）不能被当成"已采完"——教师占位恰好是 K=0 且 probes=[0 分占位]，
    一旦按"不待采"放行，这条占位就会被冻结成 a_0 送进训练。
    同理必须有 ``probe_source``：没有来源标记的探针无法区分"重采过"与"教师自述"。
    """
    try:
        if traj.meta.get("probes_pending", True):
            return False
        if not str(traj.meta.get("probe_source") or "").strip():
            return False
        n_calls = int(traj.n_calls)
        probes = list(traj.probes or [])
        if len(probes) != n_calls + 1:
            return False
        want = staged_deltas([p.score for p in probes])
        return [int(v) for v in (traj.delta_k or [])] == want
    except Exception:  # noqa: BLE001 - 分数非法/结构不全一律按"未完成"
        return False


def pending_probe_keys(trajs: Sequence[Trajectory], samples: Any,
                       cfg: Any) -> List[Tuple[str, str, int]]:
    """列出还没采的 ``(traj_id, sample_id, k)``，供幂等续跑。

    只列真正需要发请求的下标 ``0..K-1``（a_K 复用 final_answer）；K=0 的轨迹不需要任何
    请求，因此不出现在清单里，但它仍会由 ``collect_probes`` 写回（a_0=final_answer）。
    找不到对应 Sample 的轨迹、以及样本本身不可判分（``long`` / 金标为空）的轨迹
    直接跳过——后者即使采了也进不了 Δ 通道。``cfg`` 只用来建 grader 判可判分性。
    调用方通常用本函数的结果与上一轮的 ``existing_ids`` 做差再提交。
    """
    smap = _sample_map(samples)
    grader = DeterministicGrader(cfg)
    out: List[Tuple[str, str, int]] = []
    for traj in trajs:
        if probes_complete(traj):
            continue
        sample = _sample_for(traj, smap)
        if sample is None:
            continue
        if not grader.gradable(sample):
            continue
        for k in range(int(traj.n_calls)):
            out.append((str(traj.id), str(sample.id), k))
    return out


# ---------------------------------------------------------------- 写回


def _score_answer(grader: DeterministicGrader, sample: Sample,
                  answer: str) -> Tuple[int, bool]:
    """给一条探针读数判分，返回 ``(score, parse_ok)``。

    空/纯空白（``n/a`` / ``unknown`` 之类也算）是**解析失败**，按 PROPOSAL §7.1
    记 0 且不进 Δ——绝不能用 grader 的 -1：那会把"模型没给答案"记成"答错"，
    造出一个 ±2 的 Δ 尖峰。顺带让 ``advantage.validate_probes`` 的 ``parse_ok``
    检查真正可达（以前 ``_finalize`` 一律写 True，那条检查永远不会触发）。
    """
    # 按 grader 自己的口径判"没答"：它的 explain 会先 parse 再看 blank 形式，
    # 只有这里判不了的复杂情况才靠 reason 兜底
    if not str(answer or "").strip():
        return 0, False
    meta = grader.explain(sample, answer)
    if str(meta.get("reason") or "") == "empty_pred":
        # "n/a" / "-" / "unknown" 等：parse 没抽到实质内容，同样记解析失败
        return 0, False
    return int(meta["score"]), True


def _finalize(traj: Trajectory, sample: Sample, texts: Sequence[str],
              grader: DeterministicGrader, model: str,
              max_tokens: int = PROBE_MAX_TOKENS) -> None:
    """把 a_0..a_{K-1} 的回答与 a_K=final_answer 一起判分并写回轨迹。

    K>=1 时 ``texts`` 是 k=0..K-1 的重问结果，末尾补上复用 final_answer 的 a_K，
    共 K+1 条；K=0 时 ``texts`` 为空，只剩一条 a_0=final_answer（a_0 与 a_K 重合），
    保持 ``len(probes)==n_calls+1``。

    ``max_tokens`` 必须是构造请求时真正生效的那个值（``probe_max_tokens(cfg)``），
    写进 meta 便于复盘"这条读数是不是被长度上限截断的"。

    不可判分的样本（``long`` / 金标为空）直接抛 ``ProbeError``：调用方应当先用
    ``grader.gradable`` 把它们挡掉，**不能**给它们盖上 ``probes_pending=False``——
    那等于把本来进不了 Δ 通道的轨迹"洗"成可用轨迹。
    """
    if not grader.gradable(sample):
        raise ProbeError(
            f"样本不可判分（answer_type={getattr(sample, 'answer_type', '')!r}），"
            f"不该重采探针（traj={traj.id!r}）"
        )
    answers: List[str] = [str(t) for t in texts]
    # a_K 复用最终答案；K=0 时它是唯一一条读数（a_0 与 a_K 重合）
    answers.append(str(traj.final_answer or ""))
    expected = int(traj.n_calls) + 1
    if len(answers) != expected:  # 防御：调用方漏传 k 时早失败
        raise ProbeError(
            f"探针读数 {len(answers)} 与期望的 {expected}（n_calls+1）不一致（traj={traj.id!r}）"
        )

    scored = [_score_answer(grader, sample, a) for a in answers]
    traj.probes = [
        Probe(k=i, answer=answers[i], score=int(scored[i][0]), parse_ok=bool(scored[i][1]))
        for i in range(len(answers))
    ]
    traj.delta_k = staged_deltas([p.score for p in traj.probes])
    traj.delta_total = int(traj.probes[-1].score - traj.probes[0].score)
    # 探针字段统一按 PROBE_META_KEYS 写（_adopt 按同一份清单搬）
    traj.meta["probes_pending"] = False
    traj.meta["probe_source"] = PROBE_SOURCE
    traj.meta["probe_model"] = model
    traj.meta["probe_max_tokens"] = int(max_tokens)
    # 解析失败（空/纯空白）的下标如实留痕：它们记 0 分、不进 Δ
    traj.meta["probe_empty_ks"] = [i for i, (_, ok) in enumerate(scored) if not ok]
    # 之前若被标过"跳过"，此刻已经采完，旧标记必须清掉（否则下游会误以为仍不可用）
    traj.meta.pop("probe_skip_reason", None)


def _record(traj: Trajectory) -> Dict[str, Any]:
    """``out_path`` 里的一行：保留整条轨迹 + 探针，便于续跑时原样采纳。

    探针 meta 既在 ``traj.meta`` 里（轨迹自包含），也平铺一份到行首——
    老版本写出的行没有 ``probe_meta``，``_traj_from_record`` 会用行首字段兜底。
    """
    meta = dict(traj.meta or {})
    return {
        "id": str(traj.id),
        "sample_id": str(traj.sample_id),
        "traj": traj.to_json(),
        "probes": [p.to_json() for p in traj.probes],
        "delta_k": [int(v) for v in traj.delta_k],
        "delta_total": int(traj.delta_total),
        "probe_source": meta.get("probe_source"),
        "probe_meta": {k: meta[k] for k in PROBE_META_KEYS if k in meta},
    }


def _traj_from_record(rec: Mapping[str, Any]) -> Optional[Trajectory]:
    """从 ``out_path`` 的一行还原 Trajectory；结构不对就返回 None（不抛）。

    老版本写出的行把探针字段平铺在行首（``probe_source`` 等），新版还多一份
    ``probe_meta``；两者都往 ``meta`` 里补，缺的才补，不覆盖轨迹自带的 meta。
    """
    raw = rec.get("traj")
    if not isinstance(raw, Mapping):
        return None
    try:
        traj = Trajectory.from_json(dict(raw))
    except Exception:  # noqa: BLE001 - 坏行不该拖垮整批续跑
        return None
    meta = dict(traj.meta or {})
    flat = rec.get("probe_meta")
    if isinstance(flat, Mapping):
        for key in PROBE_META_KEYS:
            if key in flat and key not in meta:
                meta[key] = flat[key]
    if "probe_source" not in meta and rec.get("probe_source") is not None:
        meta["probe_source"] = rec.get("probe_source")
    traj.meta = meta
    return traj


def _adopt(dst: Trajectory, src: Trajectory) -> None:
    """把已采完的轨迹的探针读数据搬到内存对象上（steps 以 dst 自身为准）。

    探针 meta 一律按 ``PROBE_META_KEYS`` 搬运：``_finalize`` 写哪些字段，
    这里就搬哪些——否则"当场重采"与"从磁盘采纳"产出的轨迹 meta 字段集不同，
    下游 checkpoint 不可复现（缺陷 6）。``probes_pending`` 是完成度判定的
    入口字段，无论源里写没写都强制置 False。
    """
    dst.probes = [Probe(k=p.k, answer=p.answer, score=int(p.score), parse_ok=bool(p.parse_ok))
                  for p in src.probes]
    dst.delta_k = [int(v) for v in src.delta_k]
    dst.delta_total = int(src.delta_total)
    for key in PROBE_META_KEYS:
        if key in (src.meta or {}):
            dst.meta[key] = src.meta[key]
    dst.meta["probes_pending"] = False
    if not str(dst.meta.get("probe_source") or "").strip():
        dst.meta["probe_source"] = PROBE_SOURCE


class _ProbeSink:
    """探针结果的落盘器：**同 id 以最后一行为准**，且每行写完回读校验。

    为什么不能用 ``utils/io.JsonlWriter``：它按 id 去重，撞到已见 id 时
    ``write`` 直接返回 False。续跑时磁盘上往往已经有一条**同 id 但不完整**的旧行
    （比如老口径留下的 K=0、只有 2 个探针），``_finalize`` 在内存里修好了，
    但写入被静默吞掉——磁盘上那条旧行永远不会被纠正，而 ``n_completed``
    还报告成功（缺陷 1）。

    所以这里自己做"可覆盖"的追加写：

      - ``append``：一行 jsonl 追加 + flush + fsync，然后按字节增长量回读校验，
        确认这一行真的落盘了（挂载盘上偶发短写，不校验就会静默丢数据）；
      - ``close``：若本次出现同 id 多次写入，按"最后一条为准"原子压缩整份文件
        （写 tmp 再 ``os.replace``），把历史重复行清掉。

    坏行（半行/解析失败）不静默丢，搬到 ``<path>.bad``。
    """

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._order: Dict[str, int] = {}
        self._latest: Dict[str, Dict[str, Any]] = {}
        self._n_written = 0
        self._dirty = False
        self._rewrite_on_close = False
        self._f = None
        self._compact()

    # ---- 打开时压缩：同 id 只留最后一行
    def _compact(self) -> None:
        if not os.path.exists(self.path):
            return
        latest: Dict[str, Dict[str, Any]] = {}
        order: Dict[str, int] = {}
        n_line = 0
        bad: List[Dict[str, Any]] = []
        for lineno, line in enumerate(_read_lines(self.path), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad.append({"lineno": lineno, "raw": line[:4000]})
                continue
            if not isinstance(rec, Mapping) or rec.get("id") is None:
                bad.append({"lineno": lineno, "raw": line[:4000]})
                continue
            rid = str(rec["id"])
            n_line += 1
            latest[rid] = dict(rec)
            order.pop(rid, None)
            order[rid] = n_line
        if bad:
            with open(self.path + ".bad", "a", encoding="utf-8") as bf:
                for b in bad:
                    bf.write(json.dumps(b, ensure_ascii=False) + "\n")
        if not latest:
            if bad:
                _atomic_rewrite(self.path, [])
            return
        self._latest = latest
        self._order = order
        # 只有确实存在重复/坏行时才重写，正常续跑不动物理文件
        if len(latest) != n_line:
            _atomic_rewrite(self.path, [latest[rid] for rid in sorted(order, key=lambda k: order[k])])
        # 打开时就有重复行：本次即使不覆盖写，close 时也要压缩掉历史旧行
        self._rewrite_on_close = len(latest) != n_line

    # ---- 生命周期
    def __enter__(self) -> "_ProbeSink":
        # 二进制追加 + 可回读：'ab+' 把游标放在末尾，同时允许 seek/read 做落盘校验。
        # 必须是二进制：jsonl 里有中文，按字符数校验会与字节数对不上（短写漏检）。
        self._f = open(self.path, "ab+")
        self._f.seek(0, os.SEEK_END)
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        f, self._f = self._f, None
        if f is not None:
            f.flush()
            try:
                os.fsync(f.fileno())
            finally:
                f.close()
        # 本次进程内出现过同 id 覆盖，或者打开时文件里本就有重复行：原子压缩一次
        if self._dirty or self._rewrite_on_close:
            _atomic_rewrite(self.path, [self._latest[rid]
                                       for rid in sorted(self._order, key=lambda k: self._order[k])])
            self._dirty = False
            self._rewrite_on_close = False

    # ---- 写
    def write(self, rec: Mapping[str, Any]) -> None:
        """写入一行并校验落盘；同 id 的旧行在 ``close`` 时被压缩掉。"""
        if self._f is None:
            raise ProbeError("_ProbeSink 未打开（请用 with 语句）")
        rid = str(rec.get("id") or "")
        if not rid:
            raise ProbeError("探针记录缺少 id，无法落盘")
        line = json.dumps(dict(rec), ensure_ascii=False)
        payload = (line + "\n").encode("utf-8")

        f = self._f
        pos = f.tell()
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
        # 回读校验：按本次写入的字节数读回来比对，短写/丢行当场暴露
        f.seek(pos)
        back = f.read(len(payload))
        f.seek(0, os.SEEK_END)
        if back != payload:
            raise ProbeError(
                f"探针记录写入校验失败（id={rid!r}，期望 {len(payload)} 字节，"
                f"读回 {len(back)} 字节）: {self.path}"
            )

        self._n_written += 1
        if rid in self._latest:
            self._dirty = True
            self._order.pop(rid, None)
        self._latest[rid] = dict(rec)
        self._order[rid] = self._n_written

    @property
    def n_written(self) -> int:
        return self._n_written


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def _atomic_rewrite(path: str, records: Sequence[Mapping[str, Any]]) -> None:
    """整份重写：先写 tmp 再 ``os.replace``，中途被杀不会留半截文件。"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(dict(rec), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------- 主流程


def _existing_split(existing_ids: Any) -> Tuple[set, Dict[str, str]]:
    """拆出（要跳过的 custom_id 集合，能直接复用的 custom_id→答案）。

    允许两种入参：
      - 可迭代的 custom_id：只表示"已提交过"，答案未知，对应轨迹仍算未完成；
      - ``{custom_id: answer}`` / ``{custom_id: {answer: ...}}``：答案可复用，直接补齐。
    """
    skip: set = set()
    answers: Dict[str, str] = {}
    if existing_ids is None:
        return skip, answers
    if isinstance(existing_ids, Mapping):
        for cid, val in existing_ids.items():
            key = str(cid)
            skip.add(key)
            if isinstance(val, str):
                answers[key] = val
            elif isinstance(val, Mapping) and isinstance(val.get("answer"), str):
                answers[key] = str(val["answer"])
        return skip, answers
    for cid in existing_ids:
        skip.add(str(cid))
    return skip, answers


def collect_probes(trajs: Sequence[Trajectory], samples: Any, cfg: Any,
                   client: Any = None, *,
                   existing_ids: Any = None,
                   out_path: Optional[str] = None,
                   failure_sink: Optional[List[Dict[str, Any]]] = None,
                   sleep_fn: Optional[Any] = None,
                   now_fn: Optional[Any] = None) -> Dict[str, Any]:
    """按 a_0..a_{K-1} 逐前缀重问，补齐每条轨迹的 K+1 个探针。

    需要发请求的下标是 ``0..K-1``（a_K 复用 final_answer），共 K 条请求；K=0 的轨迹
    不发请求，a_0 直接由 final_answer 判分得到（``len(probes)==1==n_calls+1``）。

    Args:
        trajs:   待重采的轨迹（**就地**改写 probes/delta_k/delta_total/meta）。
        samples: Sample 序列或 ``{id: Sample}``。
        cfg:     全量配置（teacher 子项取模型/批处理参数，rl.grader 取容差）。
        client:  Batch 客户端；None 时用 ``local_teacher.local_client_from_cfg(cfg)``。
        existing_ids: 已提交过的 custom_id（可迭代），或 ``{custom_id: 答案}`` 映射用于复用。
        out_path: 给了就每完成一条轨迹立即追加一行 jsonl（崩溃可续）。
        failure_sink: 失败记录列表，元素形如
                  ``{'custom_id','traj_id','sample_id','stage','error'}``。
        sleep_fn / now_fn: 轮询用的时间函数（单测注入，默认 ``time.sleep`` / ``time.time``）。

    Returns:
        统计 dict（``n_trajs / n_requests / n_answers / n_completed / n_already /
        n_adopted / n_incomplete / n_not_gradable / n_failed / completed_ids /
        incomplete_ids / not_gradable_ids``）。
        已完成（``probes_complete``）的轨迹一律原样保留，不会被重复采样。
        ``n_not_gradable`` 是**样本**不可判分（long / 金标为空）的轨迹数，它们既不发
        请求也不写 ``probes_pending=False``，单独计数以免被误读成"采样失败"。
    """
    traj_list: List[Trajectory] = list(trajs)
    smap = _sample_map(samples)
    grader = DeterministicGrader(cfg)
    model = str(_opt(cfg, "model", "claude-opus-5"))
    sleep = sleep_fn or time.sleep
    now = now_fn or time.time
    interval = float(_opt(cfg, "poll_interval_s", 10.0))
    wait_cap = float(_opt(cfg, "max_wait_s", 86400.0))
    max_retries = int(_opt(cfg, "max_retries", 4))
    concurrency = max(1, int(_opt(cfg, "concurrency", 8)))
    max_req = int(_opt(cfg, "batch_max_requests", 100000))
    max_bytes = int(_opt(cfg, "batch_max_bytes", 268435456))

    sink: List[Dict[str, Any]] = failure_sink if failure_sink is not None else []
    n_failed = 0

    def _fail(traj_id: str, sample_id: str, stage: str, err: Any,
              custom_id: str = "") -> None:
        nonlocal n_failed
        n_failed += 1
        sink.append({
            "custom_id": custom_id,
            "traj_id": str(traj_id),
            "sample_id": str(sample_id),
            "stage": stage,
            "error": f"{type(err).__name__}: {err}",
        })

    # ---- 1. 续跑：先吃 out_path 里已经采完的轨迹（行是完整轨迹，直接采纳）
    n_adopted = 0
    if out_path and os.path.exists(out_path):
        done: Dict[str, Mapping[str, Any]] = {}
        for rec in iter_jsonl(out_path):
            tid = str(rec.get("id") or "")
            if tid:
                done[tid] = rec
        for traj in traj_list:
            if probes_complete(traj):
                continue
            rec = done.get(str(traj.id))
            stored = _traj_from_record(rec) if rec is not None else None
            if stored is not None and probes_complete(stored):
                _adopt(traj, stored)
                n_adopted += 1

    skip_ids, existing_answers = _existing_split(existing_ids)

    # ---- 2. 盘出真正要发的请求（k=0..K-1）
    needs: List[Tuple[Trajectory, Sample, int]] = []
    n_already = 0
    n_not_gradable = 0
    not_gradable_ids: List[str] = []
    not_gradable: set = set()      # 用对象身份记：两条轨迹可能撞同一个 str id
    rejected: set = set()               # custom_id 撞车、后来者的轨迹（保持待采）
    claimed: Dict[str, int] = {}        # custom_id -> 先占用它的轨迹对象 id()（一一对应）
    emitted: set = set()                # (custom_id, id(traj))：同一个对象被重复放进列表时去重
    for traj in traj_list:
        if probes_complete(traj):
            n_already += 1
            continue
        sample = _sample_for(traj, smap)
        if sample is None:
            _fail(traj.id, traj.sample_id, "sample",
                  ProbeError("找不到对应的 Sample，无法构造探针请求"))
            continue
        # 不可判分的样本（long / 金标为空）采了也进不了 Δ 通道：
        # 跳过重采，且**不**写 probes_pending=False（缺陷 8）
        if not grader.gradable(sample):
            n_not_gradable += 1
            not_gradable.add(id(traj))
            not_gradable_ids.append(str(traj.id))
            traj.meta["probe_skip_reason"] = "not_gradable"
            continue
        # 串行修一次图片路径：并发构造时同一轨迹的多个 k 会同时写 step.result.image，
        # 虽然 sanitize_images 幂等，但先在单线程里定下来更可控
        sanitize_images(traj, sample, _crops_dir(cfg))
        for k in range(int(traj.n_calls)):
            cid = probe_custom_id(traj.id, k)
            if cid in skip_ids:
                continue
            if (cid, id(traj)) in emitted:
                # 同一个轨迹对象在 traj_list 里出现了两次：第二次不再重复发请求。
                # 只用 custom_id 判重挡不住这种情形——owner 就是它自己（缺陷 9）。
                continue
            owner = claimed.get(cid)
            if owner is not None and owner != id(traj):
                # 同一个 custom_id 只能有一条轨迹。Batch API 收到重复 custom_id 会
                # **整批**报错，把同批无关的轨迹一起拖死；这里让后来者明确失败、
                # 保持待采，等它自己的 id 改掉后再采（缺陷 7）。
                # 用对象身份而不是 traj.id 判断：撞车的两条轨迹 id 恰恰是同一个。
                _fail(traj.id, sample.id, "duplicate_id",
                      ProbeError(f"custom_id {cid!r} 已被另一条轨迹占用"
                                 f"（traj.id 撞车），本条保持待采"), cid)
                rejected.add(id(traj))
                break
            claimed[cid] = id(traj)
            emitted.add((cid, id(traj)))
            needs.append((traj, sample, k))

    # ---- 3. 并发构造请求（读图 + base64 是 I/O 密集）
    built: List[Dict[str, Any]] = []
    owners: Dict[str, Tuple[Trajectory, Sample]] = {}
    if needs:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {
                ex.submit(build_probe_request, traj, sample, k, cfg): (traj, sample, k)
                for traj, sample, k in needs
            }
            for fut, (traj, sample, k) in futs.items():
                cid = probe_custom_id(traj.id, k)
                try:
                    req = fut.result()
                except Exception as e:  # noqa: BLE001 - 单条坏样本不拖垮整批
                    _fail(traj.id, sample.id, "build", e, cid)
                    continue
                built.append(req)
                owners[str(req["custom_id"])] = (traj, sample)

    # ---- 4. 提交 -> 轮询 -> 收集（失败的按轮重试）
    answers: Dict[str, str] = dict(existing_answers)
    last_err: Dict[str, str] = {}
    if built:
        if client is None:
            from .local_teacher import local_client_from_cfg

            client = local_client_from_cfg(cfg)
        remaining = list(built)
        for _round in range(max_retries + 1):
            if not remaining:
                break
            for chunk in _chunks(remaining, max_req, max_bytes):
                try:
                    batch = _create_with_retry(client, chunk, max_retries, sleep)
                    bid = str(_get(batch, "id"))
                    _poll(client, bid, interval, wait_cap, sleep, now)
                    ok, bad = _collect(client, bid)
                except Exception as e:  # noqa: BLE001 - 整块失败则整块进下一轮
                    for r in chunk:
                        last_err[str(r["custom_id"])] = f"batch: {type(e).__name__}: {e}"
                    continue
                for cid, msg in ok.items():
                    text = _message_text(msg)
                    if text is None:
                        last_err[str(cid)] = "result: 响应里没有 text block"
                    else:
                        answers[str(cid)] = text
                for cid, info in bad.items():
                    last_err[str(cid)] = f"result: {info.get('type')}: {info.get('error')}"
            remaining = [r for r in remaining if str(r["custom_id"]) not in answers]
        for r in remaining:
            cid = str(r["custom_id"])
            traj, sample = owners[cid]
            _fail(traj.id, sample.id, "result",
                  ProbeError(last_err.get(cid, "unknown")), cid)

    # ---- 5. 写回轨迹 + 逐条落盘
    n_completed = 0
    n_incomplete = 0
    completed_ids: List[str] = []
    incomplete_ids: List[str] = []
    effective_max_tokens = probe_max_tokens(cfg)

    sink_out: Optional[_ProbeSink] = None
    try:
        if out_path:
            sink_out = _ProbeSink(out_path)
            sink_out.__enter__()
        for traj in traj_list:
            if probes_complete(traj):
                continue
            if id(traj) in not_gradable:
                # 不计入 n_incomplete：它的"未完成"是样本属性决定的，不是采样失败
                continue
            if id(traj) in rejected:
                n_incomplete += 1
                incomplete_ids.append(str(traj.id))
                continue
            sample = _sample_for(traj, smap)
            if sample is None:
                n_incomplete += 1
                incomplete_ids.append(str(traj.id))
                continue
            texts: List[str] = []
            missing = False
            for k in range(int(traj.n_calls)):
                cid = probe_custom_id(traj.id, k)
                if cid not in answers:
                    missing = True
                    break
                texts.append(answers[cid])
            if missing:
                n_incomplete += 1
                incomplete_ids.append(str(traj.id))
                continue
            try:
                _finalize(traj, sample, texts, grader, model, effective_max_tokens)
            except Exception as e:  # noqa: BLE001 - 单条写回失败不拖垮整批
                _fail(traj.id, sample.id, "finalize", e)
                n_incomplete += 1
                incomplete_ids.append(str(traj.id))
                continue
            n_completed += 1
            completed_ids.append(str(traj.id))
            if sink_out is not None:
                try:
                    sink_out.write(_record(traj))
                except Exception as e:  # noqa: BLE001 - 落盘失败不能吞掉整批
                    _fail(traj.id, sample.id, "sink", e)
    finally:
        if sink_out is not None:
            sink_out.close()

    return {
        "n_trajs": len(traj_list),
        "n_requests": len(built),
        "n_answers": len(answers) - len(existing_answers),
        "n_completed": n_completed,
        "n_already": n_already,
        "n_adopted": n_adopted,
        "n_incomplete": n_incomplete,
        "n_not_gradable": n_not_gradable,
        "n_failed": n_failed,
        "completed_ids": completed_ids,
        "incomplete_ids": incomplete_ids,
        "not_gradable_ids": not_gradable_ids,
    }
