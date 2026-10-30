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

因此需要发请求的下标只有 ``0..max(K,1)-1``：K>=1 时每条轨迹发 K 条；K=0（无工具
调用）时发 k=0 一条拿到 a_0，a_1 直接复用 ``final_answer``，此时
``len(probes)==2 == K+1`` 同样满足不变量。

判分一律走 ``rl/grader.py:DeterministicGrader.score``（同输入同输出、零成本）；
空答案/解析失败由 grader 自然返回 -1，本模块不额外造分。

工程约定
--------
1. **幂等**：``custom_id = probe-<消毒后的 traj.id>-k<k>``，``existing_ids`` 里出现过的
   一律不再提交；``out_path`` 里已写全的轨迹在续跑时直接采纳，不再发请求。
2. **可续跑**：``out_path`` 给了就每采完一条轨迹立即追加一行（``utils/io.JsonlWriter``），
   进程被杀最多丢一条。
3. **失败不抛**：单条请求/单条轨迹的失败记进 ``failure_sink``，该轨迹保持
   ``probes_pending=True`` 留给下一轮，不污染同批其他轨迹。
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..rl.advantage import staged_deltas
from ..rl.grader import DeterministicGrader
from ..utils.io import JsonlWriter, iter_jsonl
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
    "ProbeError",
    "build_probe_instruction",
    "build_probe_request",
    "collect_probes",
    "pending_probe_keys",
    "probe_custom_id",
    "probes_complete",
]

#: 探针只需要一个短答案，64 token 足够覆盖 closed/short/measure 三类可判分题
PROBE_MAX_TOKENS: int = 64
#: Batch API 只接受这些字符，且 custom_id 长度 <= 64
_CUSTOM_ID_BAD = re.compile(r"[^A-Za-z0-9_-]")
_MAX_CUSTOM_ID: int = 64
#: 探针结果的来源标记（写回 meta，便于下游区分"重采过"与"教师占位"）
PROBE_SOURCE: str = "recollected"


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


def _prefix_images(traj: Trajectory, k: int) -> List[str]:
    """a_k 该看到的前缀图：前 k 次工具调用里**非空**的返回图路径。

    语义上就是"第 1..k 次调用已经执行完"，所以只扫前 k 个工具轮；某轮没有图
    （measure 只回数值）就自然少一张，而不是拿后面轮次的图来凑数——那会让 a_1
    提前看到第 2 次调用的观测，探针语义就失效了。
    """
    out: List[str] = []
    for st in _tool_turns(traj)[:max(0, int(k))]:
        img = getattr(st.result, "image", None) if st.result is not None else None
        if img:
            out.append(str(img))
    return out


def _message_text(msg: Any) -> Optional[str]:
    """把一条 Message 里所有 text block 拼起来；一个 text block 都没有时返回 None。

    返回 None 表示"这次请求没拿到答案"（拒答 / 空内容），调用方按失败处理并重试；
    返回空串则是模型确实给了空答案，按 grader 自然判 -1。
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


def build_probe_instruction(sample: Sample, k: int) -> str:
    """探针请求的文本指令。

    刻意**不**用 teacher 的 JSON schema 指令：探针要的是自由文本答案，由
    ``DeterministicGrader.parse`` 抽最后一行。也刻意只给 user 消息——该代模型禁止 prefill。
    """
    kk = max(0, int(k))
    if kk == 0:
        crop_note = (
            "Attached: the ORIGINAL image only; no tool call has been executed yet."
        )
        role_lines = "  IMAGE 1 = ORIGINAL"
    else:
        role_lines = "  IMAGE 1 = ORIGINAL\n" + "\n".join(
            f"  IMAGE {i + 2} = CROP_{i + 1} (returned by tool call {i + 1})" for i in range(kk)
        )
        crop_note = (
            f"Attached: the ORIGINAL image plus the {kk} crop image(s) returned by the "
            f"first {kk} tool call(s), i.e. exactly what a model would have seen after "
            f"executing those calls."
        )
    return f"""You are a senior radiologist answering a question about a medical image.

Images attached, in order:
{role_lines}

{crop_note}

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
        cfg:    teacher 子配置（模型名 / 长边上限 / 图片数上限）。

    Returns:
        ``{"custom_id": str, "params": {...}}``，``params`` 可直接作为 Batch 请求体，
        也可 ``client.messages.create(**params)``。**只有 user 消息**（禁止 prefill）。
    """
    n_calls = int(traj.n_calls)
    kk = int(k)
    # 需要发请求的下标是 0..max(K,1)-1：K>=1 时 a_K 复用 final_answer 不必再问；
    # K=0 时 a_0 与 a_K 重合，按约定仍然发一条 k=0 的请求拿到「只看原图」的读数。
    if kk < 0 or kk >= max(1, n_calls):
        raise ProbeError(
            f"k 必须在 0..{max(1, n_calls) - 1} 之间（k>=1 的 a_K 直接复用 traj.final_answer，"
            f"不发请求），收到 k={kk}（traj={traj.id!r}, n_calls={n_calls}）"
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
    blocks.append({"type": "text", "text": build_probe_instruction(sample, kk)})

    params: Dict[str, Any] = {
        "model": str(_opt(cfg, "model", "claude-opus-5")),
        "max_tokens": PROBE_MAX_TOKENS,
        # 只有 user 一条消息：禁止 message prefill
        "messages": [{"role": "user", "content": blocks}],
    }
    return {"custom_id": probe_custom_id(traj.id, kk), "params": params}


# ---------------------------------------------------------------- 完成度判定


def probes_complete(traj: Trajectory) -> bool:
    """该轨迹的探针是否已经可用来训练。

    三项同时成立才算完成：长度 = K+1、``probes_pending`` 不为真、``delta_k`` 与
    ``staged_deltas`` 一致。任一不满足就视为"待重采"——宁可贵一点重采一次，
    也不能把占位探针送进 OGAR。
    """
    try:
        if traj.meta.get("probes_pending"):
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

    只列真正需要发请求的下标：K>=1 时是 ``0..K-1``（a_K 复用 final_answer），K=0 时
    仍要发一条 k=0 拿到「只看原图」的读数。找不到对应 Sample 的轨迹直接跳过（没有
    原图/问题就无法构造请求）。``cfg`` 为接口对称保留，当前不参与判定；调用方通常用
    本函数的结果与上一轮的 ``existing_ids`` 做差再提交。
    """
    del cfg  # 接口对称：暂不参与判定
    smap = _sample_map(samples)
    out: List[Tuple[str, str, int]] = []
    for traj in trajs:
        if probes_complete(traj):
            continue
        sample = _sample_for(traj, smap)
        if sample is None:
            continue
        for k in range(max(1, int(traj.n_calls))):
            out.append((str(traj.id), str(sample.id), k))
    return out


# ---------------------------------------------------------------- 写回


def _finalize(traj: Trajectory, sample: Sample, texts: Sequence[str],
              grader: DeterministicGrader, model: str) -> None:
    """把 a_0..a_{K-1} 的回答与 a_K=final_answer 一起判分并写回轨迹。"""
    answers: List[str] = [str(t) for t in texts]
    answers.append(str(traj.final_answer or ""))  # a_K 复用最终答案，不再发请求
    scores: List[int] = [int(grader.score(sample, a)) for a in answers]
    if len(scores) != int(traj.n_calls) + 1:  # 防御：调用方漏传 k 时早失败
        raise ProbeError(
            f"探针读数 {len(scores)} 与 n_calls+1={int(traj.n_calls) + 1} 不一致（traj={traj.id!r}）"
        )

    traj.probes = [
        Probe(k=i, answer=answers[i], score=int(scores[i]), parse_ok=True)
        for i in range(len(answers))
    ]
    traj.delta_k = staged_deltas([p.score for p in traj.probes])
    traj.delta_total = int(traj.probes[-1].score - traj.probes[0].score)
    traj.meta["probes_pending"] = False
    traj.meta["probe_source"] = PROBE_SOURCE
    traj.meta["probe_model"] = model
    traj.meta["probe_max_tokens"] = PROBE_MAX_TOKENS
    # 空答案不会被丢弃（grader 自然判 -1），但要留痕，便于复盘是模型没答还是答错
    traj.meta["probe_empty_ks"] = [i for i, a in enumerate(answers) if not a.strip()]


def _record(traj: Trajectory) -> Dict[str, Any]:
    """``out_path`` 里的一行：保留整条轨迹 + 探针，便于续跑时原样采纳。"""
    return {
        "id": str(traj.id),
        "sample_id": str(traj.sample_id),
        "traj": traj.to_json(),
        "probes": [p.to_json() for p in traj.probes],
        "delta_k": [int(v) for v in traj.delta_k],
        "delta_total": int(traj.delta_total),
        "probe_source": traj.meta.get("probe_source"),
    }


def _traj_from_record(rec: Mapping[str, Any]) -> Optional[Trajectory]:
    """从 ``out_path`` 的一行还原 Trajectory；结构不对就返回 None（不抛）。"""
    raw = rec.get("traj")
    if not isinstance(raw, Mapping):
        return None
    try:
        return Trajectory.from_json(dict(raw))
    except Exception:  # noqa: BLE001 - 坏行不该拖垮整批续跑
        return None


def _adopt(dst: Trajectory, src: Trajectory) -> None:
    """把已采完的轨迹的探针读数据搬到内存对象上（steps 以 dst 自身为准）。"""
    dst.probes = [Probe(k=p.k, answer=p.answer, score=int(p.score), parse_ok=bool(p.parse_ok))
                  for p in src.probes]
    dst.delta_k = [int(v) for v in src.delta_k]
    dst.delta_total = int(src.delta_total)
    dst.meta["probes_pending"] = False
    dst.meta["probe_source"] = PROBE_SOURCE


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

    需要发请求的下标是 ``0..max(K,1)-1``：K>=1 时 a_K 复用 final_answer，共 K 条请求；
    K=0 时发一条 k=0 的请求，a_1 复用 final_answer。

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
        n_adopted / n_incomplete / n_failed / completed_ids / incomplete_ids``）。
        已完成（``probes_complete``）的轨迹一律原样保留，不会被重复采样。
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
    for traj in traj_list:
        if probes_complete(traj):
            n_already += 1
            continue
        sample = _sample_for(traj, smap)
        if sample is None:
            _fail(traj.id, traj.sample_id, "sample",
                  ProbeError("找不到对应的 Sample，无法构造探针请求"))
            continue
        # 串行修一次图片路径：并发构造时同一轨迹的多个 k 会同时写 step.result.image，
        # 虽然 sanitize_images 幂等，但先在单线程里定下来更可控
        sanitize_images(traj, sample, _crops_dir(cfg))
        for k in range(max(1, int(traj.n_calls))):
            cid = probe_custom_id(traj.id, k)
            if cid in skip_ids:
                continue
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

    writer: Optional[JsonlWriter] = None
    with ExitStack() as stack:
        if out_path:
            writer = stack.enter_context(JsonlWriter(out_path, id_key="id", resume=True))
        for traj in traj_list:
            if probes_complete(traj):
                continue
            sample = _sample_for(traj, smap)
            if sample is None:
                n_incomplete += 1
                incomplete_ids.append(str(traj.id))
                continue
            texts: List[str] = []
            missing = False
            for k in range(max(1, int(traj.n_calls))):
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
                _finalize(traj, sample, texts, grader, model)
            except Exception as e:  # noqa: BLE001 - 单条写回失败不拖垮整批
                _fail(traj.id, sample.id, "finalize", e)
                n_incomplete += 1
                incomplete_ids.append(str(traj.id))
                continue
            n_completed += 1
            completed_ids.append(str(traj.id))
            if writer is not None:
                try:
                    writer.write(_record(traj))
                except Exception as e:  # noqa: BLE001 - 落盘失败不能吞掉整批
                    _fail(traj.id, sample.id, "sink", e)

    return {
        "n_trajs": len(traj_list),
        "n_requests": len(built),
        "n_answers": len(answers) - len(existing_answers),
        "n_completed": n_completed,
        "n_already": n_already,
        "n_adopted": n_adopted,
        "n_incomplete": n_incomplete,
        "n_failed": n_failed,
        "completed_ids": completed_ids,
        "incomplete_ids": incomplete_ids,
    }
