"""本地 Qwen3-VL-8B 作为 teacher 的批处理客户端。

为什么需要它：``teacher.run_batch`` 只要一个实现了
``client.messages.batches.{create,retrieve,results}`` 的对象，它跟"是不是
Anthropic"无关。这里用 transformers 在本机 8 张卡上把同一个协议实现出来，
于是 ``pipeline.py --stage synth`` 在**没有 API 额度**的情况下也能产出**真轨迹**
（区别于 ``MockBatchClient``：这里没有 mock 标记，答案是真模型给的）。

与 Anthropic 的差异（都是刻意为之，写在返回的 meta 里）：
  1. **没有 json_schema 约束**：Qwen3-VL 只吃文本提示。schema 以文本形式追加到
     指令末尾，靠 ``teacher._extract_json`` 的容错（裸 JSON / ```json 围栏）兜底。
  2. **没有 Batch 折扣，也没有异步**：``create`` 里同步跑完（8 进程 = 8 卡），
     ``retrieve`` 恒返回 ``ended``。对下游是同步语义，所以 ``run_batch`` 的
     轮询会立刻返回。
  3. **贪婪解码**：``temperature`` 传 0 也不一定稳，直接 ``do_sample=False``，
     保证同一请求重跑结果一致——validate 阶段的 no_tool_ablation 要拿同一张图问第二遍。

用法::

    from medtaco.synth.local_teacher import local_client_from_cfg
    client = local_client_from_cfg(cfg)          # 8 卡，每卡一个进程
    trajs = run_batch(samples, cfg, client=client)
"""
from __future__ import annotations

import base64
import io
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

from .teacher import TeacherError, _Obj, custom_id_of

__all__ = ["LocalBatchClient", "local_client_from_cfg", "DEFAULT_MODEL_PATH"]

#: 本机缓存的权重；可用环境变量 MEDTACO_LOCAL_MODEL 覆盖。
#: 注意不能用 ``~``：容器里 $HOME 是 /home/hadoop-speech，权重在 workspace 下。
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.environ.get(
    "MEDTACO_LOCAL_MODEL",
    os.path.normpath(os.path.join(_HERE, "..", "..", "..", "models", "Qwen3-VL-8B-Instruct")),
)

# ---- worker 进程内的全局状态（每进程一份模型）
_W: Dict[str, Any] = {}
_COUNTER: Any = None


def _init_worker(model_path: str, counter: Any, device_prefix: str) -> None:
    """在子进程里加载模型到一块独立的卡上。

    父进程**绝不能**先初始化 CUDA，否则 fork 出来的子进程会共享上下文直接崩。
    这里用 fork + ``torch.cuda.set_device``，每个子进程独占 ``cuda:<idx>``。
    """
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    with counter.get_lock():
        idx = counter.value
        counter.value += 1
    dev = f"{device_prefix}:{idx}"
    torch.cuda.set_device(idx)

    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map={"": dev},
    ).eval()
    proc = AutoProcessor.from_pretrained(model_path)
    _W["model"] = model
    _W["proc"] = proc
    _W["device"] = dev
    print(f"[local-teacher] worker {idx} 加载完成 {time.time() - t0:.1f}s -> {dev}", flush=True)


def _blocks_to_messages(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把 Anthropic 的 messages 内容块翻译成 Qwen3-VL 的 chat 格式。"""
    from PIL import Image

    out: List[Dict[str, Any]] = []
    for msg in params.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, str):
            out.append({"role": msg.get("role", "user"), "content": [{"type": "text", "text": content}]})
            continue
        parts: List[Dict[str, Any]] = []
        for b in content or []:
            t = b.get("type")
            if t == "text":
                parts.append({"type": "text", "text": b.get("text", "")})
            elif t == "image":
                src = b.get("source") or {}
                if src.get("type") != "base64":
                    raise TeacherError(f"本地 teacher 只支持 base64 图片，收到 {src.get('type')}")
                raw = base64.b64decode(src.get("data", ""))
                parts.append({"type": "image", "image": Image.open(io.BytesIO(raw)).convert("RGB")})
            else:
                raise TeacherError(f"不认识的内容块: {t}")
        out.append({"role": msg.get("role", "user"), "content": parts})
    return out


def _schema_suffix(params: Dict[str, Any]) -> str:
    """把 json_schema 变成文本约束追加到最后一个文本块。"""
    try:
        schema = params["output_config"]["format"]["schema"]
    except (KeyError, TypeError):
        return ""
    return (
        "\n\nOUTPUT FORMAT — return exactly ONE JSON object and nothing else "
        "(no markdown fence, no prose before or after). It must validate against this JSON Schema:\n"
        + json.dumps(schema, ensure_ascii=False)
    )


def _run_one(request: Dict[str, Any]) -> Dict[str, Any]:
    """子进程侧：跑一条请求，返回 Anthropic 风格的 message 载荷。"""
    import torch

    cid = str(request.get("custom_id"))
    params = request.get("params") or {}
    max_new = int(params.get("max_tokens") or 3072)

    try:
        msgs = _blocks_to_messages(params)
        suffix = _schema_suffix(params)
        if suffix and msgs:
            last = msgs[-1]["content"]
            # 追加到最后一条消息的尾部文本块
            if last and last[-1].get("type") == "text":
                last[-1]["text"] = last[-1]["text"] + suffix
            else:
                last.append({"type": "text", "text": suffix})

        proc = _W["proc"]
        model = _W["model"]
        dev = _W["device"]
        inputs = proc.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(dev)
        n_in = int(inputs["input_ids"].shape[1])
        t0 = time.time()
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        new = gen[:, n_in:]
        text = proc.batch_decode(new, skip_special_tokens=True)[0]
        stop = "max_tokens" if new.shape[1] >= max_new else "end_turn"
        return {
            "custom_id": cid,
            "type": "succeeded",
            "text": text,
            "stop_reason": stop,
            "input_tokens": n_in,
            "output_tokens": int(new.shape[1]),
            "elapsed_s": round(time.time() - t0, 2),
        }
    except Exception as e:  # noqa: BLE001 - 单条失败交给 run_batch 重试/记录
        return {"custom_id": cid, "type": "errored",
                "error": f"{type(e).__name__}: {e}"}


class LocalBatchClient:
    """进程池封装的本地 teacher。接口与 ``anthropic.Anthropic()`` 的 batches 一致。"""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, n_workers: int = 8,
                 device_prefix: str = "cuda") -> None:
        if not os.path.isdir(model_path):
            raise TeacherError(f"本地模型目录不存在: {model_path}")
        self.model_path = model_path
        self.n_workers = max(1, int(n_workers))
        self.device_prefix = device_prefix
        self._pool: Optional[ProcessPoolExecutor] = None
        self._counter = None
        self._batches: Dict[str, List[Dict[str, Any]]] = {}
        self._results: Dict[str, List[Dict[str, Any]]] = {}
        self._n = 0
        self.messages = _Obj(batches=_Obj(
            create=self._create, retrieve=self._retrieve, results=self._results_iter))

    # ---- 进程池
    def _ensure_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            ctx = mp.get_context("fork")
            self._counter = ctx.Value("i", 0)
            self._pool = ProcessPoolExecutor(
                max_workers=self.n_workers, mp_context=ctx,
                initializer=_init_worker,
                initargs=(self.model_path, self._counter, self.device_prefix),
            )
            # 触发全部 worker 启动并加载模型，否则第一批任务的延迟会掩盖加载错误
            list(self._pool.map(_warmup, [0] * self.n_workers))
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def __enter__(self) -> "LocalBatchClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- Batch 协议
    def _create(self, requests: Sequence[Dict[str, Any]], **_: Any) -> Any:
        self._n += 1
        bid = f"local_batch_{self._n}"
        reqs = list(requests)
        self._batches[bid] = reqs

        pool = self._ensure_pool()
        t0 = time.time()
        out: List[Dict[str, Any]] = []
        # 不能用 map：单条异常必须变成 errored 而不是整批炸掉
        futs = [pool.submit(_run_one, r) for r in reqs]
        for f in futs:
            try:
                out.append(f.result())
            except Exception as e:  # noqa: BLE001
                out.append({"custom_id": "", "type": "errored", "error": f"worker 崩溃: {e}"})
        self._results[bid] = out
        ok = sum(1 for r in out if r["type"] == "succeeded")
        print(f"[local-teacher] {bid}: {ok}/{len(out)} 成功，用时 {time.time() - t0:.1f}s", flush=True)
        return _Obj(id=bid, processing_status="ended")

    def _retrieve(self, batch_id: str, **_: Any) -> Any:
        if batch_id not in self._batches:
            raise TeacherError(f"未知的本地 batch: {batch_id}")
        return _Obj(id=batch_id, processing_status="ended")

    def _results_iter(self, batch_id: str, **_: Any):
        for r in self._results.get(batch_id, []):
            cid = r.get("custom_id", "")
            if r["type"] == "succeeded":
                msg = _Obj(
                    model=os.path.basename(self.model_path.rstrip("/")),
                    stop_reason=r.get("stop_reason", "end_turn"),
                    content=[_Obj(type="text", text=r["text"])],
                    usage=_Obj(input_tokens=r.get("input_tokens", 0),
                               output_tokens=r.get("output_tokens", 0)),
                )
                yield _Obj(custom_id=cid, result=_Obj(type="succeeded", message=msg))
            else:
                yield _Obj(custom_id=cid, result=_Obj(
                    type="errored",
                    error=_Obj(type="local_generation_error", message=r.get("error", ""))))


def _warmup(_: Any) -> str:
    """空任务：确保每个 worker 都真正走完 initializer。"""
    return _W.get("device", "?")


def local_client_from_cfg(cfg: Any) -> LocalBatchClient:
    """从配置构造。默认 8 卡；``teacher.local_workers`` 可覆盖。"""
    path = str(getattr(getattr(cfg, "teacher", None), "local_model", "") or DEFAULT_MODEL_PATH)
    n = int(getattr(getattr(cfg, "teacher", None), "local_workers", 0) or 8)
    return LocalBatchClient(model_path=path, n_workers=n)
