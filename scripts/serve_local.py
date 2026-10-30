#!/usr/bin/env python3
"""极简 OpenAI 兼容端点，专门服务 ``pipeline.py --stage validate``。

为什么不直接用 vLLM：本机 vLLM 0.8.5 的模型注册表里没有 Qwen3-VL（会直接
``ValueError: Qwen3VLForConditionalGeneration is not supported``），而 transformers
4.57 已经支持。validate 只需要 ``POST /v1/chat/completions`` 一个路径、
一次一问、低并发，所以没必要为此升 vLLM——直接 transformers 起个小服务。

接口只实现 ask_fn 用到的字段：
  request : {"model", "messages":[{"role":"user","content":[{"type":"text","text":..},
                                                       {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,.."}}]}],
             "temperature", "max_tokens"}
  response: {"choices":[{"message":{"role":"assistant","content": "..."}}]}

用法::

    python scripts/serve_local.py --port 18422 --gpus 2
    MEDTACO_LOCAL_ENDPOINT=http://127.0.0.1:18422/v1/chat/completions \
      python scripts/pipeline.py --stage validate
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medtaco.synth.local_teacher import DEFAULT_MODEL_PATH  # noqa: E402

_MODELS: List[Any] = []   # [(model, processor, device)]
_LOCK = threading.Semaphore(1)


def _load(model_path: str, n_gpus: int) -> None:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    for i in range(max(1, n_gpus)):
        dev = f"cuda:{i}"
        m = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map={"": dev}).eval()
        p = AutoProcessor.from_pretrained(model_path)
        _MODELS.append((m, p, dev))
        print(f"[serve] 载入 {dev}", flush=True)


def _decode_data_url(url: str):
    from PIL import Image

    if not url.startswith("data:"):
        raise ValueError("只支持 data: URL")
    b64 = url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _to_chat(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        parts: List[Dict[str, Any]] = []
        if isinstance(content, str):
            parts = [{"type": "text", "text": content}]
        else:
            for b in content or []:
                if b.get("type") == "text":
                    parts.append({"type": "text", "text": b.get("text", "")})
                elif b.get("type") == "image_url":
                    url = b.get("image_url", {}).get("url", "")
                    parts.append({"type": "image", "image": _decode_data_url(url)})
        out.append({"role": msg.get("role", "user"), "content": parts})
    return out


def _generate(body: Dict[str, Any]) -> str:
    import torch

    with _LOCK:  # 单卡串行：validate 并发不高，瓶颈在显存而不是吞吐
        model, proc, dev = _MODELS[0]
        chat = _to_chat(body.get("messages") or [])
        inputs = proc.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(dev)
        max_new = int(body.get("max_tokens") or 256)
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        new = gen[:, inputs["input_ids"].shape[1]:]
        return proc.batch_decode(new, skip_special_tokens=True)[0].strip()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        if self.path.rstrip("/") in ("/health", "/v1/models"):
            payload = {"ok": True, "models": [os.path.basename(DEFAULT_MODEL_PATH)]}
        else:
            self.send_error(404)
            return
        self._json(200, payload)

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self.send_error(404)
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            text = _generate(body)
        except Exception as e:  # noqa: BLE001
            self._json(500, {"error": {"message": f"{type(e).__name__}: {e}"}})
            return
        self._json(200, {
            "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
        })

    def _json(self, code: int, obj: Dict[str, Any]) -> None:
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt: str, *a: Any) -> None:
        sys.stderr.write("[serve] " + fmt % a + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--port", type=int, default=18422)
    ap.add_argument("--gpus", type=int, default=1, help="载入几张卡；当前实现只用第一张")
    args = ap.parse_args()
    if not os.path.isdir(args.model):
        print(f"模型目录不存在: {args.model}", file=sys.stderr)
        return 2
    _load(args.model, args.gpus)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), _Handler)
    print(f"[serve] http://127.0.0.1:{args.port}/v1/chat/completions", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
