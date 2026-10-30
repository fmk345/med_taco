"""断点续传的 jsonl 读写。

设计要点（对应 PROPOSAL 的工程偏好）：
  - 单文件输出、保留全部原始字段；
  - 续跑不重刷：写入前先扫出已有 id 集合，skip 命中项；
  - 追加写、每行 flush，进程被杀最多丢一行；
  - 坏行（半行/解析失败）不静默丢，记到 <path>.bad 并计数。
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Callable, Dict, Iterable, Iterator, Optional, Set


def iter_jsonl(path: str, strict: bool = False) -> Iterator[dict]:
    """流式读取。strict=True 时坏行直接抛，否则跳过并记到 .bad。"""
    if not os.path.exists(path):
        return
    bad_path = path + ".bad"
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if strict:
                    raise
                with open(bad_path, "a", encoding="utf-8") as bf:
                    bf.write(json.dumps({"lineno": lineno, "raw": line[:4000]}, ensure_ascii=False) + "\n")


def seen_ids(path: str, id_key: str = "id") -> Set[str]:
    """扫已有输出的 id 集合，用于续跑跳过。只解析首层 id，避免全量反序列化。"""
    out: Set[str] = set()
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = obj.get(id_key)
            if v is not None:
                out.add(str(v))
    return out


class JsonlWriter:
    """追加写 + 续跑跳过 + 原子性尽可能好。

    with JsonlWriter(path) as w:
        for x in items:
            if w.skip(x["id"]): continue
            w.write(x)
    """

    def __init__(self, path: str, id_key: str = "id", resume: bool = True,
                 flush_every: int = 1, buffer: bool = False):
        self.path = path
        self.id_key = id_key
        self.resume = resume
        self.flush_every = max(1, flush_every)
        self._n = 0
        self._seen: Set[str] = seen_ids(path, id_key) if resume else set()
        self._f = None
        self._buffer = buffer
        self._buf = []
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def __enter__(self) -> "JsonlWriter":
        # resume=False 的语义是"重建这个文件"：不加载已见集合还不够，
        # 否则新内容会被追加到旧内容后面（真实数据混进 mock 就是这么发生的）。
        self._f = open(self.path, "w" if not self.resume else "a", encoding="utf-8")
        return self

    def __exit__(self, *exc) -> None:
        self.flush()
        if self._f:
            self._f.close()
            self._f = None

    @property
    def n_skipped(self) -> int:
        return len(self._seen)

    def skip(self, record_id: str) -> bool:
        return str(record_id) in self._seen

    def write(self, obj) -> bool:
        """返回 True 表示真的写入（未被跳过）。obj 可以是 dict 或有 to_json 的对象。"""
        d = obj.to_json() if hasattr(obj, "to_json") else obj
        rid = str(d[self.id_key])
        if rid in self._seen:
            return False
        line = json.dumps(d, ensure_ascii=False)
        if self._buffer:
            self._buf.append(line)
        else:
            self._f.write(line + "\n")
        self._seen.add(rid)
        self._n += 1
        if self._n % self.flush_every == 0:
            self.flush()
        return True

    def flush(self) -> None:
        if self._buf:
            self._f.write("\n".join(self._buf) + "\n")
            self._buf.clear()
        if self._f:
            self._f.flush()
            os.fsync(self._f.fileno())


def atomic_write_json(path: str, obj) -> None:
    """先写 tmp 再 rename，避免中途被杀留下半截 json（cluster 上很常见）。"""
    d = obj.to_json() if hasattr(obj, "to_json") else obj
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    shutil.move(tmp, path)


def shard_by_id(records: Iterable[dict], n_shards: int, id_key: str = "id") -> Dict[int, list]:
    """稳定分片（按 id 的 crc32），保证同一 id 永远落在同一片，便于多机并行续跑。"""
    import zlib

    out: Dict[int, list] = {i: [] for i in range(n_shards)}
    for r in records:
        out[zlib.crc32(str(r[id_key]).encode()) % n_shards].append(r)
    return out


def map_parallel(items: list, fn: Callable, workers: int = 8,
                 desc: str = "", on_error: Optional[Callable] = None) -> list:
    """ThreadPoolExecutor 并发 map，保序返回；失败项返回 None 并回调。

    不引入 tqdm 依赖，进度用 log_every 打印。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: list = [None] * len(items)
    if not items:
        return results
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            done += 1
            try:
                results[i] = fut.result()
            except Exception as e:  # noqa: BLE001 - 单条失败不应中断整批
                if on_error:
                    on_error(items[i], e)
                results[i] = None
            if done % 100 == 0:
                print(f"[{desc}] {done}/{len(items)}", flush=True)
    return results
