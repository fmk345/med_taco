"""把本地 ``data/`` 上传到 HuggingFace Dataset 仓库。

设计取舍：

* **排除 ``data/ckpt``**（66G 的 50 步冒烟检查点 + 8B 权重分片）。它不是数据资产，
  传上去既慢又没人用；需要时另行上传到 Model 仓库更合适。
* 用小文件豁免清单，把 ``*.lock``（HF datasets 自己的缓存锁，1071 个）、
  ``*.incomplete``（下载中断残留）、``*.bak`` 排除。传上去只会污染仓库。
* 用 ``upload_large_folder`` 而不是 ``upload_folder``：1.5 万个文件，
  前者会分块、并行、**断点续传**（中断后重跑只补差异），后者一次性 commit，
  传到一半断了要从头来。
* 凭据只从环境 ``HF_TOKEN`` 或 HF 官方缓存（``~/.cache/huggingface/token``）读，
  **不接受命令行参数**——避免 token 落到 shell history 和进程列表里。

用法::

    export https_proxy=http://10.217.148.40:8080
    python scripts/upload_hf.py --dry-run          # 只统计，不联网上传
    python scripts/upload_hf.py                    # 真传
    python scripts/upload_hf.py --with-ckpt        # 连 66G 检查点一起传
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ID_DEFAULT = "fmk99/med_taco"

#: 永远不传的路径/文件名模式（相对 data/ 根）
IGNORE = [
    "ckpt/**",            # 66G 冒烟检查点
    "ckpt_smoke/**",
    "**/*.lock",          # huggingface datasets 缓存锁
    "**/*.incomplete",    # 下载中断残留
    "**/*.bak",           # backfill_answers.py 留下的备份
    "**/__pycache__/**",
    "**/.ipynb_checkpoints/**",
]


def _human(n: int) -> str:
    x = float(n)
    for unit in ("B", "K", "M", "G", "T"):
        if x < 1024:
            return f"{x:.1f}{unit}"
        x /= 1024
    return f"{x:.1f}P"


def survey(root: Path, ignore: list[str]) -> dict:
    """统计将要上传的文件数与字节数（不联网）。

    这里用 fnmatch 判断，**不引入 huggingface_hub**，所以 ``--dry-run``
    在没有网络/没装 hub 的环境也能跑。
    """
    from fnmatch import fnmatch

    n_files = n_bytes = 0
    skipped = {"ckpt/**": 0}
    biggest: list[tuple[int, str]] = []
    root_s = str(root)
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root_s)
        rel_dir = "" if rel_dir == "." else rel_dir.replace(os.sep, "/")
        for name in filenames:
            rel = f"{rel_dir}/{name}" if rel_dir else name
            if any(fnmatch(rel, pat) for pat in ignore):
                key = "ckpt/**" if rel.startswith("ckpt/") else "其它"
                skipped[key] = skipped.get(key, 0) + 1
                continue
            p = os.path.join(dirpath, name)
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            n_files += 1
            n_bytes += sz
            biggest.append((sz, rel))
    biggest.sort(reverse=True)
    return {"n_files": n_files, "n_bytes": n_bytes, "skipped": skipped,
            "biggest": biggest[:10]}


def _check_token() -> str | None:
    """返回 token 来源说明；没 token 返回 None。绝不打印 token 本身。"""
    if os.environ.get("HF_TOKEN"):
        return "环境变量 HF_TOKEN"
    cached = Path("~/.cache/huggingface/token").expanduser()
    if cached.exists() and cached.stat().st_size > 0:
        return f"缓存文件 {cached}"
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="上传 data/ 到 HF Dataset")
    ap.add_argument("--data", default="data", help="本地目录（默认 data）")
    ap.add_argument("--repo", default=REPO_ID_DEFAULT)
    ap.add_argument("--with-ckpt", action="store_true", help="连 data/ckpt 一起传（66G）")
    ap.add_argument("--dry-run", action="store_true", help="只统计，不上传")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    root = Path(args.data).resolve()
    if not root.is_dir():
        print(f"[error] 目录不存在: {root}", file=sys.stderr)
        return 2

    ignore = [p for p in IGNORE if not (args.with_ckpt and p.startswith("ckpt"))]
    st = survey(root, ignore)
    print(f"仓库: {args.repo} (dataset)   本地: {root}")
    print(f"待传: {st['n_files']} 个文件 / {_human(st['n_bytes'])}")
    print(f"跳过: {st['skipped']}")
    print("最大的 10 个:")
    for sz, rel in st["biggest"]:
        print(f"  {_human(sz):>8}  {rel}")
    if args.dry_run:
        src = _check_token()
        print(f"凭据: {src or '未找到（真传会 401）'}")
        return 0

    src = _check_token()
    if src is None:
        print("[error] 没有 HuggingFace 凭据。两种办法：\n"
              "  1) 把 token 写进 ~/.cache/huggingface/token（推荐，权限 600）\n"
              "  2) export HF_TOKEN=...（注意会留在 shell history）\n"
              "token 需要 write 权限：https://huggingface.co/settings/tokens",
              file=sys.stderr)
        return 3

    # 延迟导入：dry-run 不需要它
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        me = api.whoami()
        print(f"登录身份: {me.get('name')} ({me.get('type')})，凭据来自 {src}")
    except Exception as e:  # noqa: BLE001
        print(f"[error] 凭据无效: {type(e).__name__}: {e}", file=sys.stderr)
        return 3

    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)

    print("[upload] 开始（分块、并行、断点续传；中断后重跑本命令即续传）")
    api.upload_large_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=str(root),
        ignore_patterns=ignore,
        num_workers=args.workers,
        print_report=True,
        print_report_every=30,
    )
    print("[upload] 完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
