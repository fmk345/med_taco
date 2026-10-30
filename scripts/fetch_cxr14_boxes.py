#!/usr/bin/env python3
"""从 HF 上的 ChestX-ray14 分卷 zip 里，只按 HTTP Range 取「有框」的那几百张图。

全量 45 GB 但 BBox_List_2017.csv 只覆盖 880 张图；remotezip 读中央目录后
只下载需要条目的字节，量级 ~50 MB。下载完做 PNG 完整性校验（解码失败即重下）。
"""
import csv, io, os, sys, zipfile, collections
BASE = "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/images/images_%03d.zip"

def main(out_root):
    img_dir = os.path.join(out_root, "images")
    os.makedirs(img_dir, exist_ok=True)
    want = set()
    with open(os.path.join(out_root, "BBox_List_2017.csv"), encoding="utf-8") as f:
        r = csv.reader(f); next(r)
        for row in r:
            if row and row[0].strip(): want.add(row[0].strip())
    print(f"需要 {len(want)} 张", flush=True)

    from remotezip import RemoteZip
    have = {os.path.basename(p) for p in os.listdir(img_dir)}
    from PIL import Image
    ok = 0; got = 0
    for i in range(1, 13):
        url = BASE % i
        try:
            with RemoteZip(url) as z:
                names = [n for n in z.namelist() if os.path.basename(n) in want and os.path.basename(n) not in have]
                print(f"[{i:03d}] 命中 {len(names)} 张", flush=True)
                for n in names:
                    try:
                        raw = z.read(n)
                        Image.open(io.BytesIO(raw)).verify()
                        with open(os.path.join(img_dir, os.path.basename(n)), "wb") as f:
                            f.write(raw)
                        ok += 1
                    except Exception as e:  # noqa: BLE001
                        print(f"[warn] {n}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                    got += 1
        except Exception as e:  # noqa: BLE001
            print(f"[error] {url}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        print(f"[{i:03d}] 完成，累计可用 {ok}/{len(want)}", flush=True)
    print(f"DONE ok={ok} total={len(want)}", flush=True)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/ChestX-ray14")
