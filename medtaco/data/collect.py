"""数据集采集：把各数据集的公有发布格式解析成统一的 `Sample`。

设计约束（见 INTERFACES.md 与 PROPOSAL §3.2）：
  - **本机没有数据集**，所以这里只写解析器，不做任何下载；解析器必须能在目录缺失时
    优雅跳过（打印 `[skip]`），不能抛异常——集群上跑全量时某几个数据集没到位是常态。
  - 每个解析器签名统一为 `parser(root_dir, cfg) -> Iterator[Sample]`，逐条 yield，
    单条坏记录只 `[warn]` 并跳过，不影响整批。
  - `Sample.question/answer` 在数据阶段可能为空：A 层样本的位置监督是主角，
    问题由 `synth/qgen.py` 生成、答案由教师填；这里只把「图 + 区域 + 原始元信息」落准。
  - 随机性：本模块不用随机数。

已实现的解析器（字段名按各数据集官方发布格式）：
  MS-CXR         — mscxr_bbox_annotations.json（image-sentence-box 三元组）
  VinDr-CXR      — annotations_train.csv / annotations_test.csv（列名见官方 README）
  SLAKE          — train.json / validation.json / test.json（qid/img_name/answer_type）
  ChestX-ray14   — BBox_List_2017.csv（"Bbox [x,y,w,h]" 单列，984 框）
  LIDC-IDRI      — tcia-lidc-xml/*.xml（unblindedReadNodule → roi → edgeMap）

其余数据集走 unsupported 分支，仅打印原因，不抛。
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from medtaco.utils.config import Cfg
from medtaco.utils.schema import Box, Sample

# 常见图像扩展名。顺序即解析器偏好：CXR 类先试 jpg/png，CT 类先试 dcm。
_IMG_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".dcm", ".dicom", ".tif", ".tiff")
_CT_EXTS: Tuple[str, ...] = (".dcm", ".dicom", ".png", ".jpg", ".jpeg")

# 判「不可 zoom」的硬下限（PROPOSAL §3.2 第 2 条：长边 <512 的图不进 zoom 任务）
_HARD_MIN_SIDE = 512


# ---------------------------------------------------------------- 小工具


def _get(cfg: Any, path: str, default: Any = None) -> Any:
    """同时支持 `Cfg` 与普通 dict，点号路径取值。"""
    if isinstance(cfg, Cfg):
        return cfg.get(path, default)
    cur: Any = cfg
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _log(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)


def _slug(name: str) -> str:
    """数据集名 → id 前缀：MS-CXR → ms_cxr。"""
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


def _sid(*parts: Any) -> str:
    """稳定 id：内容哈希，保证同一记录重跑得到同一 id（断点续传依赖这点）。"""
    raw = "|".join(str(p) for p in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _mk_box(x0: float, y0: float, x1: float, y1: float, label: str = "") -> Optional[Box]:
    """归一化坐标并丢弃退化框（宽或高 <= 0）；坐标顺序反了自动纠正。"""
    try:
        x0, x1 = sorted((float(x0), float(x1)))
        y0, y1 = sorted((float(y0), float(y1)))
    except (TypeError, ValueError):
        return None
    if not all(map(_finite, (x0, y0, x1, y1))):
        return None
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    return Box(x0, y0, x1, y1, str(label))


def _finite(v: float) -> bool:
    return v == v and abs(v) != float("inf")


def _bbox_from_any(obj: Any) -> Optional[Tuple[float, float, float, float]]:
    """从各种常见写法里抽出 (x0,y0,x1,y1)。

    支持：dict{x,y,width,height} / dict{x0,y0,x1,y1} / COCO 风格四元组 [x,y,w,h]
    / xyxy 四元组（由调用方声明）。COCO 默认是 xywh，这里按 xywh 解释。
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        if "bbox" in obj:
            return _bbox_from_any(obj["bbox"])
        if all(k in obj for k in ("x0", "y0", "x1", "y1")):
            return (obj["x0"], obj["y0"], obj["x1"], obj["y1"])
        if all(k in obj for k in ("x", "y")):
            w = obj.get("width", obj.get("w"))
            h = obj.get("height", obj.get("h"))
            if w is None or h is None:
                return None
            return (obj["x"], obj["y"], float(obj["x"]) + float(w), float(obj["y"]) + float(h))
        return None
    if isinstance(obj, (list, tuple)) and len(obj) >= 4:
        x, y, w, h = obj[:4]
        return (float(x), float(y), float(x) + float(w), float(y) + float(h))
    return None


def _resolve_image(root: str, name: str, subdirs: Sequence[str], exts: Sequence[str] = _IMG_EXTS) -> str:
    """在若干候选子目录里按扩展名找图；找不到返回最可能的猜测路径（下游可据此报缺图）。"""
    stem = name if not os.path.splitext(name)[1] else name
    # CSV 里常直接写 ``00013118_008.png``：先按原名找，否则会拼成 ``.png.png`` 恒找不到。
    exts = ((("",) + tuple(exts)) if os.path.splitext(name)[1] else tuple(exts))
    for sd in subdirs:
        base = os.path.join(root, sd) if sd else root
        for ext in exts:
            p = os.path.join(base, stem + ext)
            if os.path.exists(p):
                return p
    base = os.path.join(root, subdirs[0]) if subdirs and subdirs[0] else root
    return os.path.join(base, stem + exts[0])


def _dataset_meta(cfg: Any, name: str) -> Dict[str, Any]:
    """取 cfg.data.datasets.<name> 里的 tier/license/commercial，缺省给安全值。"""
    d = _get(cfg, f"data.datasets.{name}", None) or {}
    return {
        "tier": str(d.get("tier", "B")),
        "license": str(d.get("license", "unknown")),
        "commercial": bool(d.get("commercial", False)),
    }


def _make_sample(
    cfg: Any,
    name: str,
    image: str,
    question: str = "",
    answer: str = "",
    boxes: Optional[List[Box]] = None,
    answer_type: str = "short",
    masks: Optional[List[str]] = None,
    meta: Optional[Dict[str, Any]] = None,
    key: Any = None,
) -> Sample:
    """统一构造 Sample：id 由 (数据集, key) 决定，tier/license 从配置兜底。"""
    dm = _dataset_meta(cfg, name)
    boxes = boxes or []
    sid = f"{_slug(name)}:{_sid(name, key if key is not None else image)}"
    # tier 描述的是「这条样本」而非数据集：A 层数据集里没框的条目只能做答案监督，降为 B。
    tier = dm["tier"] if (boxes or dm["tier"] != "A") else "B"
    m: Dict[str, Any] = {"source_dataset": name, "has_region": bool(boxes)}
    m.update(meta or {})
    return Sample(
        id=sid,
        image=image,
        question=question,
        answer=answer,
        dataset=name,
        tier=tier,
        answer_type=answer_type,
        boxes=boxes,
        masks=list(masks or []),
        license=dm["license"],
        commercial_ok=dm["commercial"],
        meta=m,
    )


def dataset_root(name: str, cfg: Any) -> str:
    """数据集根目录：优先 cfg.data.roots.<name>，否则 <paths.raw>/<name>。"""
    roots = _get(cfg, "data.roots", None) or {}
    if isinstance(roots, dict) and name in roots:
        return str(roots[name])
    raw = str(_get(cfg, "paths.raw", "data/raw"))
    return os.path.join(raw, name)


# ---------------------------------------------------------------- MS-CXR


def parse_ms_cxr(root: str, cfg: Any) -> Iterator[Sample]:
    """MS-CXR：1,162 条 image-sentence-box 三元组，8 类心肺表现。

    官方发布是 `mscxr_bbox_annotations.json`（dict: ann_id -> {image_id, x, y, width,
    height, label_text, sentence}）。也兼容包一层 `{"annotations": [...]}` 的 COCO 变体，
    便于用自造小样本做单测。
    """
    cands = [
        "mscxr_bbox_annotations.json",
        "ms_cxr_bbox_annotations.json",
        "annotations.json",
        "bbox_annotations.json",
    ]
    paths = [os.path.join(root, c) for c in cands if os.path.exists(os.path.join(root, c))]
    if not paths:
        paths = sorted(glob.glob(os.path.join(root, "*.json")))
    if not paths:
        _log("skip", f"MS-CXR 缺少标注 json: {root}")
        return

    subdirs = ["images", "images_data", "files", ""]
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            _log("warn", f"MS-CXR 标注读取失败 {path}: {e}")
            continue

        items = _flatten_anns(data)
        for ann_id, a in items:
            if not isinstance(a, dict):
                continue
            image_id = a.get("image_id") or a.get("dicom_id") or a.get("ori_img_id") or a.get("file_name")
            if not image_id:
                continue
            image_id = os.path.splitext(str(image_id))[0]
            xy = _bbox_from_any(a)
            if xy is None:
                continue
            label = str(a.get("label_text") or a.get("bbox_name") or a.get("label") or a.get("finding") or "")
            box = _mk_box(xy[0], xy[1], xy[2], xy[3], label)
            if box is None:
                continue
            sentence = str(a.get("sentence") or a.get("phrase") or "")
            img = _resolve_image(root, image_id, subdirs)
            yield _make_sample(
                cfg,
                "MS-CXR",
                img,
                question=sentence,
                boxes=[box],
                answer_type="long",  # 句子级描述，线上不判分，仅作 SFT 目标
                meta={"image_id": image_id, "ann_id": str(ann_id), "cui": a.get("label_cui", "")},
                key=(image_id, str(ann_id), label, xy),
            )


def _flatten_anns(data: Any) -> List[Tuple[str, dict]]:
    """把 dict-of-ann / list-of-ann / COCO 包装统一成 [(key, ann), ...]。"""
    if isinstance(data, dict):
        for k in ("annotations", "data", "items"):
            if k in data and isinstance(data[k], (list, dict)):
                return _flatten_anns(data[k])
        if "image_id" in data and any(x in data for x in ("bbox", "x", "x0")):
            return [("0", data)]
        return [(str(k), v) for k, v in data.items() if isinstance(v, dict)]
    if isinstance(data, list):
        return [(str(i), v) for i, v in enumerate(data) if isinstance(v, dict)]
    return []


# ---------------------------------------------------------------- VinDr-CXR


#: VinDr-CXR 必需的**逻辑**列名；实际表头由 _VINDR_COL_ALIASES 映射，不写死拼写。
_VINDR_COLS: Tuple[str, ...] = ("image_id", "finding", "x_min", "y_min", "x_max", "y_max")

#: 逻辑列名 -> 可接受的表头别名（按优先级）。官方 Kaggle/PhysioNet 版 annotation csv 的
#: 表头是 `image_id, rad_id, class_name, class_id, x_min, y_min, x_max, y_max`；
#: 早期自造 fixture 用的是 `finding_label`。两者都接受，避免真实 csv 整表被跳过。
_VINDR_COL_ALIASES: Dict[str, Tuple[str, ...]] = {
    "image_id": ("image_id",),
    "finding": ("class_name", "finding_label"),
    "x_min": ("x_min",),
    "y_min": ("y_min",),
    "x_max": ("x_max",),
    "y_max": ("y_max",),
}


def _vindr_resolve_cols(fieldnames: Optional[Sequence[str]]) -> Optional[Dict[str, str]]:
    """把 csv 实际表头映射到逻辑列名；缺任一必需列时返回 None。

    容错：表头去空白、大小写不敏感、忽略 UTF-8 BOM（Excel 导出的 csv 常带）。
    返回的 dict 是 逻辑名 -> 原始表头，供 DictReader 取行时使用。
    """
    idx: Dict[str, str] = {}
    for c in fieldnames or ():
        if c is None:
            continue
        name = str(c).strip().lstrip("﻿").strip()
        if name:
            idx.setdefault(name.lower(), name)
    out: Dict[str, str] = {}
    for logical in _VINDR_COLS:
        for alias in _VINDR_COL_ALIASES.get(logical, (logical,)):
            hit = idx.get(alias.lower())
            if hit is not None:
                out[logical] = hit
                break
        else:
            return None
    return out


def parse_vindr_cxr(root: str, cfg: Any) -> Iterator[Sample]:
    """VinDr-CXR：18,000 CXR / 22 类病灶框，标注在 annotations_{train,test}.csv。

    官方表头 `image_id, rad_id, class_name, class_id, x_min, y_min, x_max, y_max`
    （rad_id 是多医生独立标注，同一框可重复出现多行；这里按 (image, finding) 聚合，
    保留 rad_id 计数）。`class_name` 与旧 fixture 用的 `finding_label` 通过别名表
    等价接受，见 _VINDR_COL_ALIASES；rad_id 可选，缺列时 n_rads 记 0。
    """
    csvs = sorted(glob.glob(os.path.join(root, "annotations_*.csv")))
    if not csvs:
        csvs = sorted(glob.glob(os.path.join(root, "**", "annotations_*.csv"), recursive=True))[:4]
    if not csvs:
        _log("skip", f"VinDr-CXR 缺少 annotations_*.csv: {root}")
        return

    for csv_path in csvs:
        split = _split_from_name(os.path.basename(csv_path), default="")
        subdirs = [split, "images", ""] if split else ["images", "train", ""]
        groups: Dict[Tuple[str, str], List[Box]] = {}
        rads: Dict[Tuple[str, str], set] = {}
        try:
            # utf-8-sig：去掉可能的 BOM，否则首列会被读成 '﻿image_id' 而整表列名不符。
            with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                cols = _vindr_resolve_cols(reader.fieldnames)
                if cols is None:
                    _log("warn", f"VinDr-CXR 列名不符，跳过 {csv_path}: "
                                 f"{[str(c) for c in (reader.fieldnames or [])]}")
                    continue
                # 同一个 csv 里所有行共用一份列映射，避免逐行重复解析表头。
                c_img, c_find = cols["image_id"], cols["finding"]
                c_xmin, c_ymin = cols["x_min"], cols["y_min"]
                c_xmax, c_ymax = cols["x_max"], cols["y_max"]
                # rad_id 是可选列：新版官方 csv 有，自造 fixture / 精简版可能没有。
                c_rad = next((c for c in (reader.fieldnames or [])
                              if str(c).strip().lstrip("﻿").lower() == "rad_id"), None)
                for row in reader:
                    key = (str(row.get(c_img) or "").strip(), str(row.get(c_find) or "").strip())
                    if not key[0]:
                        continue   # 无 image_id 的行无法定位图像
                    try:
                        box = _mk_box(row.get(c_xmin), row.get(c_ymin),
                                      row.get(c_xmax), row.get(c_ymax), key[1])
                    except (TypeError, ValueError):
                        continue
                    if box is None:
                        continue
                    groups.setdefault(key, []).append(box)
                    rid = str(row.get(c_rad, "") or "").strip() if c_rad else ""
                    if rid:
                        rads.setdefault(key, set()).add(rid)
        except OSError as e:
            _log("warn", f"VinDr-CXR 读取失败 {csv_path}: {e}")
            continue

        for (image_id, finding), boxes in groups.items():
            img = _resolve_image(root, image_id, subdirs, _CT_EXTS)
            yield _make_sample(
                cfg,
                "VinDr-CXR",
                img,
                boxes=boxes,
                answer_type="short",
                meta={
                    "image_id": image_id,
                    "finding": finding,
                    "n_rads": len(rads.get((image_id, finding), ())),
                    "split": split,
                },
                key=(image_id, finding, boxes[0].as_tuple()),
            )


def _split_from_name(name: str, default: str = "") -> str:
    low = name.lower()
    for s in ("train", "test", "val", "valid", "validation"):
        if s in low:
            return s
    return default


# ---------------------------------------------------------------- SLAKE


_SLAKE_ANSWER_TYPE = {"CLOSED": "closed", "OPEN": "short"}


def parse_slake(root: str, cfg: Any) -> Iterator[Sample]:
    """SLAKE：642 图 / 14,028 中英 QA，经典基准里唯一同时给框与分割掩膜的数据集。

    官方 json 是 list，每条：
      {qid, img_name, question, answer, answer_type: OPEN|CLOSED, modality,
       content_type, question_type, q_lang: en|zh, location, base_dir}
    框不在 json 里时，从 `GT_bbox/<img_name>.json` 读（可选，缺失则 boxes 为空）。
    """
    jsons = [os.path.join(root, x) for x in ("train.json", "validation.json", "test.json")]
    jsons = [p for p in jsons if os.path.exists(p)]
    if not jsons:
        jsons = sorted(glob.glob(os.path.join(root, "*.json")))
    if not jsons:
        _log("skip", f"SLAKE 缺少 train/validation/test.json: {root}")
        return

    for jp in jsons:
        split = _split_from_name(os.path.basename(jp), default="")
        try:
            with open(jp, "r", encoding="utf-8") as f:
                items = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            _log("warn", f"SLAKE 读取失败 {jp}: {e}")
            continue
        if isinstance(items, dict):
            items = items.get("data") or items.get("questions") or []
        if not isinstance(items, list):
            continue

        for it in items:
            if not isinstance(it, dict):
                continue
            img_name = str(it.get("img_name") or it.get("image") or "")
            if not img_name:
                continue
            lang = str(it.get("q_lang") or "en").lower()
            q = str(it.get("question") or "")
            a = str(it.get("answer") or "")
            at = _SLAKE_ANSWER_TYPE.get(str(it.get("answer_type", "")).upper(), "short")
            boxes = _slake_boxes(root, img_name, it)
            masks = _slake_masks(root, img_name)
            img = _resolve_image(root, img_name, ["", "imgs", "images", "SLAKE/imgs"])
            meta = {
                "image_id": img_name,
                "qid": it.get("qid"),
                "lang": lang,
                "modality": it.get("modality", ""),
                "content_type": it.get("content_type", ""),
                "question_type": it.get("question_type", ""),
                "split": split,
            }
            if lang == "zh":
                meta["question_zh"] = q
            yield _make_sample(
                cfg,
                "SLAKE",
                img,
                question=q,
                answer=a,
                boxes=boxes,
                answer_type=at,
                masks=masks,
                meta=meta,
                key=(img_name, it.get("qid"), q, a),
            )


def _slake_boxes(root: str, img_name: str, item: dict) -> List[Box]:
    """SLAKE 的框优先取 item 内联字段，其次取 GT_bbox/<img_name>.json。"""
    # 裸四元组按 COCO 口径 xywh 解释；要写 xyxy 请用 dict 的 x0/y0/x1/y1 键。
    raw = item.get("bbox") or item.get("boxes") or item.get("location")
    lab = str(item.get("content_type") or item.get("question_type") or "")
    boxes: List[Box] = []
    if isinstance(raw, (list, dict)):
        seq = raw if isinstance(raw, list) else [raw]
        # list[list[4]] 与 list[dict] 都支持
        if seq and all(isinstance(v, (int, float)) for v in seq[:4]) and len(seq) <= 4:
            seq = [seq]
        for r in seq:
            xy = _bbox_from_any(r)
            if xy:
                b = _mk_box(*xy, lab)
                if b:
                    boxes.append(b)
    if boxes:
        return boxes
    stem = os.path.splitext(img_name)[0]
    for cand in (os.path.join(root, "GT_bbox", stem + ".json"),
                 os.path.join(root, "GT_bbox", img_name + ".json")):
        if os.path.exists(cand):
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            for _, ann in _flatten_anns(data):
                xy = _bbox_from_any(ann)
                if xy:
                    b = _mk_box(*xy, lab)
                    if b:
                        boxes.append(b)
            if boxes:
                break
    return boxes


def _slake_masks(root: str, img_name: str) -> List[str]:
    stem = os.path.splitext(img_name)[0]
    out: List[str] = []
    for sd, ext in (("GT_mask", ".png"), ("mask", ".png"), ("GT_mask", ".jpg")):
        p = os.path.join(root, sd, stem + ext)
        if os.path.exists(p):
            out.append(p)
    return out


# ---------------------------------------------------------------- ChestX-ray14


def parse_chestxray14(root: str, cfg: Any) -> Iterator[Sample]:
    """NIH ChestX-ray14：112,120 张图里只有 984 个框 / 880 图（8 类），列在
    `BBox_List_2017.csv`，坐标是单列字符串 `"Bbox [x,y,w,h]"`（COCO 口径 xywh）。
    """
    cands = ["BBox_List_2017.csv", "bbox_list_2017.csv"]
    csv_path = next((os.path.join(root, c) for c in cands if os.path.exists(os.path.join(root, c))), "")
    if not csv_path:
        hits = sorted(glob.glob(os.path.join(root, "**", "BBox_List_2017.csv"), recursive=True))[:1]
        csv_path = hits[0] if hits else ""
    if not csv_path:
        _log("skip", f"ChestX-ray14 缺少 BBox_List_2017.csv: {root}")
        return

    subdirs = ["images", "images_001/images", ""]
    groups: Dict[Tuple[str, str], List[Box]] = {}
    n_bad = 0
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                _log("warn", f"ChestX-ray14 表头为空: {csv_path}")
                return
            # 官方表头是未加引号的 `Bbox [x,y,w,h]`，csv 会把它切成 5 列；
            # 数据行则是一个带引号的 3 列。所以固定前两列，其余全部拼回 bbox 字符串。
            for row in reader:
                if len(row) < 3:
                    n_bad += 1
                    continue
                img = row[0].strip()
                finding = row[1].strip()
                raw = ",".join(row[2:]).strip()
                if not img or not raw:
                    n_bad += 1
                    continue
                xy = _parse_bbox_string(raw)
                if xy is None:
                    n_bad += 1
                    continue
                box = _mk_box(*xy, finding)
                if box is None:
                    n_bad += 1
                    continue
                groups.setdefault((img, finding), []).append(box)
    except OSError as e:
        _log("warn", f"ChestX-ray14 读取失败 {csv_path}: {e}")
        return
    if n_bad:
        _log("warn", f"ChestX-ray14 有 {n_bad} 行无法解析（已跳过）")

    n_missing = 0
    for (image_id, finding), boxes in groups.items():
        img = _resolve_image(root, image_id, subdirs)
        # 只下了一部分图（如按 Range 取有框的那几张）时，_resolve_image 仍会返回
        # 一个拼出来的路径。文件不存在就跳过，否则下游会拿到一堆死路径。
        if not os.path.exists(img):
            n_missing += 1
            continue
        yield _make_sample(
            cfg,
            "ChestX-ray14",
            img,
            boxes=boxes,
            answer_type="closed",  # 8 类是 presence 型标签，天然 yes/no
            meta={"image_id": image_id, "finding": finding},
            key=(image_id, finding, boxes[0].as_tuple()),
        )
    if n_missing:
        _log("warn", f"ChestX-ray14 有 {n_missing} 条样本的图不在本地（已跳过；"
                     f"先跑 scripts/fetch_cxr14_boxes.py）")


def _parse_bbox_string(raw: str) -> Optional[Tuple[float, float, float, float]]:
    """解析 `"Bbox [x,y,w,h]"` / `"[x,y,w,h]"` / `"x,y,w,h"` 三种写法，返回 xyxy。"""
    s = raw.strip()
    if "[" in s and "]" in s:
        s = s[s.index("[") + 1: s.rindex("]")]
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    if len(parts) < 4:
        return None
    try:
        x, y, w, h = (float(p) for p in parts[:4])
    except ValueError:
        return None
    return (x, y, x + w, y + h)


# ---------------------------------------------------------------- LIDC-IDRI


def parse_lidc_idri(root: str, cfg: Any) -> Iterator[Sample]:
    """LIDC-IDRI：1,018 例 CT / 2,669 结节，4 位医生轮廓（XML）+ mm 半径。

    解析 `tcia-lidc-xml/*.xml`：LidcReadMessage → readingSession → unblindedReadNodule
    → roi → edgeMap(xCoord, yCoord)。每个 roi 是**一个层面的轮廓**，取轮廓的外接矩形
    作为该层的框（切片级区域监督），并按 (nodedID, imageSOP_UID) 一组 yield。
    像素间距只在 DICOM 里，本模块不解析 DICOM，故 `spacing_available=False`。
    """
    xmls = sorted(glob.glob(os.path.join(root, "**", "*.xml"), recursive=True))
    cap = int(_get(cfg, "data.lidc.max_xml", 5000) or 5000)
    if not xmls:
        _log("skip", f"LIDC-IDRI 缺少 XML: {root}")
        return
    if len(xmls) > cap:
        _log("warn", f"LIDC-IDRI XML 数量 {len(xmls)} 超过上限 {cap}，只取前 {cap} 个")
        xmls = xmls[:cap]

    dcm_index = _index_dcm(root, cap=int(_get(cfg, "data.lidc.max_dcm_index", 200000) or 200000))

    for xml_path in xmls:
        try:
            tree = ET.parse(xml_path)
        except (ET.ParseError, OSError) as e:
            _log("warn", f"LIDC-XML 解析失败 {xml_path}: {e}")
            continue
        r = tree.getroot()
        patient = _first_text(r, "SeriesInstanceUID") or os.path.basename(xml_path)
        for nodule in _iter_local(r, "unblindedReadNodule"):
            nid = _first_text(nodule, "noduleID") or ""
            chars = {c.tag.split("}")[-1]: (c.text or "").strip()
                     for ch in _iter_local(nodule, "characteristics")
                     for c in list(ch)}
            for roi in _iter_local(nodule, "roi"):
                sop = _first_text(roi, "imageSOP_UID") or ""
                if not sop:
                    continue   # 无 SOP UID 就无法定位到具体切片
                zpos = _first_text(roi, "imageZposition") or ""
                xs: List[float] = []
                ys: List[float] = []
                for em in _iter_local(roi, "edgeMap"):
                    x = _first_text(em, "xCoord")
                    y = _first_text(em, "yCoord")
                    if x is None or y is None:
                        continue
                    try:
                        xs.append(float(x))
                        ys.append(float(y))
                    except ValueError:
                        continue
                if len(xs) < 3:
                    continue
                box = _mk_box(min(xs), min(ys), max(xs), max(ys), nid or "nodule")
                if box is None:
                    continue
                img = dcm_index.get(sop) or _resolve_image(root, sop, ["", "images"], _CT_EXTS)
                yield _make_sample(
                    cfg,
                    "LIDC-IDRI",
                    img,
                    boxes=[box],
                    answer_type="measure",  # 有 spacing 时才是真 caliper 题，见 meta
                    meta={
                        "image_id": sop,
                        "nodule_id": nid,
                        "z_position": zpos,
                        "series": patient,
                        "characteristics": chars,
                        "spacing_available": False,
                        "source_xml": xml_path,
                    },
                    key=(sop, nid, zpos, box.as_tuple()),
                )


def _iter_local(elem: ET.Element, tag: str) -> Iterator[ET.Element]:
    """忽略命名空间地找直接/间接子节点（LIDC XML 带 xmlns="http://www.nih.gov"）。"""
    for e in elem.iter():
        if e.tag.split("}")[-1] == tag:
            yield e


def _first_text(elem: ET.Element, tag: str) -> Optional[str]:
    for e in elem.iter():
        if e.tag.split("}")[-1] == tag:
            return (e.text or "").strip()
    return None


def _index_dcm(root: str, cap: int) -> Dict[str, str]:
    """建立 SOP UID -> dcm 路径索引（按文件名 stem）。超过 cap 就停止，避免扫爆挂载盘。"""
    idx: Dict[str, str] = {}
    n = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith((".dcm", ".dicom")):
                continue
            idx.setdefault(os.path.splitext(fn)[0], os.path.join(dirpath, fn))
            n += 1
            if n >= cap:
                return idx
    return idx


# ---------------------------------------------------------------- Mock（自造数据）


#: mock 数据集的默认规模；半数是"有框且必须裁"的题，半数是"不用工具"的题，
#: 后者是为了让 check_gate 的 over_call_rate 有分母（见 scripts/check_gate.py）。
_MOCK_N = 12
_MOCK_SIDE = 1024


def _mock_rng(seed: int):
    """确定性伪随机（线性同余）。**不用 random 模块**：本文件要求重跑逐字节一致，
    而 random 的算法/seed 处理跨版本会变。"""
    state = (seed * 2654435761 + 1013904223) & 0xFFFFFFFF

    def nxt(lo: int, hi: int) -> int:
        nonlocal state
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        return lo + state % max(1, hi - lo)

    return nxt


def _mock_draw(path: str, seed: int, box: Optional[Tuple[int, int, int, int]],
               has_diagonal: bool, bright: bool) -> None:
    """画一张"必须放大才看得清"的图：整体是低频灰底 + 小块纹理，
    病灶是 100px 左右的矩形（占全图 1%），矩形内可能有一条 2px 对角线。"""
    from PIL import Image, ImageDraw  # 延迟 import：只有 mock 数据集需要

    base = 150 if bright else 90
    img = Image.new("RGB", (_MOCK_SIDE, _MOCK_SIDE), (base, base, base))
    draw = ImageDraw.Draw(img)

    rng = _mock_rng(seed)
    # 低频斑块：制造"和病灶同亮度"的干扰物，让 crop 成为必需
    for _ in range(18):
        cx, cy = rng(0, _MOCK_SIDE), rng(0, _MOCK_SIDE)
        r = rng(40, 160)
        v = base + rng(-30, 30)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(v, v, v))

    if box is not None:
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255), width=3)
        fill = base + 18
        draw.rectangle([x0 + 3, y0 + 3, x1 - 3, y1 - 3], fill=(fill, fill, fill))
        if has_diagonal:
            draw.line([x0 + 6, y0 + 6, x1 - 6, y1 - 6], fill=(255, 255, 255), width=2)

    img.save(path)


def parse_mock(root: str, cfg: Any) -> Iterator[Sample]:
    """离线自造数据集：产出真实 PNG + 标注，用来把 collect→qgen→synth→export 整条链跑通。

    **不是训练数据**：每条 `meta['mock']=True`，export 会写到 `*.mock.jsonl`，
    训练脚本只读 `sft.jsonl`，因此不会被误用。

    生成规则（全部确定性，重跑逐字节一致）：
      - 12 条：9 条带框（tier A，需要 crop/zoom 才看得清）、3 条无框（needs_tool=False）
      - 带框题的答案是"矩形内有没有对角线"，奇偶交替，无法靠猜（50% 基线）
      - 无框题的答案是整体亮/暗，直接看全图即可
    """
    n = int(_get(cfg, "data.mock_n", _MOCK_N) or _MOCK_N) if cfg is not None else _MOCK_N
    img_dir = os.path.join(root, "images")
    labels_path = os.path.join(root, "labels.jsonl")

    if not os.path.exists(labels_path):
        try:
            os.makedirs(img_dir, exist_ok=True)
        except OSError as e:
            _log("warn", f"Mock: 无法创建目录 {img_dir}: {e}")
            return
        rows: List[Dict[str, Any]] = []
        for i in range(n):
            rng = _mock_rng(1000 + i)
            name = f"mock_{i:03d}.png"
            path = os.path.join(img_dir, name)
            with_box = (i % 4) != 3          # 每 4 条里有 1 条不需要工具
            if with_box:
                w, h = rng(90, 140), rng(90, 140)
                x0 = rng(80, _MOCK_SIDE - w - 80)
                y0 = rng(80, _MOCK_SIDE - h - 80)
                box = (x0, y0, x0 + w, y0 + h)
                has_diag = (i % 2) == 0
                try:
                    _mock_draw(path, i, box, has_diag, bright=(i % 3 == 0))
                except Exception as e:  # noqa: BLE001 - 缺 PIL/写盘失败都不该整批挂掉
                    _log("warn", f"Mock: 画图失败 {path}: {type(e).__name__}: {e}")
                    return
                rows.append({
                    "image": name,
                    "question": "图中白色矩形框内的区域，是否有一条贯穿整个矩形的对角线？",
                    "answer": "是" if has_diag else "否",
                    "answer_type": "closed",
                    "boxes": [[float(x0), float(y0), float(x0 + w), float(y0 + h), "mock_lesion"]],
                    "needs_tool": True,
                })
            else:
                bright = (i % 8) != 7
                try:
                    _mock_draw(path, i, None, False, bright=bright)
                except Exception as e:  # noqa: BLE001
                    _log("warn", f"Mock: 画图失败 {path}: {type(e).__name__}: {e}")
                    return
                rows.append({
                    "image": name,
                    "question": "这幅图像整体偏亮还是偏暗？",
                    "answer": "亮" if bright else "暗",
                    "answer_type": "closed",
                    "boxes": [],
                    "needs_tool": False,
                })
        with open(labels_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        _log("mock", f"生成 {len(rows)} 条自造样本 -> {root}")

    n_ok = 0
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                _log("warn", "Mock: labels.jsonl 有坏行，已跳过")
                continue
            img = os.path.join(img_dir, r["image"])
            if not os.path.exists(img):
                _log("warn", f"Mock: 缺图 {img}，删除 labels.jsonl 可重生成")
                continue
            boxes = [_mk_box(*b[:4], b[4] if len(b) > 4 else "") for b in r.get("boxes", [])]
            boxes = [b for b in boxes if b is not None]
            s = _make_sample(
                cfg, "Mock", img,
                question=r.get("question", ""),
                answer=r.get("answer", ""),
                boxes=boxes,
                answer_type=r.get("answer_type", "closed"),
                meta={"mock": True,
                      "needs_tool": bool(r.get("needs_tool", bool(boxes))),
                      "orig_size": [_MOCK_SIDE, _MOCK_SIDE]},
                key=r["image"],
            )
            n_ok += 1
            yield s
    if n_ok == 0:
        _log("warn", f"Mock: {labels_path} 里没有可用样本")


# ---------------------------------------------------------------- 分发


PARSERS: Dict[str, Callable[[str, Any], Iterator[Sample]]] = {
    "MS-CXR": parse_ms_cxr,
    "VinDr-CXR": parse_vindr_cxr,
    "SLAKE": parse_slake,
    "ChestX-ray14": parse_chestxray14,
    "LIDC-IDRI": parse_lidc_idri,
    "Mock": parse_mock,
}

#: 由解析器自己创建目录的数据集（不要求事先存在，否则 collect_dataset 会提前 skip）
_SELF_CREATING = {"Mock"}

# 已知但本轮不实现的数据集：给出原因，便于按 PROPOSAL 的清单补齐
_UNSUPPORTED: Dict[str, str] = {
    "ChestImaGenome": "scene graph json 与 MIMIC-CXR 影像耦合，需先落地 MIMIC",
    "VinDr-CXR-VQA": "QA json 结构未定稿，待标注包到位",
    "PadChest": "标注为文本报告，无像素框；只进 B 层",
    "PadChest-GR": "grounded report 格式待确认",
    "DeepLesion": "RECIST 标注在 CSV，需 DICOM 定位（无 pydicom）",
    "CT-RATE": "需 DICOM -> HU 体数据管线",
    "ReXGroundingCT": "3D 框格式，需体数据支持",
    "CAMELYON16": "WSI 金字塔，需 openslide",
    "CAMELYON17": "WSI 金字塔，需 openslide",
    "ISIC": "待实现（掩膜目录约定见 ISIC 官方 README）",
    "HAM10000": "待实现（CSV + 图像目录）",
    "Kvasir-SEG": "待实现（图像/掩膜同名目录）",
    "CVC-ClinicDB": "待实现（tif + 掩膜）",
    "BUSI": "待实现（三类子目录 + mask）",
    "BrEaST": "待实现（DICOM + 掩膜）",
    "IDRiD": "待实现（像素级标注 tif）",
    "Object-CXR": "待实现（train/test + bbox csv）",
    "TBX11K": "待实现（bbox xml）",
    "VinDr-Mammo": "待实现（DICOM + 框 csv）",
    "VQA-RAD": "B 层，仅答案监督，接入点见 build_samples",
    "PathVQA": "B 层，仅答案监督",
    "PMC-VQA": "B 层，需先对齐图源",
    "OmniMedVQA": "73 个源各自 license，需按源分别接入",
    "GMAI-MMBench": "CC BY-NC-SA 4.0 传染性，仅评测",
    "MedQA": "C 层纯文本，不进图像管线",
    "MedMCQA": "C 层纯文本，不进图像管线",
}


def collect_dataset(name: str, cfg: Any) -> Iterator[Sample]:
    """按数据集名分发。目录不存在 / 未实现时打印 `[skip]` 并返回空迭代器，绝不抛异常。"""
    parser = PARSERS.get(name)
    if parser is None:
        reason = _UNSUPPORTED.get(name, "未收录的数据集")
        _log("skip", f"{name}: 未实现解析器（{reason}）")
        return

    root = dataset_root(name, cfg)
    if not os.path.isdir(root):
        # 自造数据集（Mock）由解析器自己建目录；其余数据集缺目录就是没下载，跳过。
        if name not in _SELF_CREATING:
            _log("skip", f"{name}: 数据目录不存在 {root}")
            return

    n = 0
    try:
        for s in parser(root, cfg):
            n += 1
            yield s
    except Exception as e:  # noqa: BLE001 - 单数据集失败不应中断整体采集
        _log("warn", f"{name}: 解析中止（已产出 {n} 条）: {type(e).__name__}: {e}")


def iter_all(cfg: Any, datasets: Optional[Sequence[str]] = None) -> Iterator[Sample]:
    """按 cfg.data.datasets 的顺序遍历所有已配置数据集（辅助函数，非契约必需）。"""
    names = list(datasets) if datasets else list((_get(cfg, "data.datasets", {}) or {}).keys())
    for name in names:
        yield from collect_dataset(name, cfg)


def scan_manifest(cfg: Any, datasets: Optional[Sequence[str]] = None,
                  limit: Optional[int] = None) -> List[dict]:
    """产出 `{dataset, image, box_count, tier, license}` 清单（按图去重）。

    limit 默认取 cfg.data.manifest_limit（缺省 20000），只用于盘点、不用于训练，
    所以截断只 warn 不报错。
    """
    lim = int(limit if limit is not None else (_get(cfg, "data.manifest_limit", 20000) or 20000))
    seen: Dict[Tuple[str, str], dict] = {}
    for s in iter_all(cfg, datasets):
        key = (s.dataset, s.image)
        row = seen.get(key)
        if row is None:
            if len(seen) >= lim:
                _log("warn", f"scan_manifest 达到上限 {lim}，提前结束")
                break
            seen[key] = {
                "dataset": s.dataset,
                "image": s.image,
                "box_count": len(s.boxes),
                "tier": s.tier,
                "license": s.license,
            }
        else:
            # 同一图的多条样本（多 finding / 多问）累加框数；有框就把 tier 抬到 A
            row["box_count"] += len(s.boxes)
            if s.boxes:
                row["tier"] = "A"
    return list(seen.values())


# ---------------------------------------------------------------- 分辨率审计


def audit_resolution(samples: Iterable[Sample], cfg: Any = None) -> dict:
    """可 zoom / 不可 zoom 统计（PROPOSAL §3.2 第 2 条）。

    判据：长边 >= cfg.data.resolution.zoom_min_side（默认 1024）→ 可 zoom；
    长边 < 512 → 连 crop 任务都不可靠（放大即插值），单列 `too_small`。
    尺寸来源：sample.meta['orig_size'] 优先，其次用 PIL 读一次（带缓存）。
    """
    zoom_min = int(_get(cfg, "data.resolution.zoom_min_side", 1024) or 1024) if cfg is not None else 1024
    size_cache: Dict[str, Optional[Tuple[int, int]]] = {}

    tot = {"n_samples": 0, "n_images": 0, "zoomable_images": 0,
           "non_zoomable_images": 0, "too_small_images": 0, "unreadable_images": 0}
    per_ds: Dict[str, Dict[str, int]] = {}
    images: set = set()
    zoom_imgs: set = set()

    for s in samples:
        tot["n_samples"] += 1
        ds = s.dataset or "unknown"
        row = per_ds.setdefault(ds, {"n_images": 0, "zoomable": 0, "non_zoomable": 0,
                                     "too_small": 0, "unreadable": 0})
        is_new = s.image not in images
        if is_new:
            images.add(s.image)
            row["n_images"] += 1
            tot["n_images"] += 1
        sz = _sample_size(s, size_cache)
        if sz is None:
            if is_new:
                tot["unreadable_images"] += 1
                row["unreadable"] += 1
            continue
        long_side = max(sz)
        if long_side >= zoom_min:
            if s.image not in zoom_imgs:
                zoom_imgs.add(s.image)
                tot["zoomable_images"] += 1
                row["zoomable"] += 1
        else:
            if is_new:
                tot["non_zoomable_images"] += 1
                row["non_zoomable"] += 1
                if long_side < _HARD_MIN_SIDE:
                    tot["too_small_images"] += 1
                    row["too_small"] += 1

    n = tot["n_images"]
    tot["zoomable_ratio"] = round(tot["zoomable_images"] / n, 4) if n else 0.0
    tot["zoom_min_side"] = zoom_min
    tot["min_side_for_zoom_tasks"] = _HARD_MIN_SIDE
    tot["by_dataset"] = {k: dict(v, zoomable_ratio=round(v["zoomable"] / v["n_images"], 4)
                                 if v["n_images"] else 0.0)
                         for k, v in sorted(per_ds.items())}
    return tot


def _sample_size(s: Sample, cache: Dict[str, Optional[Tuple[int, int]]]) -> Optional[Tuple[int, int]]:
    """原图尺寸：(w, h)。优先 meta['orig_size']（数据阶段落过盘），否则 PIL 读。"""
    m = s.meta or {}
    for k in ("orig_size", "original_size", "size"):
        v = m.get(k)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            try:
                return int(v[0]), int(v[1])
            except (TypeError, ValueError):
                pass
    if s.image in cache:
        return cache[s.image]
    size: Optional[Tuple[int, int]] = None
    try:
        from PIL import Image

        with Image.open(s.image) as im:
            size = im.size
    except Exception:  # noqa: BLE001 - 缺图/DICOM 都算读不到
        size = None
    cache[s.image] = size
    return size
