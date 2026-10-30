"""collect.py 单测：五个解析器 + 分发跳过 + manifest + 分辨率审计。

不联网、不读大数据文件：所有数据用 tmp_path 现造（PIL 小图 + 小 json/csv/xml）。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
from PIL import Image

from medtaco.data.collect import (
    PARSERS,
    audit_resolution,
    collect_dataset,
    dataset_root,
    iter_all,
    parse_chestxray14,
    parse_lidc_idri,
    parse_mock,
    parse_ms_cxr,
    parse_slake,
    parse_vindr_cxr,
    scan_manifest,
)
from medtaco.utils.config import Cfg
from medtaco.utils.schema import Sample


# ---------------------------------------------------------------- 造数工具


def _cfg(tmp_path, **data_over) -> Cfg:
    """最小可用 Cfg：paths.raw 指向 tmp_path，其余取 default.yaml 的关键子集。"""
    raw = {
        "seed": 42,
        "paths": {"raw": str(tmp_path / "raw"), "processed": str(tmp_path / "processed")},
        "data": {
            "dedup": {"phash_hamming": 6, "tile_hash": True, "tile_grid": 3, "min_tile_side": 256},
            "resolution": {"zoom_min_side": 1024, "max_side_cap": 4096},
            "commercial_only": False,
            "datasets": {
                "MS-CXR": {"tier": "A", "license": "PhysioNet-DUA", "commercial": False},
                "VinDr-CXR": {"tier": "A", "license": "PhysioNet-DUA", "commercial": False},
                "SLAKE": {"tier": "A", "license": "research-only", "commercial": False},
                "ChestX-ray14": {"tier": "A", "license": "public-domain", "commercial": True},
                "LIDC-IDRI": {"tier": "A", "license": "CC-BY-3.0", "commercial": True},
                "VQA-RAD": {"tier": "B", "license": "research-only", "commercial": False},
                "Mock": {"tier": "A", "license": "synthetic-mock", "commercial": True},
            },
        },
    }
    for k, v in data_over.items():
        raw["data"][k] = v
    return Cfg(raw)


def _noise_png(path, seed: int, size=(64, 64)) -> str:
    """确定性随机噪声图（np.random.default_rng 显式 seed）。"""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(arr).save(path)
    return path


# ---------------------------------------------------------------- MS-CXR


def test_ms_cxr_parse(tmp_path):
    root = tmp_path / "raw" / "MS-CXR"
    (root / "images").mkdir(parents=True)
    _noise_png(str(root / "images" / "img1.png"), seed=1)
    _noise_png(str(root / "images" / "img2.png"), seed=2)

    anns = {
        "0": {"image_id": "img1", "x": 10, "y": 20, "width": 30, "height": 40,
              "label_text": "consolidation", "sentence": "Left lower lobe consolidation."},
        "1": {"image_id": "img2", "x0": 5, "y0": 6, "x1": 25, "y1": 26,
              "label_text": "effusion", "sentence": "Small pleural effusion."},
        "2": {"image_id": "missing_img", "bbox": [1, 2, 3, 4], "label_text": "atelectasis"},
    }
    (root / "mscxr_bbox_annotations.json").write_text(json.dumps(anns), encoding="utf-8")

    cfg = _cfg(tmp_path)
    samples = list(parse_ms_cxr(str(root), cfg))
    assert len(samples) == 3
    s0 = samples[0]
    assert s0.dataset == "MS-CXR" and s0.tier == "A"
    assert s0.boxes[0].as_tuple() == (10.0, 20.0, 40.0, 60.0)
    assert s0.boxes[0].label == "consolidation"
    assert s0.question == "Left lower lobe consolidation."
    assert s0.license == "PhysioNet-DUA" and s0.commercial_ok is False
    assert s0.image.endswith("img1.png") and os.path.exists(s0.image)
    # 缺图仍产出（下游按缺图统计），路径是猜测值
    assert samples[2].image.endswith("missing_img.jpg")
    # id 稳定：重跑同一条记录得到同一 id
    again = list(parse_ms_cxr(str(root), cfg))
    assert [x.id for x in again] == [x.id for x in samples]


def test_ms_cxr_coco_style_wrapper(tmp_path):
    """兼容 {"annotations": [...]} + COCO xywh 四元组写法。"""
    root = tmp_path / "raw" / "MS-CXR"
    root.mkdir(parents=True)
    data = {"annotations": [
        {"image_id": "a", "bbox": [1, 2, 10, 20], "label_text": "x"},
        {"image_id": "b", "bbox": [0, 0, 0, 0], "label_text": "degenerate"},
    ]}
    (root / "annotations.json").write_text(json.dumps(data), encoding="utf-8")
    out = list(parse_ms_cxr(str(root), _cfg(tmp_path)))
    assert len(out) == 1  # 退化框被丢弃
    assert out[0].boxes[0].as_tuple() == (1.0, 2.0, 11.0, 22.0)


# ---------------------------------------------------------------- VinDr-CXR


def test_vindr_parse_groups_by_image_and_finding(tmp_path):
    root = tmp_path / "raw" / "VinDr-CXR"
    (root / "train").mkdir(parents=True)
    _noise_png(str(root / "train" / "im1.jpg"), seed=3)
    csv_text = (
        "image_id,finding_label,x_min,y_min,x_max,y_max,rad_id\n"
        "im1,Nodule/Mass,10,10,50,50,R1\n"
        "im1,Nodule/Mass,12,11,52,51,R2\n"
        "im1,Cardiomegaly,5,5,200,180,R1\n"
        "im2,Atelectasis,1,1,20,20,R1\n"
    )
    (root / "annotations_train.csv").write_text(csv_text, encoding="utf-8")
    out = list(parse_vindr_cxr(str(root), _cfg(tmp_path)))
    assert len(out) == 3  # (im1,Nodule/Mass) / (im1,Cardiomegaly) / (im2,Atelectasis)
    nm = [s for s in out if s.meta["finding"] == "Nodule/Mass"][0]
    assert len(nm.boxes) == 2 and nm.meta["n_rads"] == 2
    assert nm.meta["split"] == "train"
    assert nm.answer_type == "short"


def test_vindr_parse_official_class_name_header(tmp_path):
    """官方 annotations_*.csv 表头用 `class_name`（还带 rad_id / class_id），必须能解析。

    `finding_label` 只存在于早期自造 fixture；真实 csv 曾被整表跳过。
    """
    root = tmp_path / "raw" / "VinDr-CXR"
    (root / "train").mkdir(parents=True)
    _noise_png(str(root / "train" / "im1.jpg"), seed=13)
    csv_text = (
        "image_id,rad_id,class_name,class_id,x_min,y_min,x_max,y_max\n"
        "im1,R1,Nodule/Mass,11,10,10,50,50\n"
        "im1,R2,Nodule/Mass,11,12,11,52,51\n"
        "im1,R1,Cardiomegaly,4,5,5,200,180\n"
        "im2,R3,Atelectasis,0,1,1,20,20\n"
    )
    (root / "annotations_train.csv").write_text(csv_text, encoding="utf-8")
    out = list(parse_vindr_cxr(str(root), _cfg(tmp_path)))
    assert len(out) == 3  # (im1,Nodule/Mass) / (im1,Cardiomegaly) / (im2,Atelectasis)
    # 标签落到 meta["finding"] 与 box.label 上，与旧列名的行为一致
    assert all(s.meta["finding"] and s.boxes for s in out)
    assert all(s.boxes[0].label == s.meta["finding"] for s in out)
    nm = [s for s in out if s.meta["finding"] == "Nodule/Mass"][0]
    assert len(nm.boxes) == 2 and nm.meta["n_rads"] == 2
    assert nm.boxes[0].as_tuple() == (10.0, 10.0, 50.0, 50.0)
    assert nm.boxes[1].as_tuple() == (12.0, 11.0, 52.0, 51.0)
    assert nm.tier == "A" and nm.meta["split"] == "train"
    assert nm.image.endswith("im1.jpg")


def test_vindr_official_header_without_rad_id(tmp_path):
    """缺 rad_id 列也要能跑，n_rads 记 0（多医生计数缺失，不报错）。"""
    root = tmp_path / "raw" / "VinDr-CXR"
    root.mkdir(parents=True)
    csv_text = (
        "image_id,class_name,x_min,y_min,x_max,y_max\n"
        "im9,Effusion,1,1,30,40\n"
    )
    (root / "annotations_train.csv").write_text(csv_text, encoding="utf-8")
    out = list(parse_vindr_cxr(str(root), _cfg(tmp_path)))
    assert len(out) == 1
    assert out[0].meta["finding"] == "Effusion"
    assert out[0].meta["n_rads"] == 0
    assert out[0].boxes[0].as_tuple() == (1.0, 1.0, 30.0, 40.0)


def test_vindr_bad_header_skipped(tmp_path, capsys):
    root = tmp_path / "raw" / "VinDr-CXR"
    root.mkdir(parents=True)
    (root / "annotations_train.csv").write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    # 缺列只 warn 跳过，不能抛异常
    assert list(parse_vindr_cxr(str(root), _cfg(tmp_path))) == []
    assert "[warn]" in capsys.readouterr().out


# ---------------------------------------------------------------- SLAKE


def test_slake_parse_with_inline_box_and_mask(tmp_path):
    root = tmp_path / "raw" / "SLAKE"
    (root / "imgs").mkdir(parents=True)
    (root / "GT_mask").mkdir()
    _noise_png(str(root / "imgs" / "x1.jpg"), seed=4)
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(root / "GT_mask" / "x1.png")
    items = [
        {"qid": 1, "img_name": "x1.jpg", "question": "Does this image show a fracture?",
         "answer": "Yes", "answer_type": "CLOSED", "q_lang": "en",
         "bbox": [10, 10, 60, 60], "content_type": "bone"},
        {"qid": 2, "img_name": "x1.jpg", "question": "图像中是否有骨折？", "answer": "是",
         "answer_type": "OPEN", "q_lang": "zh", "modality": "X-Ray"},
    ]
    (root / "train.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    out = list(parse_slake(str(root), _cfg(tmp_path)))
    assert len(out) == 2
    a, b = out
    # 裸四元组按 COCO 口径 xywh 解释：x+w=70, y+h=70
    assert a.answer_type == "closed" and a.boxes[0].as_tuple() == (10.0, 10.0, 70.0, 70.0)
    assert a.masks and a.masks[0].endswith("x1.png")
    assert b.answer_type == "short" and b.meta["lang"] == "zh"
    assert b.meta["question_zh"] == "图像中是否有骨折？" and b.boxes == []
    assert b.tier == "B"          # A 层数据集里没框的题只能做答案监督
    assert a.meta["split"] == "train"


def test_slake_boxes_from_gt_bbox_dir(tmp_path):
    root = tmp_path / "raw" / "SLAKE"
    (root / "GT_bbox").mkdir(parents=True)
    (root / "GT_bbox" / "y1.json").write_text(
        json.dumps([{"bbox": [1, 2, 11, 22]}]), encoding="utf-8")
    items = [{"qid": 3, "img_name": "y1.jpg", "question": "q", "answer": "a",
              "answer_type": "OPEN", "q_lang": "en"}]
    (root / "test.json").write_text(json.dumps(items), encoding="utf-8")
    out = list(parse_slake(str(root), _cfg(tmp_path)))
    assert out[0].boxes[0].as_tuple() == (1.0, 2.0, 12.0, 24.0)


# ---------------------------------------------------------------- ChestX-ray14


def test_chestxray14_bbox_string_parsing(tmp_path):
    """官方 csv 表头 `Bbox [x,y,w,h]` 未加引号，会被 csv 切成 5 列——解析器必须容忍。"""
    root = tmp_path / "raw" / "ChestX-ray14"
    (root / "images").mkdir(parents=True)
    _noise_png(str(root / "images" / "00000001_000.png"), seed=5)
    csv_text = (
        "Image Index,Finding Label,Bbox [x,y,w,h]\n"
        '00000001_000.png,Atelectasis,"Bbox [100.5,200.2,30.0,40.0]"\n'
        '00000001_000.png,Effusion,"Bbox [1,2,3,4]"\n'
        '00000002_000.png,Nodule,"garbage"\n'
    )
    (root / "BBox_List_2017.csv").write_text(csv_text, encoding="utf-8")
    out = list(parse_chestxray14(str(root), _cfg(tmp_path)))
    assert len(out) == 2
    at = [s for s in out if s.meta["finding"] == "Atelectasis"][0]
    assert at.boxes[0].x0 == pytest.approx(100.5)
    assert at.boxes[0].x1 == pytest.approx(130.5)
    assert at.answer_type == "closed" and at.commercial_ok is True


# ---------------------------------------------------------------- LIDC-IDRI


LIDC_XML = """<?xml version="1.0" encoding="UTF-8"?>
<LidcReadMessage xmlns="http://www.nih.gov">
  <ResponseHeader><SeriesInstanceUID>1.2.3</SeriesInstanceUID></ResponseHeader>
  <readingSession>
    <unblindedReadNodule>
      <noduleID>N1</noduleID>
      <characteristics><malignancy>4</malignancy><subtlety>3</subtlety></characteristics>
      <roi>
        <imageSOP_UID>1.2.840.1</imageSOP_UID><imageZposition>-50.5</imageZposition>
        <edgeMap><xCoord>10</xCoord><yCoord>20</yCoord></edgeMap>
        <edgeMap><xCoord>12</xCoord><yCoord>20</yCoord></edgeMap>
        <edgeMap><xCoord>12</xCoord><yCoord>23</yCoord></edgeMap>
      </roi>
      <roi>
        <imageSOP_UID>1.2.840.2</imageSOP_UID><imageZposition>-48.0</imageZposition>
        <edgeMap><xCoord>1</xCoord><yCoord>1</yCoord></edgeMap>
        <edgeMap><xCoord>5</xCoord><yCoord>1</yCoord></edgeMap>
        <edgeMap><xCoord>5</xCoord><yCoord>5</yCoord></edgeMap>
      </roi>
    </unblindedReadNodule>
    <unblindedReadNodule>
      <noduleID>N2</noduleID>
      <roi><imageSOP_UID>1.2.840.3</imageSOP_UID>
        <edgeMap><xCoord>1</xCoord><yCoord>1</yCoord></edgeMap>
      </roi>
    </unblindedReadNodule>
  </readingSession>
</LidcReadMessage>
"""


def test_lidc_xml_parse(tmp_path):
    root = tmp_path / "raw" / "LIDC-IDRI"
    lp = root / "LIDC-XML-only" / "tcia-lidc-xml"
    lp.mkdir(parents=True)
    (lp / "LIDC-IDRI-0001.xml").write_text(LIDC_XML, encoding="utf-8")
    out = list(parse_lidc_idri(str(root), _cfg(tmp_path)))
    assert len(out) == 2  # N2 只有一个点，退化框被丢弃
    s = out[0]
    assert s.boxes[0].as_tuple() == (10.0, 20.0, 12.0, 23.0)
    assert s.meta["nodule_id"] == "N1" and s.meta["z_position"] == "-50.5"
    assert s.meta["characteristics"]["malignancy"] == "4"
    assert s.meta["characteristics"]["subtlety"] == "3"
    assert s.answer_type == "measure" and s.meta["spacing_available"] is False
    assert s.meta["series"] == "1.2.3"


def test_lidc_xml_cap(tmp_path):
    root = tmp_path / "raw" / "LIDC-IDRI"
    lp = root / "xml"
    lp.mkdir(parents=True)
    for i in range(3):
        (lp / f"a{i}.xml").write_text(LIDC_XML, encoding="utf-8")
    cfg = _cfg(tmp_path, lidc={"max_xml": 1})
    out = list(parse_lidc_idri(str(root), cfg))
    assert len(out) == 2  # 只解析第一个 xml 的两个 roi


# ---------------------------------------------------------------- 分发与跳过


def test_collect_dataset_missing_dir_skips(tmp_path, capsys):
    out = list(collect_dataset("MS-CXR", _cfg(tmp_path)))
    assert out == []
    assert "[skip]" in capsys.readouterr().out


def test_collect_dataset_unsupported_skips(tmp_path, capsys):
    assert list(collect_dataset("MedQA", _cfg(tmp_path))) == []
    assert "[skip]" in capsys.readouterr().out
    # 完全不认识的名字同样要优雅退出
    assert list(collect_dataset("NoSuchDataset", _cfg(tmp_path))) == []


def test_collect_dataset_does_not_raise_on_broken_json(tmp_path, capsys):
    root = tmp_path / "raw" / "MS-CXR"
    root.mkdir(parents=True)
    (root / "mscxr_bbox_annotations.json").write_text("{not json", encoding="utf-8")
    assert list(collect_dataset("MS-CXR", _cfg(tmp_path))) == []
    assert "[warn]" in capsys.readouterr().out


def test_parsers_registry_covers_required_sets():
    for name in ("MS-CXR", "VinDr-CXR", "SLAKE", "ChestX-ray14", "LIDC-IDRI"):
        assert name in PARSERS


def test_dataset_root_override(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.set("data.roots", {"MS-CXR": str(tmp_path / "other")})
    assert dataset_root("MS-CXR", cfg) == str(tmp_path / "other")
    assert dataset_root("SLAKE", cfg) == str(tmp_path / "raw" / "SLAKE")


# ---------------------------------------------------------------- manifest / audit


def _tiny_dataset(tmp_path) -> Cfg:
    """一个含 2 图 3 样本的 MS-CXR 小数据集，供 manifest/audit 用。"""
    root = tmp_path / "raw" / "MS-CXR"
    (root / "images").mkdir(parents=True)
    _noise_png(str(root / "images" / "small1.png"), seed=11, size=(64, 64))
    _noise_png(str(root / "images" / "big1.png"), seed=12, size=(1200, 800))
    anns = {
        "a": {"image_id": "small1", "bbox": [1, 1, 10, 10], "label_text": "x", "sentence": "s1"},
        "b": {"image_id": "big1", "bbox": [1, 1, 10, 10], "label_text": "y", "sentence": "s2"},
        "c": {"image_id": "big1", "bbox": [20, 20, 40, 40], "label_text": "z", "sentence": "s3"},
    }
    (root / "mscxr_bbox_annotations.json").write_text(json.dumps(anns), encoding="utf-8")
    return _cfg(tmp_path)


def test_scan_manifest(tmp_path):
    cfg = _tiny_dataset(tmp_path)
    rows = scan_manifest(cfg, datasets=["MS-CXR"])
    assert len(rows) == 2  # 按图去重
    keys = {r["image"] for r in rows}
    assert all(r["dataset"] == "MS-CXR" and r["license"] == "PhysioNet-DUA" for r in rows)
    assert all(set(r.keys()) == {"dataset", "image", "box_count", "tier", "license"} for r in rows)
    big = [r for r in rows if "big1" in r["image"]][0]
    assert big["box_count"] == 2 and big["tier"] == "A"
    assert keys == {str(tmp_path / "raw" / "MS-CXR" / "images" / "small1.png"),
                    str(tmp_path / "raw" / "MS-CXR" / "images" / "big1.png")}


def test_scan_manifest_limit(tmp_path):
    cfg = _tiny_dataset(tmp_path)
    assert len(scan_manifest(cfg, datasets=["MS-CXR"], limit=1)) == 1


# ---------------------------------------------------------------- Mock


def test_parse_mock_generates_images_and_is_idempotent(tmp_path):
    root = tmp_path / "raw" / "Mock"
    # 目录不存在也要能跑：Mock 是自建数据集
    cfg = _cfg(tmp_path)
    first = list(parse_mock(str(root), cfg))
    assert len(first) == 12
    assert all(os.path.exists(s.image) for s in first)
    with Image.open(first[0].image) as im:
        assert im.size == (1024, 1024)

    # 有框/无框各占一部分，且无框的标了 needs_tool=False（over_call_rate 的分母）
    boxed = [s for s in first if s.boxes]
    free = [s for s in first if not s.boxes]
    assert boxed and free
    assert all(s.meta["needs_tool"] is False for s in free)
    assert all(s.tier == "A" for s in boxed) and all(s.tier == "B" for s in free)

    # 二次调用读 labels.jsonl，不再画图，结果逐条一致
    second = list(parse_mock(str(root), cfg))
    assert [s.id for s in second] == [s.id for s in first]
    assert [s.answer for s in second] == [s.answer for s in first]


def test_parse_mock_constant_is_reproducible(tmp_path):
    """同一 seed 的两次独立生成（不同目录）必须逐字节相同。"""
    a = list(parse_mock(str(tmp_path / "a" / "Mock"), _cfg(tmp_path)))
    b = list(parse_mock(str(tmp_path / "b" / "Mock"), _cfg(tmp_path)))
    assert [s.answer for s in a] == [s.answer for s in b]
    with open(a[0].image, "rb") as f1, open(b[0].image, "rb") as f2:
        assert f1.read() == f2.read()


def test_mock_is_self_creating_via_collect_dataset(tmp_path):
    """collect_dataset 对 Mock 不做 isdir 前置检查。"""
    cfg = _cfg(tmp_path)
    out = list(collect_dataset("Mock", cfg))
    assert len(out) == 12 and all(s.dataset == "Mock" for s in out)


def test_audit_resolution(tmp_path):
    cfg = _tiny_dataset(tmp_path)
    samples = list(iter_all(cfg, datasets=["MS-CXR"]))
    rep = audit_resolution(samples, cfg)
    assert rep["n_samples"] == 3 and rep["n_images"] == 2
    assert rep["zoomable_images"] == 1        # big1 长边 1200 ≥ 1024
    assert rep["non_zoomable_images"] == 1
    assert rep["too_small_images"] == 1       # small1 长边 64 < 512
    assert rep["zoomable_ratio"] == 0.5
    assert rep["by_dataset"]["MS-CXR"]["zoomable"] == 1


def test_audit_resolution_prefers_meta_orig_size(tmp_path):
    s = Sample(id="1", image=str(tmp_path / "nope.png"), question="", answer="",
               dataset="X", meta={"orig_size": [3000, 2000]})
    rep = audit_resolution([s], _cfg(tmp_path))
    assert rep["zoomable_images"] == 1 and rep["unreadable_images"] == 0


def test_audit_resolution_unreadable(tmp_path):
    s = Sample(id="1", image=str(tmp_path / "missing.png"), question="", answer="", dataset="X")
    rep = audit_resolution([s], _cfg(tmp_path))
    assert rep["unreadable_images"] == 1 and rep["zoomable_images"] == 0
