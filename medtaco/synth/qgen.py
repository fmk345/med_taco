"""阶段 2：五类问题生成器 + 问题级三道过滤闸（PROPOSAL §4）。

五类生成器（与 PROPOSAL §4.1 一一对应）：
  - G1 框锚定：问题里显式出现"标记区域 / 该位置 / 框内"，裁剪被问题**构造性蕴含**；
  - G2 多尺度：构造 2~3 级嵌套框，训练"全图 → 器官 → 病灶"的多轮放大；
  - G3 负样本：全局性 / 知识性问题，**不需要任何工具**，靠配比压到 ~40%；
  - G4 测量：由框角点距离 × pixel_spacing 派生 mm 真值，监督 caliper；
  - G5 对比：要求同时用全图上下文与局部裁剪（对侧镜像框）。

三条硬性设计约束：
  1. **无 model_fn 也必须能产出可用问题**：模板库中英双语、每类 >= 8 条，纯本地；
     `model_fn(prompt, images) -> str` 只是可选的改写注入点，异常/空串/丢失锚点时
     一律回落到模板，绝不把外部调用写死在本模块里。
  2. **所有随机性来自显式 seed**（`cfg.seed` / `cfg.qgen.seed`），同 seed 两次结果逐条一致。
  3. 每条产出 Sample 的 `meta` 里固定带 `generator / is_negative / template_id`，
     另附 `source_sample_id / template_lang / yes_no / need_crop`，便于复盘与消融。

产出约定：`answer` 一般为空串（留给教师模型填），**只有 G4 例外**——其答案是
框角点距离 × spacing 的正确派生值，可直接用于确定性判分。
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from medtaco.utils.schema import Box, Sample

__all__ = [
    "Template",
    "TEMPLATES",
    "generate",
    "filter_questions",
    "filter_questions_with_audit",
    "question_stats",
    "FilterAudit",
    "g1_box_anchored",
    "g2_multi_scale",
    "g3_negative",
    "g4_measure",
    "g5_compare",
]

# 注入点签名。model_fn 返回改写过的问题文本；agree_fn 判两个答案是否一致；
# pass8_fn 返回"无工具 pass@8 是否全对"；with_box_fn 返回"给 gold 框后能否答对"。
ModelFn = Callable[[str, List[str]], str]
AgreeFn = Callable[[str, str], bool]
PassFn = Callable[[Sample], bool]

# ---------------------------------------------------------------- 常量

# 允许出现在模板里的占位符。模板文本只认这四个，其余一律当字面量。
PLACEHOLDERS: Tuple[str, ...] = ("finding", "location", "box_label", "organ")

DEFAULT_MIX: Dict[str, float] = {"G1": 0.40, "G2": 0.20, "G3": 0.25, "G4": 0.08, "G5": 0.07}

ALL_GENS: Tuple[str, ...] = ("G1", "G2", "G3", "G4", "G5")
# 需要用工具（裁剪 / 放大 / 测量）的生成器；G3 是负样本，只在全图上作答。
POSITIVE_GENS: Tuple[str, ...] = ("G1", "G2", "G4", "G5")

_MISSING = object()


# ---------------------------------------------------------------- 模板库

@dataclass(frozen=True)
class Template:
    """一条问题模板。`text` 里只允许 PLACEHOLDERS 里的占位符。

    yes_no:  该模板产出的是二元（是/否）问题，受 yes_no_max_ratio 压制；
    mode:    G4 专用，决定用框的哪种尺寸派生 mm 真值；
    requires_spacing: 该模板必须样本带 pixel_spacing 才可用（目前仅 G4）。
    modality: 该模板成立所需的成像模态（如 "CT"）；样本模态不符时不使用。
    """

    tid: str
    gen: str
    lang: str
    text: str
    yes_no: bool = False
    mode: str = "long"
    requires_spacing: bool = False
    modality: Optional[str] = None


def _t(gen: str, lang: str, idx: int, text: str, *, yes_no: bool = False,
       mode: str = "long", requires_spacing: bool = False,
       modality: Optional[str] = None) -> Template:
    return Template(
        tid=f"{gen}-{lang}-{idx:02d}",
        gen=gen,
        lang=lang,
        text=text,
        yes_no=yes_no,
        mode=mode,
        requires_spacing=requires_spacing,
        modality=modality,
    )


# G1：每一条都显式指向"标记区域 / 框内 / 该位置"，裁掉外围不会让问题失去指代对象。
_G1: List[Template] = [
    _t("G1", "zh", 1, "请描述图中标记区域内{finding}的边缘与形态特征。"),
    _t("G1", "zh", 2, "图中标记的{box_label}位于{location}，该区域内的病变有哪些影像学表现？"),
    _t("G1", "zh", 3, "图中标记区域内的{finding}边缘是否光滑？", yes_no=True),
    _t("G1", "zh", 4, "请给出标记框内{finding}的密度、大小与边界描述。"),
    _t("G1", "zh", 5, "图中标记框内的{finding}与周围组织相比，边界是否清晰？", yes_no=True),
    _t("G1", "zh", 6, "图中被框出的区域对应哪个解剖结构？"),
    _t("G1", "zh", 7, "标记区域内的{organ}存在哪些异常征象？"),
    _t("G1", "zh", 8, "请判断图中标记区域内{finding}的良恶性征象，并说明依据。"),
    _t("G1", "zh", 9, "标记区域内是否可见钙化、空洞或毛刺？", yes_no=True),
    _t("G1", "zh", 10, "请描述图中标记区域病灶相对于周围正常组织的位置关系。"),
    _t("G1", "en", 1, "Describe the margins and shape of the {finding} within the marked region."),
    _t("G1", "en", 2, "What is the morphology of the abnormality in the highlighted region?"),
    _t("G1", "en", 3, "Is the lesion inside the boxed area well circumscribed?", yes_no=True),
    _t("G1", "en", 4, "Describe the internal density of the {finding} in the marked area."),
    _t("G1", "en", 5, "Does the boxed region contain calcification or spiculation?", yes_no=True),
    _t("G1", "en", 6, "Which anatomical structure does the marked region correspond to?"),
    _t("G1", "en", 7, "Characterize the {finding} in the highlighted area in terms of size and margin."),
    _t("G1", "en", 8, "What abnormal findings are visible inside the marked region of the {organ}?"),
    _t("G1", "en", 9, "Are the borders of the {finding} in the marked region sharp or ill-defined?", yes_no=True),
    _t("G1", "en", 10, "Describe how the lesion inside the marked box relates to the surrounding tissue."),
]

# G2：显式给出"全图 → 器官/区域 → 病灶"的层级指令，配套嵌套框。
_G2: List[Template] = [
    _t("G2", "zh", 1, "先从全图中定位{organ}，再放大到{location}的病灶，描述其密度。"),
    _t("G2", "zh", 2, "请按“全图 → {organ} → 病灶”的顺序，描述{location}标记区域内{finding}的边缘。"),
    _t("G2", "zh", 3, "在全图中找到{location}，再描述该处{box_label}内部病灶的形态。"),
    _t("G2", "zh", 4, "先确定{organ}的范围，再比较其中最大病灶与其余病灶的差异。"),
    _t("G2", "zh", 5, "从整幅图像开始逐步聚焦到{location}的{finding}，说明其内部结构。"),
    _t("G2", "zh", 6, "图中{location}的病灶相对于{organ}整体呈何种分布？"),
    _t("G2", "zh", 7, "先在全局尺度判断{organ}是否对称，再放大{location}描述局部异常。"),
    _t("G2", "zh", 8, "请描述从全图到标记框逐级放大过程中，{finding}呈现的关键征象。"),
    _t("G2", "zh", 9, "定位{location}的{box_label}后，描述其与相邻结构的关系。"),
    _t("G2", "en", 1, "First locate the {organ} in the full image, then zoom into the lesion at {location} and describe its density."),
    _t("G2", "en", 2, "Following a full-image -> {organ} -> lesion order, describe the margins of the {finding} in the marked region at {location}."),
    _t("G2", "en", 3, "Find {location} in the full image, then describe the morphology of the lesion inside the {box_label} there."),
    _t("G2", "en", 4, "Determine the extent of the {organ} first, then compare the largest lesion with the remaining ones."),
    _t("G2", "en", 5, "Starting from the whole image, progressively focus on the {finding} at {location} and describe its internal structure."),
    _t("G2", "en", 6, "How are the lesions at {location} distributed relative to the whole {organ}?"),
    _t("G2", "en", 7, "Assess the symmetry of the {organ} at the global scale, then zoom into {location} to describe the local abnormality."),
    _t("G2", "en", 8, "Describe the key signs of the {finding} that emerge while zooming from the full image to the marked box."),
    _t("G2", "en", 9, "After locating the {box_label} at {location}, describe its relationship with adjacent structures."),
]

# G3：全局性 / 知识性问题。**不得**出现"标记 / 框 / 该位置"这类指代，
# 也不应依赖任何工具——模型只见整图即可作答。
_G3: List[Template] = [
    _t("G3", "zh", 1, "该影像属于哪种成像模态（X 线、CT、MRI 或超声）？"),
    _t("G3", "zh", 2, "这幅图像的整体体位或扫描方位是什么？"),
    _t("G3", "zh", 3, "从影像整体看，最可能的诊断方向是什么？"),
    _t("G3", "zh", 4, "在 CT 图像中，空气的 CT 值大约是多少 HU？", modality="CT"),
    _t("G3", "zh", 5, "肺水肿在胸部 X 线上的典型表现是什么？"),
    _t("G3", "zh", 6, "气胸的典型影像学征象有哪些？"),
    _t("G3", "zh", 7, "这幅图像是否存在明显的伪影或体位不正？", yes_no=True),
    _t("G3", "zh", 8, "该检查最可能需要使用对比剂吗？", yes_no=True),
    _t("G3", "zh", 9, "请概括这幅医学影像的整体质量与可读性。"),
    _t("G3", "zh", 10, "骨折在 X 线上的直接征象包括哪些？"),
    _t("G3", "en", 1, "Which imaging modality does this study belong to (X-ray, CT, MRI, or ultrasound)?"),
    _t("G3", "en", 2, "What is the overall projection or scanning orientation of this image?"),
    _t("G3", "en", 3, "Judging from the overall appearance, what is the most likely diagnostic category?"),
    _t("G3", "en", 4, "On a CT image, what is the approximate Hounsfield unit value of air?", modality="CT"),
    _t("G3", "en", 5, "What are the typical radiographic signs of pulmonary edema on chest X-ray?"),
    _t("G3", "en", 6, "What are the classic imaging signs of pneumothorax?"),
    _t("G3", "en", 7, "Does this image show obvious artifacts or incorrect positioning?", yes_no=True),
    _t("G3", "en", 8, "Is this examination likely to require contrast administration?", yes_no=True),
    _t("G3", "en", 9, "Summarize the overall quality and readability of this medical image."),
    _t("G3", "en", 10, "What are the direct radiographic signs of a fracture on plain film?"),
]

# G4：答案由框角点距离 × pixel_spacing 派生，answer_type='measure'。
_G4: List[Template] = [
    _t("G4", "zh", 1, "请测量图中标记病灶的长径，单位为 mm。", mode="long", requires_spacing=True),
    _t("G4", "zh", 2, "标记区域的最大径线是多少毫米？", mode="long", requires_spacing=True),
    _t("G4", "zh", 3, "请给出框内病灶短径的测量值（mm）。", mode="short", requires_spacing=True),
    _t("G4", "zh", 4, "图中标记区域的横径（mm）是多少？", mode="long", requires_spacing=True),
    _t("G4", "zh", 5, "请测量标记病灶的对角线长度（mm）。", mode="diag", requires_spacing=True),
    _t("G4", "zh", 6, "标记区域的面积约为多少平方毫米？", mode="area", requires_spacing=True),
    _t("G4", "zh", 7, "请给出标记区域最长径与最短径的估计（mm）。", mode="long", requires_spacing=True),
    _t("G4", "zh", 8, "按实际物理尺寸，标记病灶的纵向长度是多少毫米？", mode="long", requires_spacing=True),
    _t("G4", "en", 1, "Measure the long-axis diameter of the marked lesion in mm.", mode="long", requires_spacing=True),
    _t("G4", "en", 2, "What is the maximum diameter of the marked region in millimeters?", mode="long", requires_spacing=True),
    _t("G4", "en", 3, "Give the short-axis diameter of the lesion inside the box (mm).", mode="short", requires_spacing=True),
    _t("G4", "en", 4, "What is the diagonal length of the boxed lesion in mm?", mode="diag", requires_spacing=True),
    _t("G4", "en", 5, "Estimate the area of the marked region in square millimeters.", mode="area", requires_spacing=True),
    _t("G4", "en", 6, "Measure the lateral extent of the marked abnormal area in mm.", mode="long", requires_spacing=True),
    _t("G4", "en", 7, "What physical size (mm) does the marked lesion have along its longest dimension?", mode="long", requires_spacing=True),
    _t("G4", "en", 8, "Report the largest caliper measurement of the highlighted lesion in mm.", mode="long", requires_spacing=True),
]

# G5：必须同时用到全图上下文与局部裁剪；对侧镜像框作为辅助区域。
_G5: List[Template] = [
    _t("G5", "zh", 1, "与对侧相应位置相比，图中标记区域是否出现新发或增大的病灶？", yes_no=True),
    _t("G5", "zh", 2, "对比全图与标记区域，该病灶的密度是否高于对侧相应部位？", yes_no=True),
    _t("G5", "zh", 3, "结合全图上下文，描述标记区域与对侧结构的不对称之处。"),
    _t("G5", "zh", 4, "请同时参考全图与标记区域，描述该异常相对于{organ}的分布是否对称。"),
    _t("G5", "zh", 5, "对侧相应区域的表现有何不同？请具体描述差异。"),
    _t("G5", "zh", 6, "在全图对照下，标记病灶是否引起周围结构牵拉或变形？", yes_no=True),
    _t("G5", "zh", 7, "标记区域内的{finding}与对侧正常组织相比，边界和密度有何差异？"),
    _t("G5", "zh", 8, "结合全图定位{location}，说明标记区域病灶是孤立性还是弥漫性分布。"),
    _t("G5", "en", 1, "Compared with the contralateral side, is the marked region showing a new or enlarging lesion?", yes_no=True),
    _t("G5", "en", 2, "Using the full image as context, is the lesion in the marked region denser than the contralateral counterpart?", yes_no=True),
    _t("G5", "en", 3, "With the whole image in view, describe the asymmetry between the marked region and the contralateral structures."),
    _t("G5", "en", 4, "Referring to both the full image and the marked crop, describe whether the abnormality is symmetric within the {organ}."),
    _t("G5", "en", 5, "How does the contralateral counterpart differ? Describe the difference specifically."),
    _t("G5", "en", 6, "Does the marked lesion distort or retract surrounding structures when compared with the full image?", yes_no=True),
    _t("G5", "en", 7, "How do the margins and density of the {finding} in the marked region differ from normal contralateral tissue?"),
    _t("G5", "en", 8, "Locate {location} in the full image, then state whether the lesion in the marked region is focal or diffuse."),
]

TEMPLATES: Dict[str, List[Template]] = {"G1": _G1, "G2": _G2, "G3": _G3, "G4": _G4, "G5": _G5}

# 各语言的兜底词：占位符取不到值时用它，保证句子语法完整。
_FALLBACK: Dict[str, Dict[str, str]] = {
    "zh": {"finding": "病灶", "location": "图像中部", "box_label": "该标注区域", "organ": "该解剖区域"},
    "en": {"finding": "lesion", "location": "the image center", "box_label": "the marked region", "organ": "the anatomical region"},
}

# G1 改写后必须仍能指代"标记区域"，否则裁剪就不再被问题蕴含，改写作废。
_G1_ANCHOR = re.compile(
    r"(标记|标注|被框|框内|框出|框选|选中|该位置|该区域|此区域|marked|boxed|highlighted|box|region|area)",
    re.I,
)

_YES_NO_ZH = re.compile(r"(是否|有没有|有无|是不是|吗\s*[？?]?\s*$)")
_YES_NO_EN = re.compile(
    r"^\s*(is|are|does|do|was|were|has|have|can|should|could|will|would)\b|is there\b|are there\b",
    re.I,
)

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


# ---------------------------------------------------------------- 配置小工具

def _cget(cfg: Any, path: str, default: Any = None) -> Any:
    """同时兼容 `Cfg`（点号路径）与普通嵌套 dict。"""
    if cfg is None:
        return default
    try:
        v = cfg.get(path, _MISSING)  # Cfg.get 支持 "a.b.c"
    except TypeError:
        v = _MISSING
    if v is not _MISSING and v is not None:
        return v
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _seed(cfg: Any) -> int:
    v = _cget(cfg, "qgen.seed", _MISSING)
    if v is _MISSING or v is None:
        v = _cget(cfg, "seed", 42)
    return int(v if v is not None else 42)


def _rng(seed: int, salt: str) -> random.Random:
    """按 (seed, salt) 造独立随机源；str 种子在 Py3 下走 sha512，跨进程稳定。"""
    return random.Random(f"{seed}|{salt}")


def _enabled_generators(cfg: Any) -> List[str]:
    raw = _cget(cfg, "qgen.generators", list(ALL_GENS))
    if not raw:
        return list(ALL_GENS)
    if isinstance(raw, str):
        raw = [raw]
    wanted = {str(g).upper() for g in raw}
    return [g for g in ALL_GENS if g in wanted]


def _mix_weights(cfg: Any) -> Dict[str, float]:
    raw = _cget(cfg, "qgen.mix", {}) or {}
    out = dict(DEFAULT_MIX)
    if isinstance(raw, dict):
        for k, v in raw.items():
            k = str(k).upper()
            if k in out:
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    pass
    return out


def _target_total(n_samples: int, cfg: Any) -> int:
    """总候选量 = per_sample × 样本数，并受 qgen.target_prompts 封顶。

    per_sample 未配置时默认 5（40k 目标 / 150k 候选 ≈ 每样本几条），
    target_prompts 在本模块里只当**上限**用：真正的 35k–50k 是过滤后的量。
    """
    per_sample = _cget(cfg, "qgen.per_sample", _MISSING)
    if per_sample is _MISSING or per_sample is None:
        per_sample = 5
    try:
        total = int(round(float(per_sample) * n_samples))
    except (TypeError, ValueError):
        total = 5 * n_samples
    cap = _cget(cfg, "qgen.target_prompts", _MISSING)
    if cap is not _MISSING and cap is not None:
        try:
            total = min(total, int(cap))
        except (TypeError, ValueError):
            pass
    return max(1, total) if n_samples > 0 else 0


# ---------------------------------------------------------------- 样本字段派生

def _clone_meta(base: Sample) -> Dict[str, Any]:
    """浅拷贝 meta（嵌套 dict 再拷一层），避免污染输入样本。"""
    out: Dict[str, Any] = {}
    for k, v in base.meta.items():
        out[k] = dict(v) if isinstance(v, dict) else v
    return out


def _image_size(sample: Sample) -> Optional[Tuple[int, int]]:
    """从 meta 里尽力取原图 (w, h)；取不到返回 None（此时位置短语退化为通用说法）。"""
    m = sample.meta
    for key in ("image_size", "size", "orig_size", "width_height"):
        v = m.get(key)
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            try:
                w, h = int(v[0]), int(v[1])
                if w > 0 and h > 0:
                    return w, h
            except (TypeError, ValueError):
                pass
    w = h = None
    for key in ("width", "image_width", "w"):
        if isinstance(m.get(key), (int, float)) and m[key] > 0:
            w = int(m[key])
            break
    for key in ("height", "image_height", "h"):
        if isinstance(m.get(key), (int, float)) and m[key] > 0:
            h = int(m[key])
            break
    if w and h:
        return w, h
    return None


def _spacing(sample: Sample) -> Optional[Tuple[float, float]]:
    """取 (sx, sy) mm/px。约定与 tools/ops.py 一致：pixel_spacing=(sx, sy)。"""
    m = sample.meta
    for key in ("pixel_spacing", "spacing", "PixelSpacing"):
        v = m.get(key)
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            try:
                sx, sy = float(v[0]), float(v[1])
            except (TypeError, ValueError):
                continue
            if sx > 0 and sy > 0:
                return sx, sy
    return None


def _first_box(sample: Sample) -> Optional[Box]:
    for b in sample.boxes:
        if b.w > 0 and b.h > 0:
            return b
    return None


def _location_phrase(sample: Sample, box: Optional[Box], lang: str) -> str:
    """把框中心映射成"图像左上部 / the upper-left part of the image"。"""
    size = _image_size(sample)
    if size is None:
        cx = cy = 0.5
        known = False
    else:
        w, h = size
        if box is not None:
            cx = ((box.x0 + box.x1) / 2) / w
            cy = ((box.y0 + box.y1) / 2) / h
        else:
            cx, cy = 0.5, 0.5
        known = True
    if not known and box is None:
        return "整幅图像" if lang == "zh" else "the whole image"

    vpart = "upper" if cy < 1 / 3 else ("middle" if cy < 2 / 3 else "lower")
    hpart = "left" if cx < 1 / 3 else ("center" if cx < 2 / 3 else "right")
    if lang == "zh":
        vzh = {"upper": "上", "middle": "中", "lower": "下"}[vpart]
        hzh = {"left": "左", "center": "中", "right": "右"}[hpart]
        return f"图像{hzh}{vzh}部"
    if hpart == "center" and vpart == "middle":
        return "the central part of the image"
    return f"the {vpart}-{hpart} part of the image"


def _fields(sample: Sample, box: Optional[Box], lang: str) -> Dict[str, str]:
    """组装模板占位符取值；任何缺失都回落到该语言的兜底词。"""
    fb = _FALLBACK[lang]
    label = (box.label if box is not None and box.label else "").strip()
    meta_finding = str(sample.meta.get("finding", "") or "").strip()
    meta_organ = str(sample.meta.get("organ", "") or "").strip()
    finding = meta_finding or label or fb["finding"]
    organ = meta_organ or label or fb["organ"]
    box_label = label or fb["box_label"]
    return {
        "finding": finding,
        "location": _location_phrase(sample, box, lang),
        "box_label": box_label,
        "organ": organ,
    }


def _fill(text: str, fields: Dict[str, str]) -> str:
    """只替换白名单占位符；未知占位符原样保留（便于测试发现拼写错误）。"""
    def repl(m: "re.Match[str]") -> str:
        key = m.group(1)
        if key not in PLACEHOLDERS:
            return m.group(0)
        return fields.get(key, "")

    return _PLACEHOLDER_RE.sub(repl, text).strip()


def _looks_yes_no(text: str) -> bool:
    return bool(_YES_NO_ZH.search(text) or _YES_NO_EN.search(text))


#: 开放式问题的问法标志（中英双语），命中则判为 long
_LONG_MARKERS: Tuple[str, ...] = (
    "describe", "what are", "summar", "list", "explain",
    "描述", "给出", "有哪些", "包括哪些", "概括", "简述", "说明",
)


def _answer_type(gen: str, yes_no: bool, base_type: str, text: str = "") -> str:
    """由**问题本身**决定答案形态，而不是继承源样本的 answer_type。

    之前直接回退到 base_type，而 ChestX-ray14 的源样本全是 closed，
    导致 3620 条问题无一例外被标成 closed——"描述病灶形态""有哪些典型征象"
    这类开放式问题被当成是/否题去判分，SFT 与 RL 的判分器都会错位。
    """
    if gen == "G4":
        return "measure"
    if yes_no:
        return "closed"
    t = (text or "").lower()
    if any(m in t for m in _LONG_MARKERS):
        return "long"
    if base_type in ("short", "long"):
        return base_type
    return "short"


def _make(
    base: Sample,
    *,
    qid: str,
    question: str,
    gen: str,
    tpl: Template,
    is_negative: bool,
    yes_no: bool,
    boxes: Optional[List[Box]] = None,
    answer: str = "",
    lang: Optional[str] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Sample:
    """基于原样本派生一条问题样本，meta 中固定写复盘所需的四个键。"""
    meta = _clone_meta(base)
    meta.update(extra_meta or {})
    meta.update({
        "generator": gen,
        "is_negative": bool(is_negative),
        "template_id": tpl.tid,
        "template_lang": lang or tpl.lang,
        "yes_no": bool(yes_no),
        "source_sample_id": base.id,
        "need_crop": not is_negative,
        "rewritten": False,
    })
    return Sample(
        id=qid,
        image=base.image,
        question=question,
        answer=answer,
        dataset=base.dataset,
        tier=base.tier,
        answer_type=_answer_type(gen, yes_no, base.answer_type, question),
        boxes=list(base.boxes if boxes is None else boxes),
        masks=list(base.masks),
        license=base.license,
        commercial_ok=base.commercial_ok,
        meta=meta,
    )


# ---------------------------------------------------------------- model_fn 改写

def _rewrite_prompt(base_q: str, gen: str, lang: str) -> str:
    lang_word = "中文" if lang == "zh" else "英文"
    anchor_rule = (
        "\n4) 必须保留对“标记区域 / 该位置”的明确指代，不得改成脱离标记的问法。"
        if gen == "G1" else ""
    )
    return (
        "你是医学影像数据标注员。请改写下面这条问题，要求：\n"
        "1) 不改变所问内容与预期答案类型（是/否题仍是是/否题）；\n"
        f"2) 保持{lang_word}；\n"
        "3) 只输出改写后的问题本身，不要解释、不要加引号。"
        f"{anchor_rule}\n"
        f"原问题：{base_q}"
    )


def _clean_model_output(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    s = text.strip()
    if not s:
        return ""
    # 去掉常见的“问题：/Question:”前缀与整体包裹的引号。
    s = re.sub(r"^\s*(问题|改写后的问题|question)\s*[:：]\s*", "", s, flags=re.I)
    s = s.strip().strip("“”\"'").strip()
    if "\n" in s:  # 模型多说了几行 -> 取第一行非空
        s = next((ln.strip() for ln in s.splitlines() if ln.strip()), "")
    return s


def _maybe_rewrite(
    model_fn: Optional[ModelFn],
    base_q: str,
    sample: Sample,
    gen: str,
    lang: str,
) -> Tuple[str, bool]:
    """调模型改写；失败/空串/G1 丢失锚点 一律回落模板原文，返回 (问题, 是否改写)。"""
    if model_fn is None:
        return base_q, False
    try:
        raw = model_fn(_rewrite_prompt(base_q, gen, lang), [sample.image])
    except Exception:
        return base_q, False
    out = _clean_model_output(raw)
    if not out:
        return base_q, False
    if gen == "G1" and not _G1_ANCHOR.search(out):
        return base_q, False
    return out, True


# ---------------------------------------------------------------- 各级框构造

def _expand(box: Box, factor: float, size: Optional[Tuple[int, int]]) -> Box:
    """以框中心为原点按 factor 放大，并按原图尺寸裁剪。"""
    cx, cy = box.center
    hw, hh = box.w * factor / 2.0, box.h * factor / 2.0
    x0, y0, x1, y1 = cx - hw, cy - hh, cx + hw, cy + hh
    if size is not None:
        w, h = size
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(w), x1), min(float(h), y1)
    else:
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = max(x0 + 1.0, x1), max(y0 + 1.0, y1)
    return Box(x0, y0, x1, y1, box.label)


def _mirror(box: Box, size: Optional[Tuple[int, int]]) -> Optional[Box]:
    """沿图像垂直中线镜像，用于 G5 的"对侧"参照框。"""
    if size is None:
        return None
    w, _ = size
    return Box(max(0.0, w - box.x1), box.y0, min(float(w), w - box.x0), box.y1, box.label)


def _measure(box: Box, spacing: Tuple[float, float], mode: str) -> Tuple[float, str]:
    """由框角点距离 × spacing 派生物理尺寸，返回 (数值, 单位)。"""
    sx, sy = spacing
    w_mm, h_mm = box.w * sx, box.h * sy
    if mode == "long":
        return max(w_mm, h_mm), "mm"
    if mode == "short":
        return min(w_mm, h_mm), "mm"
    if mode == "diag":
        return math.hypot(w_mm, h_mm), "mm"
    if mode == "area":
        return w_mm * h_mm, "mm^2"
    return max(w_mm, h_mm), "mm"


# ---------------------------------------------------------------- G1..G5

def _iter_templates(gen: str, lang: Optional[str] = None,
                    sample: Optional[Sample] = None) -> Iterable[Template]:
    """按语言与样本模态筛模板。

    模态不符的模板必须剔除：在胸片上问"CT 值是多少 HU"，教师只能答
    "这是 X 光片，拿不到 CT 值"——这种题进训练集只会教模型答非所问。
    """
    mod = _sample_modality(sample) if sample is not None else None
    for tpl in TEMPLATES[gen]:
        if lang is not None and tpl.lang != lang:
            continue
        if tpl.modality and tpl.modality.upper() != (mod or "").upper():
            continue
        yield tpl


def _sample_modality(sample: Optional[Sample]) -> Optional[str]:
    """样本模态：优先 meta.modality，其次按数据集名推断，最后看答案类型。"""
    if sample is None:
        return None
    m = str((sample.meta or {}).get("modality", "") or "").strip()
    if m:
        return m
    ds = str(sample.dataset or "").lower()
    for key, mod in (
        ("chestx", "X-ray"), ("cxr", "X-ray"), ("padchest", "X-ray"),
        ("slake", "X-ray"), ("vindr-cxr", "X-ray"), ("ms-cxr", "X-ray"),
        ("ct", "CT"), ("deep", "CT"), ("lidc", "CT"),
        ("mammo", "Mammography"), ("busi", "Ultrasound"), ("idrid", "Fundus"),
    ):
        if key in ds:
            return mod
    return None


def g1_box_anchored(samples: List[Sample], cfg: Any, model_fn: Optional[ModelFn] = None) -> List[Sample]:
    """G1 框锚定：无 box 的样本直接跳过（锚定不了就没有"构造性蕴含"）。

    A 层核心产能。产出样本沿用原 boxes，并在 meta 里记 need_crop=True。
    """
    out: List[Sample] = []
    seq = 0
    for s in samples:
        box = _first_box(s)
        if box is None:
            continue
        for tpl in _iter_templates("G1", sample=s):
            q = _fill(tpl.text, _fields(s, box, tpl.lang))
            q, rewritten = _maybe_rewrite(model_fn, q, s, "G1", tpl.lang)
            yes_no = _looks_yes_no(q) if rewritten else tpl.yes_no
            seq += 1
            item = _make(
                s,
                qid=f"{s.id}#{tpl.tid}",
                question=q,
                gen="G1",
                tpl=tpl,
                is_negative=False,
                yes_no=yes_no,
                boxes=list(s.boxes),
                extra_meta={"anchor_box": box.to_json(), "need_crop": True},
            )
            item.meta["rewritten"] = rewritten
            out.append(item)
    return out


def g2_multi_scale(samples: List[Sample], cfg: Any, model_fn: Optional[ModelFn] = None) -> List[Sample]:
    """G2 多尺度：构造 2~3 级嵌套框（外层放大框 → … → gold 框）。"""
    nest_cfg = _cget(cfg, "qgen.multi_scale_nest", [2, 3]) or [2, 3]
    try:
        nests = sorted({int(x) for x in nest_cfg if int(x) >= 2}) or [2, 3]
    except (TypeError, ValueError):
        nests = [2, 3]
    factors = (4.0, 2.0, 1.0)  # 与 nest=2/3 对齐：取末尾若干级

    out: List[Sample] = []
    seq = 0
    for i, s in enumerate(samples):
        box = _first_box(s)
        if box is None:
            continue
        size = _image_size(s)
        nest = nests[i % len(nests)]
        levels: List[Box] = [_expand(box, f, size) for f in factors[-nest:]]
        levels[-1] = box  # 最内层必须是 gold 框本身
        scale_boxes = [b.to_json() for b in levels]
        for tpl in _iter_templates("G2", sample=s):
            q = _fill(tpl.text, _fields(s, box, tpl.lang))
            q, rewritten = _maybe_rewrite(model_fn, q, s, "G2", tpl.lang)
            yes_no = _looks_yes_no(q) if rewritten else tpl.yes_no
            seq += 1
            item = _make(
                s,
                qid=f"{s.id}#{tpl.tid}",
                question=q,
                gen="G2",
                tpl=tpl,
                is_negative=False,
                yes_no=yes_no,
                boxes=levels,
                extra_meta={
                    "scale_boxes": scale_boxes,
                    "n_scales": nest,
                    "anchor_box": box.to_json(),
                    "need_crop": True,
                },
            )
            item.meta["rewritten"] = rewritten
            out.append(item)
    return out


def g3_negative(samples: List[Sample], cfg: Any, model_fn: Optional[ModelFn] = None) -> List[Sample]:
    """G3 负样本：全局/知识性问题，不需要任何工具。

    产出样本 **清空 boxes**，避免下游误以为需要裁剪；B 层（无区域标注）样本
    也可以进这个池子，这正是 PROPOSAL §4.1 对 B 层数据的用法。
    """
    out: List[Sample] = []
    seq = 0
    for s in samples:
        for tpl in _iter_templates("G3", sample=s):
            q = _fill(tpl.text, _fields(s, None, tpl.lang))
            q, rewritten = _maybe_rewrite(model_fn, q, s, "G3", tpl.lang)
            yes_no = _looks_yes_no(q) if rewritten else tpl.yes_no
            seq += 1
            item = _make(
                s,
                qid=f"{s.id}#{tpl.tid}",
                question=q,
                gen="G3",
                tpl=tpl,
                is_negative=True,
                yes_no=yes_no,
                boxes=[],
                extra_meta={"need_crop": False, "tool_free": True},
            )
            item.meta["rewritten"] = rewritten
            out.append(item)
    return out


def g4_measure(samples: List[Sample], cfg: Any, model_fn: Optional[ModelFn] = None) -> List[Sample]:
    """G4 测量：仅对带 pixel_spacing 且有 gold 框的样本可用。

    答案是**确定性派生**的（框角点距离 × spacing），因此直接填 `answer`，
    answer_type='measure'，可进 rl/grader 的 measure 判分分支。
    """
    out: List[Sample] = []
    seq = 0
    for s in samples:
        box = _first_box(s)
        spacing = _spacing(s)
        if box is None or spacing is None:
            continue
        for tpl in _iter_templates("G4", sample=s):
            value, unit = _measure(box, spacing, tpl.mode)
            ans = f"{value:.1f} mm^2" if unit == "mm^2" else f"{value:.1f} mm"
            q = _fill(tpl.text, _fields(s, box, tpl.lang))
            q, rewritten = _maybe_rewrite(model_fn, q, s, "G4", tpl.lang)
            # 改写不能改变所问的量纲；是/否判定仍以模板为准（G4 从不是二元题）。
            seq += 1
            item = _make(
                s,
                qid=f"{s.id}#{tpl.tid}",
                question=q,
                gen="G4",
                tpl=tpl,
                is_negative=False,
                yes_no=False,
                boxes=list(s.boxes),
                answer=ans,
                extra_meta={
                    "need_crop": True,
                    "measure_mode": tpl.mode,
                    "derived_mm": round(value, 4),
                    "measure_unit": unit,
                    "pixel_spacing": [spacing[0], spacing[1]],
                    "measure_box": box.to_json(),
                },
            )
            item.meta["rewritten"] = rewritten
            out.append(item)
    return out


def g5_compare(samples: List[Sample], cfg: Any, model_fn: Optional[ModelFn] = None) -> List[Sample]:
    """G5 对比：全图上下文 + 局部裁剪，附对侧镜像框。"""
    out: List[Sample] = []
    seq = 0
    for s in samples:
        box = _first_box(s)
        if box is None:
            continue
        size = _image_size(s)
        mirror = _mirror(box, size)
        extra = {
            "need_crop": True,
            "anchor_box": box.to_json(),
            "compare_box": mirror.to_json() if mirror is not None else None,
            "needs_global_context": True,
        }
        for tpl in _iter_templates("G5", sample=s):
            q = _fill(tpl.text, _fields(s, box, tpl.lang))
            q, rewritten = _maybe_rewrite(model_fn, q, s, "G5", tpl.lang)
            yes_no = _looks_yes_no(q) if rewritten else tpl.yes_no
            seq += 1
            item = _make(
                s,
                qid=f"{s.id}#{tpl.tid}",
                question=q,
                gen="G5",
                tpl=tpl,
                is_negative=False,
                yes_no=yes_no,
                boxes=[box] + ([mirror] if mirror is not None else []),
                extra_meta=extra,
            )
            item.meta["rewritten"] = rewritten
            out.append(item)
    return out


_GEN_FNS: Dict[str, Callable[[List[Sample], Any, Optional[ModelFn]], List[Sample]]] = {
    "G1": g1_box_anchored,
    "G2": g2_multi_scale,
    "G3": g3_negative,
    "G4": g4_measure,
    "G5": g5_compare,
}


# ---------------------------------------------------------------- 候选池与配额

def _drop_identity(lst: List[Any], obj: Any) -> None:
    """按对象身份删除（Sample 是 dataclass，默认 __eq__ 按值比较，不能直接用 remove）。"""
    for i, x in enumerate(lst):
        if x is obj:
            del lst[i]
            return


class _Pool:
    """一个生成器的候选池：前缀是已选中的，cursor 之后是尚未使用的。

    items 由 `_order_pool` 保证"非 yes/no 在前"，从而默认产出尽量少的二元题。
    """

    def __init__(self, name: str, items: List[Sample]) -> None:
        self.name = name
        self.items = items
        self.cursor = 0
        self.picked: List[Sample] = []
        self.used: set = set()

    def remaining(self) -> int:
        return max(0, len(self.items) - self.cursor)

    def take(self, n: int) -> int:
        took = 0
        while took < n and self.cursor < len(self.items):
            s = self.items[self.cursor]
            self.cursor += 1
            if id(s) in self.used:
                continue
            self.used.add(id(s))
            self.picked.append(s)
            took += 1
        return took

    def promote_yesno(self, k: int) -> int:
        """把尚未使用的 yes/no 候选换进来替换已选的非 yes/no，最多 k 条。"""
        yn_left = [s for s in self.items[self.cursor:] if s.meta.get("yes_no") and id(s) not in self.used]
        non_yn = [s for s in self.picked if not s.meta.get("yes_no")]
        k = min(k, len(yn_left), len(non_yn))
        for i in range(k):
            victim, repl = non_yn[i], yn_left[i]
            _drop_identity(self.picked, victim)
            self.used.add(id(repl))
            self.picked.append(repl)
        return k

    def swap_out_yesno(self, k: int) -> int:
        """换出超额的 yes/no；没有非 yes/no 备胎时直接丢弃（总量会略减）。"""
        spare = [s for s in self.items[self.cursor:] if not s.meta.get("yes_no") and id(s) not in self.used]
        done = 0
        for s in [x for x in reversed(self.picked) if x.meta.get("yes_no")]:
            if done >= k:
                break
            _drop_identity(self.picked, s)
            if spare:
                repl = spare.pop()
                self.used.add(id(repl))
                self.picked.append(repl)
            done += 1
        return done


def _order_pool(items: List[Sample], salt: str, seed: int) -> List[Sample]:
    """先确定性洗牌，再把非 yes/no 排到前面（保持组内顺序）。"""
    rng = _rng(seed, salt)
    shuffled = list(items)
    rng.shuffle(shuffled)
    return [s for s in shuffled if not s.meta.get("yes_no")] + [s for s in shuffled if s.meta.get("yes_no")]


def _allocate(weights: Dict[str, float], budget: int) -> Dict[str, int]:
    """按权重做最大余数分配；预算够时先给每个活跃生成器保底 1 条。"""
    out: Dict[str, int] = {g: 0 for g in weights}
    active = {g: w for g, w in weights.items() if w > 0}
    if budget <= 0 or not active:
        return out
    if budget >= len(active):
        for g in active:
            out[g] += 1
        budget -= len(active)
    total_w = sum(active.values()) or float(len(active))
    raw = {g: budget * active[g] / total_w for g in active}
    for g in active:
        out[g] += int(math.floor(raw[g]))
    rest = budget - sum(int(math.floor(raw[g])) for g in active)
    order = sorted(active, key=lambda g: (-(raw[g] - math.floor(raw[g])), g))
    for g in order[:max(0, rest)]:
        out[g] += 1
    return out


def _fill_quotas(pools: Dict[str, _Pool], weights: Dict[str, float], budget: int) -> int:
    """按 mix 配额取候选；某类产能不足时按权重向其他类借，最后裁到 budget。"""
    if budget <= 0 or not pools:
        return 0
    active_w = {g: max(weights.get(g, 0.0), 0.0) for g in pools}
    if sum(active_w.values()) <= 0:
        active_w = {g: 1.0 for g in pools}
    quotas = _allocate(active_w, budget)
    for g, q in quotas.items():
        pools[g].take(q)

    deficit = budget - sum(len(p.picked) for p in pools.values())
    guard = 0
    while deficit > 0 and guard < budget + 8:
        guard += 1
        avail = [(g, p) for g, p in pools.items() if p.remaining() > 0]
        if not avail:
            break
        # 优先给"权重高且已选得少"的池子补一条，保证小配额下各类都不缺席
        g, p = max(
            avail,
            key=lambda gp: (active_w.get(gp[0], 0.0) * gp[1].remaining() / (len(gp[1].picked) + 1), gp[0]),
        )
        if p.take(1) == 0:
            break
        deficit -= 1

    total = sum(len(p.picked) for p in pools.values())
    excess = total - budget
    while excess > 0:
        g = max(pools, key=lambda k: len(pools[k].picked))
        if not pools[g].picked:
            break
        pools[g].picked.pop()
        excess -= 1
    return sum(len(p.picked) for p in pools.values())


def _yesno_budget(n: int, yn_max: float) -> int:
    """yes/no 的条数上限：保证条数/n **严格小于** yn_max。

    直接用 floor(yn_max*n) 会在 40 条时给出 2 条（2/40 == 5%，并不小于上限），
    所以取 ceil(yn_max*n)-1，再夹到 [0, n]。
    """
    if n <= 0:
        return 0
    b = int(math.ceil(yn_max * n - 1e-9)) - 1
    return max(0, min(int(n), b))


def _yesno_count(pools: Sequence[_Pool]) -> int:
    return sum(1 for p in pools for s in p.picked if s.meta.get("yes_no"))


def _promote_yesno(pools: Sequence[_Pool], budget: int) -> int:
    cur = _yesno_count(pools)
    for p in pools:
        if cur >= budget:
            break
        cur += p.promote_yesno(budget - cur)
    return cur


def _cap_yesno(pools: Sequence[_Pool], budget: int) -> int:
    cur = _yesno_count(pools)
    for p in pools:
        if cur <= budget:
            break
        cur -= p.swap_out_yesno(cur - budget)
    return cur


# ---------------------------------------------------------------- 统计

def question_stats(items: Sequence[Sample]) -> Dict[str, Any]:
    """产出池的配比统计（复盘 / 断言都用这一份口径）。"""
    n = len(items)
    by_gen: Counter = Counter()
    neg = 0
    yn = 0
    rewritten = 0
    lang: Counter = Counter()
    for s in items:
        g = s.meta.get("generator", "")
        by_gen[g] += 1
        if s.meta.get("is_negative"):
            neg += 1
        if s.meta.get("yes_no"):
            yn += 1
        if s.meta.get("rewritten"):
            rewritten += 1
        lang[s.meta.get("template_lang", "?")] += 1
    return {
        "n": n,
        "n_positive": n - neg,
        "n_negative": neg,
        "negative_ratio": (neg / n) if n else 0.0,
        "n_yes_no": yn,
        "yesno_ratio": (yn / n) if n else 0.0,
        "n_rewritten": rewritten,
        "by_generator": dict(sorted(by_gen.items())),
        "by_lang": dict(sorted(lang.items())),
    }


# ---------------------------------------------------------------- 主入口 generate

def generate(samples: List[Sample], cfg: Any, model_fn: Optional[ModelFn] = None) -> List[Sample]:
    """五类生成器 + 配比控制，产出待过滤的候选问题池。

    配比规则（全部确定性）：
      1. 总候选量 T = min(qgen.per_sample × 样本数, qgen.target_prompts)；
      2. 负样本目标 = round(target_negative_ratio × T)，其余按 qgen.mix 分给 G1/G2/G4/G5；
      3. 某类产能不足（如无 spacing 时 G4 为空）按权重实时借调，不影响比例；
      4. 负样本实际数量按 round(pos × r/(1-r)) 反推，保证 60:40；
      5. yes/no 候选默认排在池尾，只按 floor(yes_no_max_ratio × total) 的预算补进。

    结束时用 `question_stats` 统计并断言两条配比红线。
    """
    bases = [s for s in samples if isinstance(s, Sample)]
    if not bases:
        return []

    seed = _seed(cfg)
    r = float(_cget(cfg, "qgen.target_negative_ratio", 0.40))
    yn_max = float(_cget(cfg, "qgen.yes_no_max_ratio", 0.05))
    total_target = _target_total(len(bases), cfg)
    enabled = _enabled_generators(cfg)
    weights = _mix_weights(cfg)

    # 1) 逐生成器产出候选，并按 seed 洗牌 / 排序
    pools: Dict[str, _Pool] = {}
    for gen in enabled:
        raw = _GEN_FNS[gen](bases, cfg, model_fn)
        pools[gen] = _Pool(gen, _order_pool(raw, f"pool|{gen}", seed))

    pos_pools = {g: pools[g] for g in POSITIVE_GENS if g in pools}
    neg_pool = pools.get("G3")

    # 2) 正负预算
    neg_target = int(round(r * total_target)) if neg_pool is not None else 0
    pos_budget = max(0, total_target - neg_target)

    pos_weights = {g: max(weights.get(g, 0.0), 0.0) for g in pos_pools}
    n_pos = _fill_quotas(pos_pools, pos_weights, pos_budget)

    n_neg = 0
    need_neg = 0
    neg_capacity = 0
    if neg_pool is not None and n_pos > 0:
        need_neg = int(round(n_pos * r / max(1e-9, 1.0 - r)))
        neg_capacity = neg_pool.remaining()
        n_neg = neg_pool.take(min(need_neg, neg_capacity))

    # 3) yes/no 预算：默认已排在池尾，这里只在预算内主动补几条，超额再换出
    all_pools = list(pos_pools.values()) + ([neg_pool] if neg_pool is not None else [])
    yn_budget = _yesno_budget(n_pos + n_neg, yn_max)
    _promote_yesno(all_pools, yn_budget)
    _cap_yesno(all_pools, yn_budget)

    # 4) 合并并做最后一次确定性洗牌
    final: List[Sample] = []
    for p in all_pools:
        final.extend(p.picked)
    _rng(seed, "final").shuffle(final)

    # 5) 统计 + 断言
    st = question_stats(final)
    if st["n"]:
        assert st["yesno_ratio"] <= yn_max + 1e-9, (
            f"yes/no 占比 {st['yesno_ratio']:.4f} 超过上限 {yn_max}"
        )
        if n_pos > 0 and need_neg > 0 and neg_capacity >= need_neg:
            tol = 1.5 / max(1, st["n"]) + 1e-9
            assert abs(st["negative_ratio"] - r) <= tol, (
                f"负样本占比 {st['negative_ratio']:.4f} 偏离目标 {r} 超过容差 {tol:.4f}"
            )
    return final


# ---------------------------------------------------------------- 三道过滤闸

@dataclass
class FilterAudit:
    """三道闸的路由结果。kept 才是下游可用的；其余桶用于复盘与消融。"""

    kept: List[Sample] = field(default_factory=list)
    arbitration: List[Sample] = field(default_factory=list)     # 双模型不一致 -> 人工/仲裁
    to_negative_pool: List[Sample] = field(default_factory=list)  # 无工具全对 -> 转 G3 负样本池
    too_hard: List[Sample] = field(default_factory=list)        # 给 gold 框仍全错 -> 太难
    ratio_dropped: List[Sample] = field(default_factory=list)   # 比例控制下采样掉的
    stats: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kept": [s.id for s in self.kept],
            "arbitration": [s.id for s in self.arbitration],
            "to_negative_pool": [s.id for s in self.to_negative_pool],
            "too_hard": [s.id for s in self.too_hard],
            "ratio_dropped": [s.id for s in self.ratio_dropped],
            "stats": self.stats,
        }


def _copy_for_filter(s: Sample) -> Sample:
    """过滤会往 meta 里写 verdict，先造一份浅拷贝，别改调用方的对象。"""
    return Sample(
        id=s.id,
        image=s.image,
        question=s.question,
        answer=s.answer,
        dataset=s.dataset,
        tier=s.tier,
        answer_type=s.answer_type,
        boxes=list(s.boxes),
        masks=list(s.masks),
        license=s.license,
        commercial_ok=s.commercial_ok,
        meta=_clone_meta(s),
    )


def _agree_prompt(s: Sample) -> str:
    return f"请回答以下医学影像问题，只输出答案本身。\n问题：{s.question}"


def _gate_agree(
    s: Sample,
    cfg: Any,
    base_model_fn: Optional[ModelFn],
    agree_fn: Optional[AgreeFn],
) -> Tuple[str, str]:
    """闸 1：双模型一致性。返回 (状态, 备注)。

    状态：skipped | agree | disagree
      - agree_fn 未注入 或 filters.dual_model_agree=false -> skipped；
      - 任一侧答案为空（比如教师答案尚未回填）-> skipped（记 missing_answer）；
      - 只在两侧都有答案且 agree_fn 判 true 时保留。
    """
    if agree_fn is None or not bool(_cget(cfg, "qgen.filters.dual_model_agree", True)):
        return "skipped", "no_agree_fn"
    a = (s.answer or "").strip()
    b = ""
    if base_model_fn is not None:
        try:
            b = _clean_model_output(base_model_fn(_agree_prompt(s), [s.image]))
        except Exception:
            b = ""
    if not a or not b:
        return "skipped", "missing_answer"
    return ("agree" if bool(agree_fn(a, b)) else "disagree"), ""


def _gate_necessity(
    s: Sample,
    cfg: Any,
    pass8_fn: Optional[PassFn],
    with_box_fn: Optional[PassFn],
) -> Tuple[str, str]:
    """闸 2：工具必要性。

    状态：not_applicable（本身就是负样本）| skipped | too_easy | too_hard | hard_ok
      - pass8_fn(sample) == True（无工具 pass@8 全对）-> too_easy，转负样本池；
      - pass8 非全对，但 with_box_fn 判"给 gold 框仍全错"-> too_hard，删除/降权；
      - 其余保留为正样本。
    """
    if bool(s.meta.get("is_negative")):
        return "not_applicable", ""
    need = int(_cget(cfg, "qgen.filters.tool_necessity_pass8", 0) or 0)
    if pass8_fn is None or need <= 0:
        return "skipped", "no_pass8_fn"
    try:
        easy = bool(pass8_fn(s))
    except Exception:
        return "skipped", "pass8_error"
    if easy:
        return "too_easy", ""
    if with_box_fn is not None:
        try:
            solvable = bool(with_box_fn(s))
        except Exception:
            solvable = True  # 注入函数自己出错时不要误杀样本
        if not solvable:
            return "too_hard", ""
    return "hard_ok", ""


def _enforce_ratio(
    items: List[Sample],
    r: float,
    yn_max: float,
    seed: int,
    max_iter: int = 8,
) -> Tuple[List[Sample], List[Sample]]:
    """闸 3：正负 60:40 + yes/no < yn_max，超了就按 seed 下采样。

    返回 (kept, dropped)。删除一律按对象身份，避免 dataclass 按值比较误删。
    """
    kept: List[Sample] = list(items)
    dropped: List[Sample] = []
    rng = _rng(seed, "filter|ratio")

    def drop(victims: List[Sample]) -> None:
        nonlocal kept
        ids = {id(v) for v in victims}
        for v in victims:
            dropped.append(v)
        kept = [s for s in kept if id(s) not in ids]

    for _ in range(max_iter):
        n = len(kept)
        if n == 0:
            break
        yn_budget = _yesno_budget(n, yn_max)
        yn = [s for s in kept if s.meta.get("yes_no")]
        pos = [s for s in kept if not s.meta.get("is_negative")]
        neg = [s for s in kept if s.meta.get("is_negative")]

        # (a) 先砍超额的 yes/no，优先砍"会导致比例更偏"的那一类
        if len(yn) > yn_budget:
            excess = len(yn) - yn_budget
            want_neg = int(round(r * n))
            yn_neg = [s for s in yn if s.meta.get("is_negative")]
            yn_pos = [s for s in yn if not s.meta.get("is_negative")]
            order = [yn_neg, yn_pos] if len(neg) > want_neg else [yn_pos, yn_neg]
            for group in order:
                if excess <= 0:
                    break
                group = list(group)
                rng.shuffle(group)
                take = min(excess, len(group))
                drop(group[:take])
                excess -= take
            continue

        # (b) 再修正负比例。已经是纯正/纯负时无法再修正，
        #     继续删只会把池子清空（小样本快照下常见），直接收手。
        if not pos or not neg:
            break
        want_neg = int(round(r * n))
        if len(neg) > want_neg:
            victims = list(neg)
            rng.shuffle(victims)
            drop(victims[: len(neg) - want_neg])
            continue
        if len(neg) < want_neg:
            want_pos = int(round(len(neg) * (1.0 - r) / max(1e-9, r)))
            if len(pos) > want_pos:
                victims = list(pos)
                rng.shuffle(victims)
                drop(victims[: len(pos) - want_pos])
                continue
        break  # 两条红线都满足
    return kept, dropped


def filter_questions_with_audit(
    samples: List[Sample],
    cfg: Any,
    base_model_fn: Optional[ModelFn] = None,
    agree_fn: Optional[AgreeFn] = None,
    pass8_fn: Optional[PassFn] = None,
    with_box_fn: Optional[PassFn] = None,
) -> FilterAudit:
    """PROPOSAL §4.2 的三道闸，带完整分桶（复盘用）。

    闸 1 dual_model_agree：注入 agree_fn(a, b) 判两个模型的答案是否一致；
                          未注入或任一答案为空则 skipped（meta.filter.agree='skipped'）。
    闸 2 tool_necessity：注入 pass8_fn(sample) 表示"无工具 pass@8 全对"；
                          为 True 的样本不删除，而是标 is_negative 转入负样本池。
                          可选 with_box_fn 再判"给 gold 框仍全错"-> too_hard。
    闸 3 比例控制：正负 60:40、yes/no < yes_no_max_ratio，超了按固定 seed 下采样。
    """
    seed = _seed(cfg)
    r = float(_cget(cfg, "qgen.target_negative_ratio", 0.40))
    yn_max = float(_cget(cfg, "qgen.yes_no_max_ratio", 0.05))

    audit = FilterAudit()
    survivors: List[Sample] = []

    for raw in samples:
        if not isinstance(raw, Sample):
            continue
        s = _copy_for_filter(raw)
        f = s.meta.setdefault("filter", {})

        # 闸 1
        state, note = _gate_agree(s, cfg, base_model_fn, agree_fn)
        f["agree"] = state
        if note:
            f["agree_note"] = note
        if state == "disagree":
            f["route"] = "arbitration"
            audit.arbitration.append(s)
            continue

        # 闸 2
        nstate, nnote = _gate_necessity(s, cfg, pass8_fn, with_box_fn)
        f["necessity"] = nstate
        if nnote:
            f["necessity_note"] = nnote
        if nstate == "too_easy":
            # 无工具就能全对 -> 不该教它调工具，转负样本池（§4.2 第 2 条）
            s.meta["is_negative"] = True
            s.meta["need_crop"] = False
            s.meta["tool_free"] = True
            s.boxes = []
            s.meta["routed_from"] = s.meta.get("generator", "")
            f["route"] = "to_negative_pool"
            audit.to_negative_pool.append(s)
            continue
        if nstate == "too_hard":
            f["route"] = "too_hard"
            audit.too_hard.append(s)
            continue

        f["route"] = "kept"
        survivors.append(s)

    # 闸 3
    kept, dropped = _enforce_ratio(survivors, r, yn_max, seed)
    for s in dropped:
        s.meta.setdefault("filter", {})["route"] = "ratio_dropped"
    audit.kept = kept
    audit.ratio_dropped = dropped
    audit.stats = question_stats(kept)

    if audit.stats["n"]:
        assert audit.stats["yesno_ratio"] <= yn_max + 1e-9, (
            f"yes/no 占比 {audit.stats['yesno_ratio']:.4f} 超过上限 {yn_max}"
        )
    return audit


def filter_questions(
    samples: List[Sample],
    cfg: Any,
    base_model_fn: Optional[ModelFn] = None,
    agree_fn: Optional[AgreeFn] = None,
    pass8_fn: Optional[PassFn] = None,
    with_box_fn: Optional[PassFn] = None,
) -> List[Sample]:
    """三道闸的正式入口；只返回通过全部闸门的样本（其余见 filter_questions_with_audit）。"""
    return filter_questions_with_audit(
        samples,
        cfg,
        base_model_fn=base_model_fn,
        agree_fn=agree_fn,
        pass8_fn=pass8_fn,
        with_box_fn=with_box_fn,
    ).kept
