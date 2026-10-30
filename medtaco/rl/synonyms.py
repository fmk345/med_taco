"""医学影像答案的内置同义词表与词法常量（grader 的词典层）。

设计要点：
  - `DEFAULT_SYNONYMS`：canonical 词 -> 变体列表（中英文混排）。判分时把所有变体
    都映射到 canonical，从而 `effusion` ≈ `pleural effusion`、`胸腔积液` ≈ `pleural effusion`。
  - 规范化在 `medtaco.rl.grader.normalize` 里做，本模块只做"词表"，不依赖 grader，
    避免循环 import（`build_alias_map` 接受一个 normalizer 回调）。
  - 外部 json 可以覆盖/扩充词表：json 形状与 `DEFAULT_SYNONYMS` 相同，
    `{"canonical": ["变体1", "变体2", ...]}`。同 key 时**取并集**（外部在前），
    这样外部既能新增词条，也能给已有词条追加本单位/本院的说法。
  - 本模块无随机性、无第三方依赖。
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

# ------------------------------------------------------------------ 同义词表
#
# 说明：`nodule` 与 `mass` **刻意不合并**——临床上二者以尺寸（>3cm 为 mass）区分，
# 合并会把"把结节说成肿块"判成对，属于判分器系统性偏高。词表只合并"同一发现的
# 不同措辞"，不合并"不同分级"。
DEFAULT_SYNONYMS: Dict[str, List[str]] = {
    # ---- 常见胸部影像发现
    "pleural effusion": [
        "effusion", "effusions", "pleural effusions", "pleural fluid",
        "胸腔积液", "胸膜腔积液", "胸水", "双侧胸腔积液", "少量胸腔积液",
    ],
    "opacity": [
        "opacification", "opacities", "focal opacity", "increased opacity",
        "hazy opacity", "increased density", "密度增高影", "片状影",
        "斑片影", "模糊影", "磨玻璃影", "ground glass opacity", "groundglass",
    ],
    "consolidation": [
        "consolidative", "consolidative change", "consolidations",
        "实变", "实变影", "肺实变",
    ],
    "nodule": [
        "nodules", "pulmonary nodule", "lung nodule", "pulmonary nodules",
        "结节", "肺结节", "小结节", "结节影",
    ],
    "mass": [
        "masses", "pulmonary mass", "soft tissue mass", "肿块", "肿块影",
        "占位", "占位性病变", "团块影",
    ],
    "cardiomegaly": [
        "enlarged heart", "cardiac enlargement", "heart enlargement",
        "enlarged cardiac silhouette", "心影增大", "心脏增大", "心影变大",
        "心影饱满",
    ],
    "atelectasis": [
        "atelectases", "atelectatic", "lung collapse", "pulmonary collapse",
        "肺不张", "肺叶不张", "肺组织塌陷",
    ],
    "pneumothorax": [
        "pneumothoraces", "ptx", "气胸", "少量气胸", "张力性气胸",
    ],
    "fracture": [
        "fractures", "fx", "bone fracture", "骨折", "骨裂", "骨质断裂",
    ],
    "pulmonary edema": [
        "edema", "oedema", "pulmonary oedema", "interstitial edema",
        "肺水肿", "间质性肺水肿", "肺淤血",
    ],
    "infiltrate": [
        "infiltration", "infiltrates", "infiltrative change",
        "浸润", "浸润影", "渗出影",
    ],
    "pneumonia": [
        "pneumonic", "pneumonitis", "肺炎", "肺部感染", "感染性病变",
    ],
    "pleural thickening": [
        "胸膜增厚", "胸膜肥厚", "pleural thickening",
    ],
    "fibrosis": [
        "fibrotic change", "fibrotic", "fibrotic band", "纤维化", "纤维灶",
        "条索影", "纤维条索",
    ],
    "calcification": [
        "calcified", "calcifications", "calcified lesion", "钙化", "钙化灶",
        "钙化影",
    ],
    "pulmonary embolism": [
        "embolism", "肺栓塞", "肺动脉栓塞",
    ],
    "lesion": [
        "lesions", "abnormality", "abnormalities", "病灶", "病变", "异常",
    ],
    "lymphadenopathy": [
        "adenopathy", "lymph node enlargement", "enlarged lymph nodes",
        "淋巴结肿大", "淋巴结增大",
    ],
    "emphysema": [
        "emphysematous change", "肺气肿", "肺大泡", "肺大疱",
    ],
    "cavitation": [
        "cavity", "cavities", "cavitary lesion", "空洞", "空洞形成", "空腔",
    ],
    # ---- 正常/阴性表述（与否定极性配合使用）
    "normal": [
        "unremarkable", "unremarkable study", "normal study", "no abnormality",
        "no abnormalities", "no acute abnormality", "negative study",
        "正常", "未见异常", "未见明显异常", "无异常", "未见明显病变",
    ],
    # ---- 其他部位
    "hepatomegaly": ["enlarged liver", "肝大", "肝脏增大", "肝影增大"],
    "splenomegaly": ["enlarged spleen", "脾大", "脾脏增大"],
    "gallstone": [
        "gallstones", "cholelithiasis", "胆结石", "胆囊结石", "胆石症",
    ],
    "hydronephrosis": ["肾积水", "肾盂积水"],
    "pleural plaque": ["胸膜斑", "胸膜斑块"],
}

# ---------------------------------------------------------------- 是非词表
# closed 题的 affirm/deny 等价类（normalize 之后再做单复数归一）。
AFFIRM_WORDS: List[str] = [
    "yes", "y", "true", "present", "positive", "affirmative", "confirmed",
    "是", "有", "存在", "阳性", "正确", "对", "可见", "显示", "能",
]
DENY_WORDS: List[str] = [
    "no", "n", "false", "absent", "negative", "none", "not present",
    "否", "不", "没有", "无", "阴性", "错误", "错", "不存在", "未见", "不能",
]

# ---------------------------------------------------------------- 否定词法
# 出现在答案里即视为"否定极性"。注意排序无关：build 时会按长度降序匹配，
# 保证 `no evidence of` 先于 `no`、`未见明显` 先于 `未见`、`未见` 先于 `未`。
NEGATION_PATTERNS: List[str] = [
    # 英文短语
    "no evidence of", "no evidence for", "no sign of", "no signs of",
    "without evidence of", "absence of", "free of", "not seen",
    "not detected", "not identified", "no definite", "no focal",
    "not present", "negative for", "negative",
    # 英文单词
    "no", "not", "without", "absent", "none", "never", "nil",
    # 中文短语（长的必须排在短的前面，见 build_negation_regex 的排序）
    # ⚠️ 刻意不放裸的「未」「不」：它们是构词成分而非否定标记，
    #    例如「肺不张 / 未分化」会被误判成阴性（错误地把 atelectasis 判成 no atelectasis）。
    "未见明显异常", "未见明显", "未见确切", "未见", "未发现", "未显示",
    "没有", "无", "阴性", "否认", "排除", "不存在", "不能",
]

# ---------------------------------------------------------------- 工具函数


def _has_cjk(s: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in s)


def load_synonyms(path: str) -> Dict[str, List[str]]:
    """从 json 文件读同义词表。文件不存在时抛 FileNotFoundError（由调用方决定是否兜底）。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"同义词 json 顶层必须是 dict: {path}")
    out: Dict[str, List[str]] = {}
    for k, v in raw.items():
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, (list, tuple)):
            raise ValueError(f"同义词条目 {k!r} 必须是 list[str]，实际 {type(v).__name__}")
        out[str(k)] = [str(x) for x in v]
    return out


def merge_synonyms(
    base: Dict[str, List[str]], extra: Optional[Dict[str, List[str]]] = None
) -> Dict[str, List[str]]:
    """合并两张同义词表：同 key 取并集（extra 变体在前），保持确定性顺序。"""
    merged: Dict[str, List[str]] = {k: list(v) for k, v in base.items()}
    for key, variants in (extra or {}).items():
        for v in variants:
            merged.setdefault(str(key), []).append(str(v))
        # 去重但保序：extra 在前 -> 外部写法优先命中
        seen: set = set()
        uniq: List[str] = []
        for v in merged[str(key)]:
            if v not in seen:
                seen.add(v)
                uniq.append(v)
        merged[str(key)] = uniq
    return merged


def build_alias_map(
    table: Dict[str, List[str]],
    normalizer,
) -> Tuple[Dict[str, str], Optional[re.Pattern]]:
    """把同义词表编译成 (变体归一化后 -> canonical 归一化后, 替换用正则)。

    正则按变体长度降序排列，保证优先匹配更长的短语（如 `pleural effusion`
    先于 `effusion`）。含中文的变体不加词边界（中文没有空格分词），
    纯拉丁变体加 `(?<![a-z0-9]) ... (?![a-z0-9])` 边界。
    """
    mapping: Dict[str, str] = {}
    for canonical, variants in table.items():
        canon_norm = normalizer(str(canonical))
        if not canon_norm:
            continue
        mapping.setdefault(canon_norm, canon_norm)
        for v in variants:
            vn = normalizer(str(v))
            if vn:
                # 先到先得：外部（已排在前面）优先
                mapping.setdefault(vn, canon_norm)

    # 只对"确实需要改写"的变体建正则，避免无谓开销
    needed: List[Tuple[str, str]] = [
        (vn, canon) for vn, canon in mapping.items() if vn != canon
    ]
    if not needed:
        return mapping, None
    needed.sort(key=lambda kv: (-len(kv[0]), kv[0]))
    branches: List[str] = []
    for vn, _ in needed:
        esc = re.escape(vn)
        if _has_cjk(vn):
            branches.append(esc)
        else:
            branches.append(r"(?<![a-z0-9])" + esc + r"(?![a-z0-9])")
    pattern = re.compile("|".join(branches))
    return mapping, pattern


def build_negation_regex(patterns: Optional[List[str]] = None) -> re.Pattern:
    """按长度降序拼否定词正则；拉丁词加词边界，中文词裸匹配。"""
    pats = sorted(set(patterns or NEGATION_PATTERNS), key=lambda p: (-len(p), p))
    branches: List[str] = []
    for p in pats:
        esc = re.escape(p)
        if _has_cjk(p):
            branches.append(esc)
        else:
            branches.append(r"(?<![a-z])" + esc + r"(?![a-z])")
    return re.compile("|".join(branches))


__all__ = [
    "DEFAULT_SYNONYMS",
    "AFFIRM_WORDS",
    "DENY_WORDS",
    "NEGATION_PATTERNS",
    "load_synonyms",
    "merge_synonyms",
    "build_alias_map",
    "build_negation_regex",
]
