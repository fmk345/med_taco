"""确定性医学答案判分器（PROPOSAL §7.2.1）。

这是 RL 阶段线上唯一的 $R_{\\text{ORM}}$ 实现：纯规则、零成本、**同输入同输出**。
任何涉及模型采样的判分都不允许进入训练循环（判分噪声会直接污染 GRPO 的 advantage）。

判分口径（对应 PROPOSAL §7.2 表格 1–4 行）：
  - closed ：归一化后精确匹配；yes/no 与 affirm/deny 等价类互换；否则 0。
  - short  ：归一化（小写/去冠词标点/单复数/单位统一 mm/全角转半角）+ 同义词归一
             + 双向 token 包含；**否定极性必须一致**（`no effusion` ≠ `effusion`）。
  - measure：数字 + 单位 -> mm，相对误差 ≤10% 判 1；≤25% 判 0（meta 标 half_credit）；
             >25% 判 0（不做负分，避免"差得越远扣得越狠"制造梯度尖峰）。
  - long   ：`gradable()` 返回 False，不进 RL；`score_offline_judge()` 显式抛
             NotImplementedError，防止有人误把 judge 接进循环。

所有方法不读文件、不联网、不用随机数。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from medtaco.rl.synonyms import (
    AFFIRM_WORDS,
    DENY_WORDS,
    DEFAULT_SYNONYMS,
    NEGATION_PATTERNS,
    build_alias_map,
    build_negation_regex,
    load_synonyms,
    merge_synonyms,
)
from medtaco.utils.schema import Sample

# ------------------------------------------------------------------ 常量

GRADER_VERSION = 1

ARTICLES = {"a", "an", "the"}

# 语序无关比较时忽略的虚词/方位连接词（本身不承载发现信息）
_FUNCTION_WORDS = {
    "a", "an", "the", "of", "in", "at", "on", "to", "with", "and", "or",
    "is", "are", "was", "were", "be", "there", "seen", "shows", "show",
    "showing", "noted", "identified", "present", "consistent", "suggestive",
}

# 数字 + 单位：mm 必须排在 m 之前，否则 "5mm" 会被 "m" 吃掉后面那个 m
_UNIT_RE = re.compile(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s*(mm|cm|m)(?![a-z])")
_UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0}

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_ANSWER_OPEN_RE = re.compile(r"<answer>", re.IGNORECASE)
# 只匹配类标记的尖括号标签，避免把 "p < 20 mm" 这类比较符也吃掉
_TAG_RE = re.compile(r"</?[a-zA-Z_][\w-]*>")

_CJK_RE = re.compile(r"[一-鿿]")
_DOT_NOT_DECIMAL_RE = re.compile(r"(?<!\d)\.|\.(?!\d)")
_NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")

# 视为"没给答案"的文本（判 -1）。注意 "none" 不在其中：它是真答案（否定），
# 应与金标做极性比较得到 0，而不是当作解析失败。
_EMPTY_FORMS = {"", "-", "na", "n/a", "unknown", "not given", "notgiven", "?"}


# ------------------------------------------------------------------ 归一化


def _singular(word: str) -> str:
    """极简单复数归一（够用即可；不规则词靠同义词表兜底）。"""
    if not word.isascii() or not word.isalpha() or len(word) <= 3:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("sses", "shes", "ches", "xes")):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("us"):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def _is_blank_answer(text: str) -> bool:
    """判断答案是否等于"没答"。在 normalize **之前**判，因为 normalize 会把
    `n/a` 变成 `n a`、把单个选项字母 `A` 当冠词吃掉。"""
    t = unicodedata.normalize("NFKC", str(text or "")).strip().lower()
    t = _WS_RE.sub(" ", t).strip(" .。`'\"")
    return t in _EMPTY_FORMS


def _norm_units(text: str) -> str:
    """把 `1.5 cm` / `15mm` / `0.02 m` 统一成 `XX mm`。"""
    def _rep(m: "re.Match[str]") -> str:
        val = float(m.group(1)) * _UNIT_TO_MM[m.group(2)]
        # 去掉浮点毛刺：15.0 -> 15，1.55 -> 1.55
        s = f"{val:.6f}".rstrip("0").rstrip(".")
        return f" {s} mm "

    return _UNIT_RE.sub(_rep, text)


def normalize(s: str) -> str:
    """答案文本归一化：全角转半角 -> 小写 -> 单位统一 mm -> 去标点/冠词
    -> 单复数归一 -> 空白归一。

    注意：本函数**不做**同义词改写，也**不做**否定剥离；那两步在判分器内部完成，
    以保证 `normalize` 是一个无状态、可独立单测的纯函数。
    """
    if not s:
        return ""
    t = unicodedata.normalize("NFKC", str(s)).lower()
    t = _norm_units(t)
    # 下划线属于 \w，单独换成空格
    t = t.replace("_", " ")
    t = _NON_WORD_RE.sub(" ", t)
    t = _DOT_NOT_DECIMAL_RE.sub(" ", t)
    toks = t.split()
    out: List[str] = []
    for tok in toks:
        # 只有一个 token 时不吃冠词：闭集选项 "A"/"B"/"C"/"D" 不能被当成冠词 a
        if tok in ARTICLES and len(toks) > 1:
            continue
        out.append(_singular(tok))
    return _WS_RE.sub(" ", " ".join(out)).strip()


# ------------------------------------------------------------------ 小工具

# 是非题的等价类：在 normalize 定义之后算一次，避免每次打分重算
_AFFIRM_NORM = frozenset(normalize(w) for w in AFFIRM_WORDS)
_DENY_NORM = frozenset(normalize(w) for w in DENY_WORDS)
# "正常/未见异常" 视作否定极（对 "is there X?" 的回答等价于 no）
_NORMAL_CANON = frozenset({"normal"})


def _cfg_get(cfg: Any, path: str, default: Any) -> Any:
    """同时兼容 Cfg（dict 子类，支持点号路径）与普通 dict / None。"""
    if cfg is None:
        return default
    cur: Any = cfg
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return default if cur is None else cur


def _is_cjk(s: str) -> bool:
    return bool(_CJK_RE.search(s))


def _token_contains(a: str, b: str) -> bool:
    """判断 a 是否作为"整体 token 序列"出现在 b 中（中文退化为子串匹配）。"""
    if not a:
        return True
    if _is_cjk(a) or _is_cjk(b):
        return a in b
    ta, tb = a.split(), b.split()
    n = len(ta)
    if n > len(tb):
        return False
    return any(tb[i:i + n] == ta for i in range(len(tb) - n + 1))


def _token_subset(a: str, b: str) -> bool:
    """语序无关的 token 包含：a 的实词集合是 b 的子集即算包含。

    用于 `consolidation of the left lower lobe` ≈ `left lower lobe consolidation`
    这类纯语序差异；虚词不参与比较。反向不成立（那由 `_token_contains` 管）。
    """
    if not a or not b:
        return False
    if _is_cjk(a) or _is_cjk(b):
        return False
    sa = {t for t in a.split() if t not in _FUNCTION_WORDS}
    sb = {t for t in b.split() if t not in _FUNCTION_WORDS}
    # 单实词的情况交给有序包含，这里只管多词短语的语序差异
    return len(sa) >= 2 and sa <= sb


def _parse_measures(text: str) -> List[float]:
    """从自由文本里抽出所有带单位的量，统一换算成 mm。无单位数字不计入。"""
    if not text:
        return []
    t = unicodedata.normalize("NFKC", str(text)).lower()
    vals: List[float] = []
    for m in _UNIT_RE.finditer(t):
        vals.append(float(m.group(1)) * _UNIT_TO_MM[m.group(2)])
    return vals


# ------------------------------------------------------------------ 判分器


class DeterministicGrader:
    """确定性判分器。一个实例可复用；`last_meta` 记录最近一次打分的细节。"""

    def __init__(
        self,
        cfg: Any = None,
        synonyms: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.cfg = cfg
        # 外部可传 json 路径；cfg 里也可配 rl.grader.synonyms_path
        extra: Optional[Dict[str, List[str]]] = None
        if isinstance(synonyms, str):
            if synonyms:
                extra = load_synonyms(synonyms)
        elif synonyms:
            extra = {str(k): list(v) for k, v in synonyms.items()}
        path = _cfg_get(cfg, "rl.grader.synonyms_path", None)
        if path:
            extra = merge_synonyms(load_synonyms(str(path)), extra or {})

        self.synonyms: Dict[str, List[str]] = merge_synonyms(DEFAULT_SYNONYMS, extra)

        self.full_tol: float = float(_cfg_get(cfg, "rl.grader.measure_full_tol", 0.10))
        self.half_tol: float = float(_cfg_get(cfg, "rl.grader.measure_half_tol", 0.25))

        # 编译同义词/否定词正则（一次性，纯函数，保证确定性）
        self._alias_map, self._alias_re = build_alias_map(self.synonyms, normalize)
        neg_patterns = [normalize(p) for p in _negation_raw()]
        self._neg_re = build_negation_regex([p for p in neg_patterns if p])

        # 最近一次打分的元信息（确定性：同一输入重复调用结果一致）
        self.last_meta: Dict[str, Any] = {}

    # -------------------------------------------------------------- 解析
    def parse(self, raw: str) -> Tuple[str, bool]:
        """从模型原始输出抽取最终答案。

        优先级：最后一个 `<answer>...</answer>` 块 > `<answer>` 之后到文本末尾
        > 最后一行非空文本。返回 (answer, ok)；ok=False 表示没抽到任何有效内容，
        调用方应记 0 分且该样本不进 Δ。
        """
        if raw is None:
            return "", False
        text = str(raw)
        if not text.strip():
            return "", False

        matches = _ANSWER_RE.findall(text)
        if matches:
            ans = matches[-1].strip()
            # 嵌套标签常见于模型复读，取内层纯文本
            ans = _TAG_RE.sub(" ", ans)
            ans = _WS_RE.sub(" ", ans).strip()
            return ans, bool(ans)

        opens = list(_ANSWER_OPEN_RE.finditer(text))
        if opens:
            tail = text[opens[-1].end():]
            tail = tail.split("<")[0]
            ans = _WS_RE.sub(" ", tail).strip().strip("`").strip()
            return ans, bool(ans)

        # 没有标签：取最后一行非空文本（顺带剥掉残缺的 answer 标签）
        for line in reversed(text.splitlines()):
            line = line.strip()
            if line:
                ans = _TAG_RE.sub(" ", line)
                ans = _WS_RE.sub(" ", ans).strip()
                return ans, bool(ans)
        return "", False

    # -------------------------------------------------------------- 归一化核心
    def canonicalize(self, s: str) -> str:
        """normalize + 同义词归一（把变体都改写成 canonical 形式）。"""
        n = normalize(s)
        if not n or self._alias_re is None:
            return n
        alias = self._alias_map
        return _WS_RE.sub(" ", self._alias_re.sub(lambda m: alias.get(m.group(0), m.group(0)), n)).strip()

    def analyze(self, s: str) -> Tuple[str, bool]:
        """返回 (去掉否定词后的规范化内容, 是否否定)。"""
        canon = self.canonicalize(s)
        if not canon:
            return "", False
        neg = False
        if self._neg_re is not None and self._neg_re.search(canon):
            neg = True
            canon = self._neg_re.sub(" ", canon)
        canon = _WS_RE.sub(" ", canon).strip()
        # 否定剥离后可能残留纯标点/空
        canon = canon.strip(" -")
        return canon, neg

    # -------------------------------------------------------------- 判分
    def score(self, sample: Sample, pred: str) -> int:
        """主入口，返回 ∈ {-1,0,1}。空答案返回 -1。"""
        return int(self.explain(sample, pred)["score"])

    def explain(self, sample: Sample, pred: str) -> Dict[str, Any]:
        """带理由的打分，结果同时写入 `self.last_meta`。"""
        meta: Dict[str, Any] = {"answer_type": getattr(sample, "answer_type", "")}
        raw_pred = "" if pred is None else str(pred)

        answer, ok = self.parse(raw_pred)
        if not ok:
            meta.update(score=-1, reason="empty_pred")
            self.last_meta = meta
            return meta
        if _is_blank_answer(answer):
            # "n/a" / "unknown" 之类没有任何可判内容，同空串处理
            meta.update(score=-1, reason="empty_pred", parsed=answer)
            self.last_meta = meta
            return meta
        meta["parsed"] = answer

        gold = getattr(sample, "answer", "") or ""
        if not str(gold).strip():
            meta.update(score=-1, reason="empty_gold")
            self.last_meta = meta
            return meta

        atype = str(getattr(sample, "answer_type", "short") or "short").lower()

        if atype == "closed":
            s = self._score_closed(gold, answer, meta)
        elif atype == "measure":
            s = self._score_measure(gold, answer, meta)
        elif atype == "short":
            s = self._score_short(gold, answer, meta)
        else:
            # long / 未知类型：不进 RL，记 0 并留痕（不抛，避免批量跑挂）
            meta.update(score=0, reason=f"not_gradable_type:{atype}")
            self.last_meta = meta
            return meta

        meta["score"] = int(s)
        self.last_meta = meta
        return meta

    # ---- closed ----
    def _score_closed(self, gold: str, pred: str, meta: Dict[str, Any]) -> int:
        g, p = normalize(gold), normalize(pred)
        meta.update(gold_norm=g, pred_norm=p)
        if g and p and g == p:
            meta["reason"] = "exact"
            return 1

        g_pol = self._closed_polarity(gold)
        p_pol = self._closed_polarity(pred)
        meta.update(gold_polarity=g_pol, pred_polarity=p_pol)
        if g_pol is not None and g_pol == p_pol:
            meta["reason"] = "polarity"
            return 1

        # 兜底：两侧原文都带同样的否定且去掉否定后同义（如 "no evidence of X" ≈ "negative"）
        g_c, g_neg = self.analyze(gold)
        p_c, p_neg = self.analyze(pred)
        if g_neg == p_neg and g_c == p_c and (g_neg or g_c):
            meta["reason"] = "negation_equivalent"
            return 1

        meta["reason"] = "mismatch"
        return 0

    def _closed_polarity(self, s: str) -> Optional[bool]:
        """把 closed 答案映射到布尔极性；不是是非题返回 None。"""
        n = normalize(s)
        if not n:
            return None
        # "normal / unremarkable / 未见异常" 在是非题里等价于否定（PROPOSAL §7.2 表）
        if self.canonicalize(s) in _NORMAL_CANON:
            return False
        aff = _AFFIRM_NORM
        den = _DENY_NORM
        if n in aff:
            return True
        if n in den:
            return False
        # 多词答案取第一个 token（"yes, mild" / "no, none"）
        head = n.split()[0]
        if head in aff:
            return True
        if head in den:
            return False
        return None

    # ---- short ----
    def _score_short(self, gold: str, pred: str, meta: Dict[str, Any]) -> int:
        g_c, g_neg = self.analyze(gold)
        p_c, p_neg = self.analyze(pred)
        meta.update(gold_canon=g_c, pred_canon=p_c, gold_neg=g_neg, pred_neg=p_neg)

        # 否定极性不一致 -> 直接判错（no effusion vs effusion 必须分开）
        if g_neg != p_neg:
            meta["reason"] = "polarity_mismatch"
            return 0

        if g_c == p_c:
            meta["reason"] = "equal"
            return 1

        if not g_c and not p_c:
            # 两侧都剥成空（"none" vs "no findings"）且极性相同：视为同一阴性
            if g_neg and p_neg:
                meta["reason"] = "both_bare_negative"
                return 1
            meta["reason"] = "both_empty"
            return 0
        if not g_c or not p_c:
            # 一侧剩实体而另一侧为空（"absent" vs "no effusion"）无从比较
            meta["reason"] = "empty_side"
            return 0

        # 过短的串（<=2 字符）只允许完全相等，避免 "pe" 命中一切
        shorter = g_c if len(g_c) <= len(p_c) else p_c
        longer = p_c if shorter is g_c else g_c
        if len(shorter.strip()) <= 2 and len(shorter) != len(longer):
            meta["reason"] = "too_short"
            return 0

        if _token_contains(shorter, longer):
            meta["reason"] = "contains"
            return 1

        # 语序无关的实词包含（如 "consolidation of the left lower lobe"
        # vs "left lower lobe consolidation"）；过短的缩写串不参与，防误命中
        if len(shorter.strip()) > 2 and _token_subset(shorter, longer):
            meta["reason"] = "token_subset"
            return 1

        meta["reason"] = "mismatch"
        return 0

    # ---- measure ----
    def _score_measure(self, gold: str, pred: str, meta: Dict[str, Any]) -> int:
        g_vals = _parse_measures(gold)
        p_vals = _parse_measures(pred)
        meta.update(gold_values=g_vals, pred_values=p_vals)

        if not g_vals:
            # 金标没有可解析的测量值：退化为 short 判分（常见于 "not measurable"）
            s = self._score_short(gold, pred, meta)
            # _score_short 会写 reason，这里覆盖回"走过退化路径"，便于离线统计
            meta["fallback_short_score"] = s
            meta["fallback_short_reason"] = meta.get("reason", "")
            meta["reason"] = "no_gold_number_fallback_short"
            return 1 if s == 1 else 0
        if not p_vals:
            meta.update(reason="no_pred_number", half_credit=False)
            return 0

        # 双向取最小相对误差：多测量值时只要有一个对得上就算对
        best = float("inf")
        for gv in g_vals:
            for pv in p_vals:
                if gv == 0:
                    err = 0.0 if pv == 0 else float("inf")
                else:
                    err = abs(pv - gv) / abs(gv)
                best = min(best, err)

        meta["rel_err"] = None if best == float("inf") else round(best, 6)
        if best <= self.full_tol:
            meta.update(reason="within_full_tol", half_credit=False)
            return 1
        if best <= self.half_tol:
            # 半分档：线上仍返回 0（不进 Δ），只在 meta 留痕供离线分析
            meta.update(reason="within_half_tol", half_credit=True)
            return 0
        meta.update(reason="out_of_tol", half_credit=False)
        return 0

    # -------------------------------------------------------------- 准入
    def gradable(self, sample: Sample) -> bool:
        """该样本能否进 RL 的 Δ 通道（long 与未知类型一律 False）。"""
        atype = str(getattr(sample, "answer_type", "") or "").lower()
        if atype not in ("closed", "short", "measure"):
            return False
        gold = str(getattr(sample, "answer", "") or "").strip()
        if not gold:
            return False
        if atype == "measure":
            # measure 题若金标抽不出数字则不能稳定判分；退化到 short 时允许
            return True
        return bool(normalize(gold))

    def score_offline_judge(self, sample: Sample, pred: str) -> None:
        """占位：离线 judge 不得进入训练循环，故意抛异常。"""
        raise NotImplementedError(
            "离线 judge 仅供评测，禁止在 RL 循环中调用；线上判分只能走 score()"
        )

    # -------------------------------------------------------------- 序列化
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": GRADER_VERSION,
            "synonyms": {k: list(v) for k, v in self.synonyms.items()},
            "full_tol": self.full_tol,
            "half_tol": self.half_tol,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any], cfg: Any = None) -> "DeterministicGrader":
        """从 to_dict 的结果恢复；配置里的容差被 d 覆盖（保证 round-trip 一致）。"""
        syn = {str(k): list(v) for k, v in (d.get("synonyms") or {}).items()}
        base = cfg
        if base is None and ("full_tol" in d or "half_tol" in d):
            base = {
                "rl": {
                    "grader": {
                        "measure_full_tol": d.get("full_tol"),
                        "measure_half_tol": d.get("half_tol"),
                    }
                }
            }
        obj = cls(base, synonyms=syn)
        if "full_tol" in d and d["full_tol"] is not None:
            obj.full_tol = float(d["full_tol"])
        if "half_tol" in d and d["half_tol"] is not None:
            obj.half_tol = float(d["half_tol"])
        return obj

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeterministicGrader):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        return (
            f"DeterministicGrader(n_synonyms={len(self.synonyms)}, "
            f"full_tol={self.full_tol}, half_tol={self.half_tol})"
        )


# ------------------------------------------------------------------ 内部辅助


def _negation_raw() -> List[str]:
    return list(NEGATION_PATTERNS)


__all__ = ["DeterministicGrader", "normalize", "GRADER_VERSION"]
