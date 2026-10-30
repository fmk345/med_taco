"""把教师轨迹的 final_answer 回填成 Sample.answer（只填可判分题型）。

为什么需要：``qgen`` 只有 G4（measure）能确定性造出答案，G1/G2/G3/G5 的
``Sample.answer`` 一律是空串（见 ``pipeline.py`` 里 ``_mock_ask_fn`` 的注释）。
而 ``DeterministicGrader.gradable()`` 要求金标非空，于是整批题在 RL 侧不可判分。

本脚本做的事很窄：

  * 只填 ``answer_type ∈ {closed, short, measure}`` 的题。``long`` 不填——
    线上判分器对 long 恒返回 not_gradable，填了反而制造「有金标却不可判」的假象。
  * 只在 ``answer`` 为空时填，已有答案（例如 G4 的确定性答案）**绝不覆盖**。
  * 写入 ``meta['answer_source'] = 'teacher_final_answer'`` 与来源轨迹 id，
    这样下游能区分「人工/确定性金标」和「教师自答金标」。

⚠️ 诚实声明：教师自答答案作为金标，衡量的是「学生是否复现教师」，不是
「是否医学正确」。§6 闸门里 ``correct`` 一列因此只能解释为**蒸馏一致性**，
不能当成临床准确率。真正的准确率需要独立标注（SLAKE/VQA-RAD 等），见 scout 报告。

用法::

    python scripts/backfill_answers.py --dry-run     # 只看会填多少条
    python scripts/backfill_answers.py               # 就地重写 questions.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medtaco.utils.io import iter_jsonl

#: 只回填这些题型；long 不可线上判分，不回填
GRADABLE_TYPES = ("closed", "short", "measure")


def _answer_type(row: Dict[str, Any]) -> str:
    """answer_type 在顶层；老数据可能落在 meta 里，两处都认。"""
    at = row.get("answer_type")
    if at:
        return str(at)
    meta = row.get("meta") or {}
    return str(meta.get("answer_type") or "")


def _traj_answers(paths: Iterable[str]) -> Dict[str, Tuple[str, str]]:
    """读轨迹文件，返回 ``{sample_id: (final_answer, traj_id)}``。

    多个轨迹文件都给同一条样本答案时，**先读到的优先**（validated 排在 teacher
    后面传，这样通过验证的版本赢）。空答案不占位。
    """
    out: Dict[str, Tuple[str, str]] = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        for d in iter_jsonl(path):
            t = d.get("traj") or d
            sid = t.get("sample_id")
            ans = (t.get("final_answer") or "").strip()
            if not sid or not ans or sid in out:
                continue
            out[sid] = (ans, str(t.get("id") or ""))
    return out


def backfill(
    questions_path: str,
    traj_paths: List[str],
    dry_run: bool = False,
    backup: bool = True,
) -> Dict[str, Any]:
    """就地回填；返回统计。``dry_run=True`` 时不写盘。"""
    answers = _traj_answers(traj_paths)
    rows: List[Dict[str, Any]] = []
    stats = {
        "total": 0,
        "filled": 0,
        "already_had_answer": 0,
        "not_gradable_type": 0,
        "no_teacher_answer": 0,
        "by_type": {},
    }
    for row in iter_jsonl(questions_path):
        stats["total"] += 1
        at = _answer_type(row)
        if (row.get("answer") or "").strip():
            stats["already_had_answer"] += 1
            rows.append(row)
            continue
        if at not in GRADABLE_TYPES:
            stats["not_gradable_type"] += 1
            rows.append(row)
            continue
        hit = answers.get(row["id"])
        if hit is None:
            stats["no_teacher_answer"] += 1
            rows.append(row)
            continue
        ans, traj_id = hit
        row["answer"] = ans
        meta = dict(row.get("meta") or {})
        meta["answer_source"] = "teacher_final_answer"
        meta["answer_source_traj"] = traj_id
        row["meta"] = meta
        stats["filled"] += 1
        stats["by_type"][at] = stats["by_type"].get(at, 0) + 1
        rows.append(row)

    if dry_run:
        return stats

    if backup:
        bak = questions_path + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(questions_path, bak)
    tmp = questions_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, questions_path)
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="教师答案回填")
    ap.add_argument("--questions", default="data/questions/questions.jsonl")
    ap.add_argument("--trajs", nargs="*", default=None,
                    help="轨迹文件，后者优先级低；默认 validated + teacher")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args(argv)

    trajs = args.trajs
    if trajs is None:
        base = os.path.dirname(args.questions)
        parent = os.path.dirname(base)
        tdir = os.path.join(parent, "trajectories")
        trajs = [os.path.join(tdir, "validated.jsonl"),
                 os.path.join(tdir, "teacher.jsonl")]
        trajs = [p for p in trajs if os.path.exists(p)]
    if not trajs:
        print("[backfill] 找不到任何轨迹文件", file=sys.stderr)
        return 2

    stats = backfill(args.questions, trajs, dry_run=args.dry_run,
                     backup=not args.no_backup)
    tag = "[dry-run] " if args.dry_run else ""
    print(f"{tag}回填 {stats['filled']}/{stats['total']}；"
          f"原本有答案 {stats['already_had_answer']}；"
          f"不可判分题型跳过 {stats['not_gradable_type']}；"
          f"无教师答案 {stats['no_teacher_answer']}；"
          f"分型 {stats['by_type']}")
    print(f"{tag}轨迹来源: {trajs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
