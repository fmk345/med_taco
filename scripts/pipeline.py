#!/usr/bin/env python3
"""MedTACO 全流程入口。

每个阶段独立可跑、可断点续传：

    python scripts/pipeline.py --stage collect
    python scripts/pipeline.py --stage qgen --limit 2000
    python scripts/pipeline.py --stage synth
    python scripts/pipeline.py --stage validate
    python scripts/pipeline.py --stage export --for sft
    python scripts/pipeline.py --stage eval --set SLAKE-EN
    python scripts/pipeline.py --stage all --dry-run

设计原则：
  - 每个阶段只读上游 jsonl、只写自己的 jsonl，产物落 paths.*；
  - 阶段之间不共享内存，任何一步都可以单独重跑；
  - 所有外部调用（模型 / API / 沙箱）都从参数注入，便于 mock 与离线跑。
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Callable, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medtaco.utils.config import load_config
from medtaco.utils.io import JsonlWriter, iter_jsonl
from medtaco.utils.schema import Sample, Trajectory, Verdict

STAGES = ["collect", "qgen", "synth", "validate", "export", "sft", "rl", "eval"]


# ---------------------------------------------------------------- 各阶段实现
# 每个函数签名统一为 (cfg, args) -> dict，返回统计信息并打印。


def stage_collect(cfg, args) -> dict:
    from medtaco.data.collect import audit_resolution, collect_dataset, scan_manifest

    out = os.path.join(cfg.paths.samples, "samples.jsonl")
    # --mock 时只走自造的 Mock 数据集：本机没有真实影像，否则下面每个解析器都 skip，
    # 整条链会静默跑出 0 条，看起来像"跑通了"。
    only = ["Mock"] if getattr(args, "mock", False) else None
    manifest = scan_manifest(cfg, only)
    # manifest 是**按图**一行的清单；解析要按数据集名去重后再跑，否则同一数据集会被
    # 解析 N 次（每次都产出 N 条、再靠 id 去重，慢且日志误导）。
    names: List[str] = []
    for row in manifest:
        if isinstance(row, dict):
            ds = row.get("dataset") or row.get("name")
            if ds and ds not in names:
                names.append(ds)
    print(f"[collect] manifest: {len(manifest)} 张图 / {len(names)} 个数据集 {names}")

    # 配置里列了、但 manifest 里一条都没产出的数据集：从 names 反推是推不出来的
    # （scan_manifest 按文件扫，没文件的数据集根本不出现在结果里），必须拿配置的
    # 键集合做差集，否则这类静默漏采在日志上完全看不见。
    # 注意只在跑完全部解析器之后报：数据集可能因列名不符/文件缺失产出 0 条但 manifest
    # 里仍有记录（按图扫描≠解析成功），所以「名字在不在 names 里」不是唯一的判据。
    configured = list(only) if only else list((cfg.data.datasets or {}).keys())

    def _warn_empty(ds: str) -> None:
        print(f"[warn] 数据集 {ds} 配置了但产出 0 条（缺标注文件？）")

    total = 0
    produced: set = set()
    with JsonlWriter(out) as w:
        for ds in names:
            try:
                n = 0
                for s in collect_dataset(ds, cfg):
                    if w.write(s):
                        n += 1
                total += n
                if n:
                    produced.add(ds)
                print(f"[collect] {ds}: 新增 {n}")
            except Exception as e:  # noqa: BLE001
                print(f"[collect] {ds} 失败，已跳过: {e}")
    # 两路合并：配置了但不在 manifest（无任何文件）的，与跑了但 0 条（解析失败/缺标注）的。
    for ds in configured:
        if ds not in names or ds not in produced:
            _warn_empty(ds)
    # 审计要覆盖**文件全量**而不是本轮新增：续跑时新增是 0，按新增算会误报"共 0 条"。
    all_samples = [Sample.from_json(d) for d in iter_jsonl(out)]
    stats = audit_resolution(all_samples)
    print(f"[collect] 本轮新增 {total} 条，文件内共 {len(all_samples)} 条；分辨率审计: {stats}")
    return {"n_new": total, "n": len(all_samples), **stats}


def stage_qgen(cfg, args) -> dict:
    from medtaco.synth.qgen import filter_questions, generate

    src = os.path.join(cfg.paths.samples, "samples.jsonl")
    dst = os.path.join(cfg.paths.questions, "questions.jsonl")
    samples = [Sample.from_json(d) for d in iter_jsonl(src)]
    if args.limit:
        samples = samples[: args.limit]
    print(f"[qgen] 输入 {len(samples)} 条样本")

    out = generate(samples, cfg)
    print(f"[qgen] 生成 {len(out)} 条候选问题")
    kept = filter_questions(out, cfg)
    print(f"[qgen] 过滤后 {len(kept)} 条")

    with JsonlWriter(dst, resume=False) as w:
        for s in kept:
            w.write(s)
    return {"candidates": len(out), "kept": len(kept)}


def stage_synth(cfg, args) -> dict:
    from medtaco.synth.teacher import estimate_cost, run_batch

    src = os.path.join(cfg.paths.questions, "questions.jsonl")
    dst = os.path.join(cfg.paths.trajectories, "teacher.jsonl")
    samples = [Sample.from_json(d) for d in iter_jsonl(src)]
    if args.limit:
        samples = samples[: args.limit]

    cost = estimate_cost(len(samples), cfg, calls_per_traj=args.calls_per_traj)
    print(f"[synth] {len(samples)} 条；成本估算: {cost}")
    if args.dry_run:
        return {"n": len(samples), "cost": cost, "dry_run": True}

    # 裁剪图必须由确定性代码从 gold 框渲染：没有裁剪图，teacher 只能把三条 step
    # 全填 none，合成的轨迹就退化成"普通 VQA 答案"。
    from medtaco.synth.crops import prepare_crops

    crop_stats = prepare_crops(samples, cfg)
    print(f"[synth] 裁剪图: {crop_stats}")

    client, backend = _make_teacher(cfg, args, samples)
    print(f"[synth] teacher 后端: {backend}")

    failures: List[Dict] = []
    from medtaco.utils.io import seen_ids

    # 逐块落盘：4980 条在 8 卡上要跑 3 小时，等全部返回再写等于把中途被杀的风险
    # 全担在一次函数返回上。writer 的 id 集合同时充当续跑的去重表。
    with JsonlWriter(dst) as w:
        n = 0

        def _sink(t) -> None:
            nonlocal n
            if w.write(t):
                n += 1

        try:
            run_batch(samples, cfg, client=client,
                      existing_ids=seen_ids(dst), failure_sink=failures, on_traj=_sink)
        finally:
            w.flush()
    print(f"[synth] 写入 {n} 条轨迹；失败 {len(failures)} 条")
    if failures:
        fp = os.path.join(cfg.paths.logs, "synth_failures.jsonl")
        # 失败记录的主键是 custom_id（sample.id 的一对一映射），不是 'id'；
        # 用追加而不是覆盖，分块跑 synth 时不要把上一块的失败抹掉。
        with JsonlWriter(fp, id_key="custom_id") as w:
            for f in failures:
                w.write(f)
        by_err: Dict[str, int] = {}
        for f in failures:
            k = str(f.get("error", ""))[:60]
            by_err[k] = by_err.get(k, 0) + 1
        for k, v in sorted(by_err.items(), key=lambda kv: -kv[1])[:5]:
            print(f"[synth]   {v:4d}× {k}")
        print(f"[synth] 失败明细: {fp}")
    return {"n": n, "failures": len(failures), "backend": backend, **crop_stats}


def _make_teacher(cfg, args, samples):
    """选 teacher 后端：mock / local / api，默认 auto。"""
    want = getattr(args, "teacher", "auto") or "auto"
    if args.mock:
        want = "mock"

    if want in ("auto", "local"):
        from medtaco.synth.local_teacher import DEFAULT_MODEL_PATH

        model_path = str(getattr(cfg.teacher, "local_model", "") or DEFAULT_MODEL_PATH)
        if os.path.isdir(model_path):
            from medtaco.synth.local_teacher import local_client_from_cfg

            return local_client_from_cfg(cfg), f"local({os.path.basename(model_path)})"
        if want == "local":
            raise SystemExit(f"--teacher local 但模型目录不存在: {model_path}")

    if want in ("auto", "api"):
        try:
            import anthropic  # noqa: F401
        except ImportError:
            if want == "api":
                raise SystemExit("--teacher api 但未安装 anthropic SDK")
        else:
            return anthropic.Anthropic(), f"api({cfg.teacher.model})"

    from medtaco.synth.teacher import mock_client

    print("[synth] 使用 mock 客户端（轨迹不可用于训练）")
    return mock_client(samples), "mock"


def stage_validate(cfg, args) -> dict:
    """三项防 bypassing。需要能问模型的 ask_fn；离线时用 mock。"""
    from medtaco.synth.validate import validate_all
    from medtaco.tools.ops import ToolContext, load_image

    src_t = os.path.join(cfg.paths.trajectories, "teacher.jsonl")
    src_s = os.path.join(cfg.paths.questions, "questions.jsonl")
    # mock 判定绝不能落到 export 会读的 validated.jsonl 上：mock 的 ask_fn 是
    # 查表替身，它的"通过/不通过"与被验模型无关。写进正式路径一次，就会把
    # 一批假判定混进训练集，而且从文件本身看不出来。真实链路用正式文件名。
    is_mock = bool(getattr(args, "mock", False)) or not os.environ.get(
        "MEDTACO_LOCAL_ENDPOINT")
    name = "validated_mock.jsonl" if is_mock else "validated.jsonl"
    dst = os.path.join(cfg.paths.trajectories, name)
    if is_mock:
        print(f"[validate] mock 模式：判定写入 {name}（export 不读这个文件）")

    samples = {d["id"]: Sample.from_json(d) for d in iter_jsonl(src_s)}
    trajs = [Trajectory.from_json(d) for d in iter_jsonl(src_t)]
    # 老轨迹里 step.result.image 可能是教师写的标签（'CROP_1'），按裁剪调用次数重排成真路径。
    # 幂等；合成进程还跑着老代码，不能指望源头已经修好。
    from medtaco.synth.crops import crops_dir
    from medtaco.synth.teacher import sanitize_images
    cdir = crops_dir(cfg)
    n_fix = 0
    for t in trajs:
        s0 = samples.get(t.sample_id)
        if s0 is not None:
            n_fix += sanitize_images(t, s0, cdir)
    if n_fix:
        print(f"[validate] 修正 {n_fix} 处裁剪图引用")
    if args.limit:
        trajs = trajs[: args.limit]

    ask_fn = _make_ask_fn(cfg, args)
    passed = 0
    reasons: Dict[str, int] = {}
    with JsonlWriter(dst) as w:
        for t in trajs:
            s = samples.get(t.sample_id)
            if s is None:
                continue
            try:
                img = load_image(os.path.join(cfg.paths.processed, s.image))
                ctx = ToolContext(image=img, meta=s.meta,
                                  out_dir=os.path.join(cfg.paths.processed, "_obs"),
                                  obs_max_side=cfg.teacher.observation_max_side,
                                  jpeg_quality=cfg.teacher.jpeg_quality)
                v: Verdict = validate_all(t, s, ctx, ask_fn, cfg)
            except Exception as e:  # noqa: BLE001
                v = Verdict(traj_id=t.id, passed=False, reason=f"error:{e}")
            reasons[v.reason or "ok"] = reasons.get(v.reason or "ok", 0) + 1
            if v.passed:
                passed += 1
                w.write({"id": t.id, "traj": t.to_json(), "verdict": v.to_json()})
    print(f"[validate] 通过 {passed}/{len(trajs)}；原因分布: {reasons}")
    return {"passed": passed, "total": len(trajs), "reasons": reasons}


def _make_ask_fn(cfg, args) -> Callable:
    if args.mock or not os.environ.get("MEDTACO_LOCAL_ENDPOINT"):
        if not args.mock:
            print("[validate] 未配置 MEDTACO_LOCAL_ENDPOINT，退回 mock")
        return _mock_ask_fn(cfg)
    # 真实实现：本地部署的 base 模型，走 OpenAI 兼容端点。
    # 这里保持接口清晰，具体端点从配置或环境变量读。
    endpoint = os.environ["MEDTACO_LOCAL_ENDPOINT"]

    import base64
    import io
    import json
    import urllib.request

    import numpy as np
    from PIL import Image

    def ask(prompt: str, image) -> str:
        buf = io.BytesIO()
        arr = image if isinstance(image, np.ndarray) else np.asarray(image)
        Image.fromarray(arr.astype("uint8")).save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode()
        body = json.dumps({
            "model": cfg.validate.no_tool_ablation.model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]}],
            "temperature": 0.0, "max_tokens": 64,
        }).encode()
        req = urllib.request.Request(endpoint, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)["choices"][0]["message"]["content"]

    return ask


def _mock_ask_fn(cfg) -> Callable:
    """离线替身：按（题面, 图）回查标准答案。

    不能简单地恒返回一个字符串——no_tool_ablation 的比较对象就是答案本身，
    恒答错的替身会让「工具必要性」这道检查失去意义：负样本（不需要工具、
    裸答就该答对）会被全部误判为 `negative_sample_unsolved` 丢掉。

    规则（模拟一个理想的"无工具即可作答"的模型）：
      - 无框样本（负样本）→ 直接给标准答案，裸答答对；
      - 有框样本（正样本）→ 给一个固定错答，裸答答错，于是"裁剪是承重的"成立。

    两个坑（都是实测踩出来的）：

    1. **题面高度重复**。4980 条问题只有 317 种不同题面（模板槽位没填完的
       G1/G3 更甚，"从影像整体看，最可能的诊断方向是什么？"一条就占 158 个样本）。
       只按题面建索引会让所有同题样本塌成一条，回查到的 `s.id` 根本不是本题，
       负样本拿不到自己的教师答案 → 又全掉进 `negative_sample_unsolved`。
       所以键必须是 (题面, 图像指纹)。
    2. 真实链路里 `sample.answer` 几乎总是空的（ChestX-ray14 只有 finding 标签，
       答案由教师写出来），`no_tool_ablation` 的基准因此是 `traj.final_answer`。
       负样本的"裸答答案"要从**已产出的教师轨迹**里取，而不是从空答案派生的
       占位符，否则每一条负样本都对不上自己的参考答案。
    """
    import hashlib

    from medtaco.synth.teacher import mock_answer_for
    from medtaco.tools.ops import load_image

    qmap: Dict[tuple, Sample] = {}
    path = os.path.join(cfg.paths.questions, "questions.jsonl")
    raw_rows = []
    for d in iter_jsonl(path):
        try:
            raw_rows.append(Sample.from_json(d))
        except Exception:  # noqa: BLE001
            continue

    # 885 张图 × ~17 ms ≈ 15 s，只在 mock 路径上付一次；按需缓存，不常驻内存
    fp_cache: Dict[str, str] = {}

    def _fp(image_path: str) -> str:
        if image_path not in fp_cache:
            try:
                arr = load_image(image_path)
                fp_cache[image_path] = hashlib.md5(
                    arr.tobytes()).hexdigest()[:16]
            except Exception:  # noqa: BLE001
                fp_cache[image_path] = ""
        return fp_cache[image_path]

    for s in raw_rows:
        qmap[(s.question.strip(), _fp(s.image))] = s

    # 教师对同一题给出的最终答案 = 无工具时的"理想回答"
    teacher_ans: Dict[str, str] = {}
    tpath = os.path.join(cfg.paths.trajectories, "teacher.jsonl")
    if os.path.exists(tpath):
        for d in iter_jsonl(tpath):
            sid = d.get("sample_id") or d.get("id") or ""
            fa = str(d.get("final_answer") or "").strip()
            if sid and fa:
                teacher_ans[sid] = fa

    def ask(prompt: str, image) -> str:
        # 提示词里含 "Question: <题面>" 一行；取不到就退回固定错答（保守方向：
        # 判成"工具必需"不会污染正样本，只会多留负样本，不是危险方向）。
        q = ""
        for line in str(prompt).splitlines():
            if line.startswith("Question:"):
                q = line[len("Question:"):].strip()
                break
        try:
            key = hashlib.md5(image.tobytes()).hexdigest()[:16]
        except Exception:  # noqa: BLE001
            key = ""
        s = qmap.get((q, key))
        if s is None:
            return "mock_answer"
        if not s.boxes:
            return teacher_ans.get(s.id) or mock_answer_for(s)
        return "mock_wrong"

    return ask


def stage_export(cfg, args) -> dict:
    from medtaco.data.training_set import render_trajectory, to_rl_record, to_sft_record
    from medtaco.synth.crops import crops_dir
    from medtaco.synth.teacher import sanitize_images

    src = os.path.join(cfg.paths.trajectories, "validated.jsonl")
    samples = {d["id"]: Sample.from_json(d)
               for d in iter_jsonl(os.path.join(cfg.paths.questions, "questions.jsonl"))}
    dst = os.path.join(cfg.paths.train, f"{args.for_}.jsonl")
    # resume=False：导出是**全量重算**，不是增量。默认的追加语义会把上一次
    # （哪怕是 mock 或旧口径）的记录留在文件里，而且从行数上完全看不出来——
    # 实测第一次导出时文件里混着 60 条 `traj-mock:*`，训练集直接被污染。
    n = 0
    n_skip = 0
    n_fix = 0
    n_not_teacher = 0
    n_mock = 0
    with JsonlWriter(dst, resume=False) as w:
        for row in iter_jsonl(src):
            traj = Trajectory.from_json(row["traj"])
            # mock 合成的轨迹 id 形如 `traj-mock:*`，它们和真实教师轨迹混在
            # 同一个 validated.jsonl 里（mock 的判定另存 validated_mock.jsonl，
            # 但 mock 期的轨迹若被搬到正式文件，这里要兜住）。
            if traj.source != "teacher":
                n_not_teacher += 1
                continue
            s = samples.get(traj.sample_id)
            if s is None:
                continue
            # Mock 数据集是 `--stage collect --mock` 造出来的合成图，只在链路自测时
            # 有意义。它和真实样本一起躺在 questions.jsonl 里，靠 id 前缀区分，
            # 一旦混进训练集就是拿假图教模型。
            if str(s.dataset).lower() == "mock":
                n_mock += 1
                continue
            n_fix += sanitize_images(traj, s, crops_dir(cfg))
            try:
                rendered = render_trajectory(s, traj, cfg.teacher.observation_max_side)
            except Exception as e:  # noqa: BLE001 - 单条渲染失败不能拖垮整批
                n_skip += 1
                print(f"[export] 跳过 {traj.id}: {type(e).__name__}: {e}")
                continue
            rec = (to_sft_record(rendered) if args.for_ == "sft"
                   else to_rl_record(rendered, traj))
            rec["id"] = traj.id
            if w.write(rec):
                n += 1
    if n_fix:
        print(f"[export] 修正 {n_fix} 处裁剪图引用")
    if n_skip:
        print(f"[export] 渲染失败跳过 {n_skip} 条")
    print(f"[export] {args.for_} -> {dst}，{n} 条")
    if n_not_teacher:
        print(f"[export] 跳过非教师轨迹 {n_not_teacher} 条")
    return {"n": n, "skipped": n_skip, "img_fixed": n_fix,
            "not_teacher": n_not_teacher}


def stage_sft(cfg, args) -> dict:
    """SFT 入口。真正的训练循环在 scripts/train_sft.py（需要 GPU）。"""
    train = os.path.join(cfg.paths.train, "sft.jsonl")
    print(f"[sft] 训练数据: {train}")
    print(f"[sft] 启动命令: bash scripts/train_sft.sh")
    if args.dry_run:
        return {"dry_run": True}
    raise SystemExit("请用 bash scripts/train_sft.sh 启动（需要 GPU 节点）")


def stage_rl(cfg, args) -> dict:
    print("[rl] TACO 式 GRPO。需要 verl + GPU。")
    print("[rl] 启动命令: bash scripts/train_rl.sh")
    print("[rl] 集成说明见 medtaco/rl/verl_plugin/README.md")
    if args.dry_run:
        return {"dry_run": True}
    raise SystemExit("请用 bash scripts/train_rl.sh 启动（需要 verl 与 GPU 节点）")


def stage_eval(cfg, args) -> dict:
    from medtaco.eval.metrics import run_veg

    print(f"[eval] 评测集: {args.set or cfg.eval.sets.medical}")
    print("[eval] 完整评测见 scripts/eval_all.py")
    return {"sets": args.set or list(cfg.eval.sets.medical)}


STAGE_FN: Dict[str, Callable] = {
    "collect": stage_collect, "qgen": stage_qgen, "synth": stage_synth,
    "validate": stage_validate, "export": stage_export,
    "sft": stage_sft, "rl": stage_rl, "eval": stage_eval,
}


# ---------------------------------------------------------------- CLI


def main() -> int:
    ap = argparse.ArgumentParser(description="MedTACO 全流程")
    ap.add_argument("--stage", required=True,
                    help=f"阶段之一: {STAGES} / all")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="K=V",
                    help="覆盖配置，如 --set rl.alpha_2=0.1")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（调试）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock", action="store_true", help="不调外部模型")
    ap.add_argument("--teacher", default="auto", choices=["auto", "api", "local", "mock"],
                    help="synth 用的教师：auto=有本地权重就用本地，否则 API，再否则 mock")
    ap.add_argument("--calls-per-traj", type=float, default=2.0)
    ap.add_argument("--for", dest="for_", default="sft", choices=["sft", "rl"])
    args = ap.parse_args()

    overrides = {}
    for kv in args.set:
        k, _, v = kv.partition("=")
        try:
            v = eval(v, {"__builtins__": {}}, {})  # noqa: S307 - 本地配置，允许数字/布尔
        except Exception:  # noqa: BLE001
            pass
        overrides[k.strip()] = v

    cfg = load_config(args.config, overrides)

    stages = STAGES if args.stage == "all" else [args.stage]
    for st in stages:
        if st not in STAGE_FN:
            print(f"未知阶段: {st}", file=sys.stderr)
            return 2
        print(f"\n===== {st} =====")
        try:
            info = STAGE_FN[st](cfg, args)
            print(f"[{st}] 完成: {info}")
        except SystemExit:
            raise
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            print(f"[{st}] 失败", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
