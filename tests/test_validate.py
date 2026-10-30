"""medtaco/synth/validate.py 的单测：不联网、不读大数据、不依赖本地没装的包。"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest
from PIL import Image

from medtaco.synth.validate import (
    answers_match,
    area_matched_random_box,
    crop_array,
    crop_box_of,
    first_crop_step,
    iou,
    no_tool_ablation,
    normalize_answer,
    premise_perturbation,
    resolve_image,
    tool_necessity,
    validate_all,
)
from medtaco.utils.schema import Box, Sample, Step, ToolCall, ToolResult, Trajectory


# ---------------------------------------------------------------- 造数据


def make_image(path=None) -> np.ndarray:
    """64x64 假图：全黑背景 + 一块 22x22 亮斑（放在 gold crop 内）。"""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[21:43, 21:43] = 255
    if path is not None:
        Image.fromarray(img).save(str(path))
    return img


def make_sample(tmp_path, boxes: Optional[List[Box]] = None, answer: str = "nodule") -> Sample:
    p = tmp_path / "img.png"
    if not p.exists():
        make_image(p)
    return Sample(
        id="s1",
        image=str(p),
        question="Is there a nodule in the left lower lobe?",
        answer=answer,
        dataset="MS-CXR",
        tier="A",
        answer_type="closed",
        boxes=boxes if boxes is not None else [Box(20, 20, 44, 44, "nodule")],
    )


def make_traj(tmp_path, crop: bool = True, final_answer: str = "nodule") -> Trajectory:
    steps = [Step(text="先看一下整体。")]
    if crop:
        steps.append(Step(
            text="裁剪可疑区域。",
            call=ToolCall("crop", {"box": [20, 20, 44, 44]}),
            result=ToolResult(ok=True, image=str(tmp_path / "crop.png"),
                              meta={"crop_box": [20, 20, 44, 44]}),
        ))
    steps.append(Step(text="答：nodule。"))
    return Trajectory(id="t1", sample_id="s1", steps=steps, final_answer=final_answer,
                      source="teacher")


def bright_ask_fn(threshold: float = 0.5):
    """亮斑占比 > threshold 才答 nodule（模拟「看了真裁剪才答得对」）。"""

    def ask(prompt: str, image: np.ndarray) -> str:
        assert isinstance(prompt, str) and prompt
        frac = float((np.asarray(image)[..., 0] > 128).mean())
        return "nodule" if frac > threshold else "normal"

    return ask


def const_ask_fn(answer: str):
    def ask(prompt: str, image: np.ndarray) -> str:
        return answer

    return ask


def make_cfg(**over) -> Dict:
    cfg: Dict = {
        "seed": 42,
        "validate": {
            "premise_perturbation": {
                "enabled": True,
                "area_match_tol": 0.25,
                "n_replacement_trials": 3,
                "min_answer_change": 1,
            },
            "no_tool_ablation": {"enabled": True, "model": "qwen3-vl-8b"},
        },
    }
    for k, v in over.items():
        cfg["validate"][k].update(v)
    return cfg


# ---------------------------------------------------------------- 几何与采样


def test_iou_and_crop_array():
    a, b = Box(0, 0, 10, 10), Box(5, 5, 15, 15)
    assert iou(a, b) == pytest.approx(25 / 175)
    assert iou(a, Box(20, 20, 30, 30)) == 0.0
    img = make_image()
    patch = crop_array(img, Box(20, 20, 44, 44))
    assert patch.shape == (24, 24, 3)
    assert (patch[..., 0] > 128).mean() > 0.8


def test_area_matched_random_box_constraints():
    """固定 seed：框确实满足面积约束、IoU 约束、落在图内，且可复现。"""
    ref = Box(20, 20, 44, 44)
    gold = [ref]
    b1 = area_matched_random_box(ref, 64, 64, area_tol=0.25, max_iou=0.1,
                                 rng=random.Random(7), avoid=gold)
    b2 = area_matched_random_box(ref, 64, 64, area_tol=0.25, max_iou=0.1,
                                 rng=random.Random(7), avoid=gold)
    assert b1 is not None and b2 is not None
    # 同 seed 同结果
    assert b1.as_tuple() == b2.as_tuple()
    # 面积匹配
    assert 0.75 <= b1.area / ref.area <= 1.25
    # IoU 约束
    assert iou(b1, ref) <= 0.1 + 1e-12
    # 落在图内
    assert 0 <= b1.x0 < b1.x1 <= 64 and 0 <= b1.y0 < b1.y1 <= 64
    # 与金标框不重合
    assert iou(b1, Box(20, 20, 44, 44, "nodule")) <= 0.1 + 1e-12


def test_area_matched_random_box_exact_values_fixed_seed():
    """固定 seed 的数值断言（回归保护：改动采样顺序会立刻炸）。"""
    ref = Box(20, 20, 44, 44)
    b = area_matched_random_box(ref, 64, 64, area_tol=0.25, max_iou=0.1,
                                rng=random.Random(7), avoid=[ref])
    coords = tuple(round(v, 6) for v in b.as_tuple())
    assert coords == _EXPECTED_SEED7


def test_area_matched_random_box_impossible_returns_none():
    """金标框几乎占满整图时，等面积且 IoU ≤ 0.1 的框不存在，50 次后放弃。"""
    ref = Box(0, 0, 64, 64)
    assert area_matched_random_box(ref, 64, 64, rng=random.Random(1), avoid=[ref]) is None


# ---------------------------------------------------------------- 轨迹解析


def test_first_crop_step_and_box(tmp_path):
    traj = make_traj(tmp_path)
    step = first_crop_step(traj)
    assert step is not None and step.call.name == "crop"
    box = crop_box_of(step)
    assert box.as_tuple() == (20, 20, 44, 44)


def test_crop_box_falls_back_to_call_args(tmp_path):
    traj = make_traj(tmp_path)
    traj.steps[1].result = ToolResult(ok=True, image="x.png", meta={})
    assert crop_box_of(traj.steps[1]).as_tuple() == (20, 20, 44, 44)


def test_resolve_image_accepts_ndarray_dict_and_path(tmp_path):
    img = make_image(tmp_path / "a.png")
    assert resolve_image(img).shape == (64, 64, 3)
    assert resolve_image({"image": img}).shape == (64, 64, 3)
    assert resolve_image(str(tmp_path / "a.png")).shape == (64, 64, 3)
    sample = make_sample(tmp_path)
    assert resolve_image(None, sample).shape == (64, 64, 3)
    assert resolve_image(None, None) is None


# ---------------------------------------------------------------- 答案比较


def test_normalize_and_match():
    assert normalize_answer("  Yes. ") == "yes"
    assert normalize_answer(None) == ""
    assert answers_match("Yes", "yes.")
    assert answers_match("present", "Yes")
    assert answers_match("No effusion.", "no effusion")
    assert answers_match("No", "no.")
    assert not answers_match("yes", "no")
    # 否定极性必须一致：「no effusion」不能匹配「effusion」
    assert not answers_match("no effusion", "effusion")
    assert answers_match("a small nodule", "nodule")
    assert not answers_match("nodule in left lobe", "nodule in right lobe")


# ---------------------------------------------------------------- 检查一：前提扰动


def test_premise_perturbation_passes_when_answer_changes(tmp_path):
    img = make_image()
    sample = make_sample(tmp_path)
    traj = make_traj(tmp_path)
    v = premise_perturbation(traj, sample, img, bright_ask_fn(0.5), make_cfg())
    assert v.passed is True
    assert v.reason == ""
    assert v.checks["n_valid_trials"] == 3
    assert v.checks["n_answer_change"] == 3
    assert v.checks["change_rate"] == 1.0
    assert v.checks["baseline_answer"] == "nodule"
    # 每个随机框都满足面积与 IoU 约束
    ref = Box(20, 20, 44, 44)
    gold = sample.boxes[0]
    for box in v.checks["perturbed_boxes"]:
        b = Box(*box)
        assert 0.75 <= b.area / ref.area <= 1.25
        assert iou(b, gold) <= 0.1 + 1e-12


def test_premise_perturbation_decorative_crop(tmp_path):
    img = make_image()
    sample = make_sample(tmp_path)
    traj = make_traj(tmp_path)
    v = premise_perturbation(traj, sample, img, const_ask_fn("nodule"), make_cfg())
    assert v.passed is False
    assert v.reason == "decorative_crop"
    assert v.checks["n_answer_change"] == 0
    assert v.checks["change_rate"] == 0.0


def test_premise_perturbation_min_answer_change_two(tmp_path):
    """要求至少 2 次变化：只变 1 次应当不通过。"""
    img = make_image()
    sample = make_sample(tmp_path)
    traj = make_traj(tmp_path)
    calls = {"n": 0}

    def ask(prompt, image):
        calls["n"] += 1
        return "normal" if calls["n"] == 1 else "nodule"

    cfg = make_cfg(premise_perturbation={"min_answer_change": 2})
    v = premise_perturbation(traj, sample, img, ask, cfg)
    assert v.checks["n_answer_change"] == 1
    assert v.passed is False and v.reason == "decorative_crop"


def test_premise_perturbation_no_tool_trajectory(tmp_path):
    """无裁剪的 G3 负样本：裸答答对 → passed，reason=no_tool_trajectory。"""
    img = make_image()
    sample = make_sample(tmp_path)
    traj = make_traj(tmp_path, crop=False)
    v = premise_perturbation(traj, sample, img, const_ask_fn("nodule"), make_cfg())
    assert v.reason == "no_tool_trajectory"
    assert v.passed is True

    # 裸答答错 → 这个负样本与题面矛盾
    traj2 = make_traj(tmp_path, crop=False)
    v2 = premise_perturbation(traj2, sample, img, const_ask_fn("normal"), make_cfg())
    assert v2.reason == "no_tool_trajectory"
    assert v2.passed is False


def test_premise_perturbation_disabled_and_missing_image(tmp_path):
    img = make_image()
    sample = make_sample(tmp_path)
    traj = make_traj(tmp_path)
    v = premise_perturbation(traj, sample, img, const_ask_fn("x"),
                             make_cfg(premise_perturbation={"enabled": False}))
    assert v.passed is True and v.reason == "disabled"

    traj2 = make_traj(tmp_path)
    v2 = premise_perturbation(traj2, sample, None, const_ask_fn("x"), make_cfg())
    # sample.image 存在，能读盘；换成两条都拿不到的情况
    assert v2.passed in (True, False)
    s2 = Sample(id="s2", image="/nonexistent/x.png", question="q", answer="a")
    v3 = premise_perturbation(traj2, s2, None, const_ask_fn("x"), make_cfg())
    assert v3.passed is False and v3.reason == "image_unavailable"


# ---------------------------------------------------------------- 检查二：无工具消融


def test_no_tool_ablation(tmp_path):
    sample = make_sample(tmp_path)
    traj = make_traj(tmp_path)
    v = no_tool_ablation(traj, sample, const_ask_fn("nodule"), make_cfg())
    assert v.passed is False and v.reason == "solved_without_tool"
    assert v.checks["answered_without_tool"] is True

    v2 = no_tool_ablation(traj, sample, const_ask_fn("normal"), make_cfg())
    assert v2.passed is True and v2.reason == ""

    v3 = no_tool_ablation(traj, sample, const_ask_fn("x"),
                          make_cfg(no_tool_ablation={"enabled": False}))
    assert v3.passed is True and v3.reason == "disabled"


def test_no_tool_ablation_ask_fn_error(tmp_path):
    sample = make_sample(tmp_path)

    def boom(prompt, image):
        raise RuntimeError("model down")

    v = no_tool_ablation(make_traj(tmp_path), sample, boom, make_cfg())
    assert v.passed is False and v.reason == "ask_fn_error"


# ---------------------------------------------------------------- 检查三 + 组合


def test_tool_necessity_combines_both(tmp_path):
    img = make_image()
    sample = make_sample(tmp_path)
    traj = make_traj(tmp_path)
    v = validate_all(traj, sample, img, bright_ask_fn(0.5), make_cfg())
    assert v.passed is True
    assert v.checks["final"]["decision"] == "tool_necessary"
    assert v.checks["premise_perturbation"]["checks"]["n_answer_change"] == 3
    assert v.checks["no_tool_ablation"]["checks"]["answered_without_tool"] is False


def test_validate_all_decorative(tmp_path):
    img = make_image()
    sample = make_sample(tmp_path)
    traj = make_traj(tmp_path)
    # 无论给什么图都答 nodule：无工具也答对 + 扰动后不变
    v = validate_all(traj, sample, img, const_ask_fn("nodule"), make_cfg())
    assert v.passed is False
    assert v.reason == "decorative_crop"
    assert traj.meta["validate"]["premise_perturbation"]["reason"] == "decorative_crop"


def test_validate_all_solved_without_tool(tmp_path):
    """扰动后答案变了，但裸答也答对 → 工具非必需。"""
    img = make_image()
    sample = make_sample(tmp_path)
    traj = make_traj(tmp_path)

    def ask(prompt, image):
        arr = np.asarray(image)
        if arr.shape[0] == 64:                       # 整图：裸答
            return "nodule"
        frac = float((arr[..., 0] > 128).mean())
        return "nodule" if frac > 0.5 else "normal"  # 随机裁剪：答错

    v = validate_all(traj, sample, img, ask, make_cfg())
    assert v.passed is False and v.reason == "solved_without_tool"


def test_validate_all_negative_sample(tmp_path):
    img = make_image()
    sample = make_sample(tmp_path)
    sample.meta = {"need_crop": False}          # qgen 给 G3 负样本打的标
    sample.boxes = []
    traj = make_traj(tmp_path, crop=False)
    v = validate_all(traj, sample, img, const_ask_fn("nodule"), make_cfg())
    assert v.passed is True
    assert v.checks["premise_perturbation"]["reason"] == "no_tool_trajectory"
    assert v.checks["final"]["decision"] == "negative_sample"

    traj2 = make_traj(tmp_path, crop=False)
    v2 = validate_all(traj2, sample, img, const_ask_fn("normal"), make_cfg())
    assert v2.passed is False and v2.reason == "negative_sample_unsolved"


def test_validate_all_expected_crop_missing(tmp_path):
    """带金框（need_crop=True）却一步没裁：归因到轨迹，不是负样本。"""
    img = make_image()
    sample = make_sample(tmp_path)               # 默认带 Box，meta 无 need_crop
    traj = make_traj(tmp_path, crop=False)
    v = validate_all(traj, sample, img, const_ask_fn("nodule"), make_cfg())
    assert v.passed is False and v.reason == "expected_crop_missing"
    assert v.checks["final"]["decision"] == "expected_crop_missing"


def test_validate_all_unexpected_crop(tmp_path):
    """need_crop=False 的样本却裁了一刀：不该学，但不能算 solved_without_tool。"""
    img = make_image()
    sample = make_sample(tmp_path)
    sample.meta = {"need_crop": False}
    traj = make_traj(tmp_path)                   # 带一次裁剪
    v = validate_all(traj, sample, img, const_ask_fn("nodule"), make_cfg())
    assert v.passed is False and v.reason == "unexpected_crop"


def test_tool_necessity_missing_premise(tmp_path):
    v = tool_necessity(make_traj(tmp_path), make_sample(tmp_path), make_cfg())
    assert v.passed is False and v.reason == "missing_premise_check"


# ---------------------------------------------------------------- 固定 seed 数值

# 由 area_matched_random_box(ref=Box(20,20,44,44), 64x64, tol=0.25, IoU<=0.1,
# rng=Random(7)) 首次生成；写死用于回归保护，改动采样顺序会立刻失败。
_EXPECTED_SEED7: Tuple[float, ...] = (2.894078, 4.02598, 25.464545, 23.644481)
