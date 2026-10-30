"""SFT collator 单测：labels 掩码范围、批内图片元数据的拼接维度。

这两条都是实测踩过的坑，且**只在真正 forward 时才炸**：
  - 没有 labels -> Trainer 报 "model did not return a loss"；
  - image_grid_thw 被 stack 成 [B, n, 3] -> linspace() 收到 Tensor。
单测里用手写的假 feature 就能锁住这两点，不需要 GPU。
"""
import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.sft_trainer import (  # noqa: E402
    IM_END,
    IM_START,
    VLMCollator,
    build_labels,
)

PAD = 151643
ASST = 77091
USER = 872


def _seq(roles_and_texts):
    """按 Qwen3-VL 的形态拼一段 ids：<im_start><role>\n<content><im_end>\n"""
    ids = []
    for role_id, content in roles_and_texts:
        ids += [IM_START, role_id]
        ids += list(content)
        ids += [IM_END]
    return ids


def test_labels_only_on_assistant_segments():
    user_part = [10, 11, 12]
    asst_part = [20, 21, 22]
    ids = _seq([(USER, user_part), (ASST, asst_part)])
    lab = build_labels(ids, PAD)

    n_user_start = 0
    # user 段（含 im_start / role / im_end）全 -100
    for k in range(0, 2 + len(user_part) + 1):
        assert lab[k] == -100, f"user 段第 {k} 个 token 不该算 loss"
    # assistant 内容 + im_end 参与
    start = 2 + len(user_part) + 1 + 2
    assert lab[start:start + len(asst_part)] == asst_part
    assert lab[start + len(asst_part)] == IM_END
    # 角色 token 本身不算 loss
    assert lab[start - 1] == -100


def test_labels_mask_all_observation_turns():
    """工具返回的图与文本都在 user 段，必须整体被 mask。"""
    obs = [30] * 8            # 模拟 <|vision_start|> + image_pad*n
    ids = _seq([(USER, [1, 2]), (ASST, [3, 4]), (USER, obs), (ASST, [5, 6])])
    lab = build_labels(ids, PAD)
    # 第一个 assistant 段正常参与 loss
    assert lab[lab.index(3):lab.index(3) + 3] == [3, 4, IM_END]
    # 观测轮（第二个 user 段）整体 -100 —— 找到它前面的 im_start 位置
    obs_start = ids.index(30) - 2          # 退回 <im_start><role>
    assert all(l == -100 for l in lab[obs_start:obs_start + 2 + len(obs) + 1])
    assert lab[lab.index(5):] == [5, 6, IM_END]


def test_collator_cats_image_metadata_not_stacks():
    torch = pytest.importorskip("torch")
    a = {"input_ids": [1, 2, 3], "pixel_values": torch.ones(4, 2),
         "image_grid_thw": torch.tensor([[1, 2, 2]])}
    b = {"input_ids": [4, 5], "pixel_values": torch.ones(6, 2),
         "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]])}
    batch = VLMCollator(PAD)([a, b])

    assert batch["input_ids"].shape == (2, 3)
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["labels"].shape == (2, 3)
    # 关键：沿 0 维拼接 -> [3, 3]，而不是 [2, 1, 3]
    assert batch["image_grid_thw"].shape == (3, 3)
    assert batch["pixel_values"].shape == (10, 2)
    # 填充位不参与 loss
    assert (batch["labels"][1, 2] == -100).item()


def test_collator_honours_explicit_loss_mask():
    torch = pytest.importorskip("torch")
    ids = _seq([(USER, [1, 2]), (ASST, [3, 4])])
    f = {"input_ids": ids, "loss_mask": [1] * len(ids),
         "pixel_values": torch.ones(1, 2),
         "image_grid_thw": torch.tensor([[1, 2, 2]])}
    batch = VLMCollator(PAD)([f])
    # loss_mask 全 1 不应把 assistant 段的标签也抹掉
    assert (batch["labels"] > -100).sum().item() == 3  # 3, 4, im_end
