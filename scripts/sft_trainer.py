#!/usr/bin/env python3
"""Qwen3-VL-8B SFT 训练器。

只做两件事：读 training_set.py 导出的 messages，以及在 assistant 段做 loss。
观测 token 的掩码由 loss_mask 提供（若存在），否则退化为"只在 assistant role 上算 loss"。

    torchrun --nproc_per_node 8 scripts/sft_trainer.py \
        --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
        --train_file data/train/sft.jsonl --output_dir data/ckpt/sft

注意：本文件在无 GPU 的容器里**不会被实际执行**；它依赖 transformers/trl，
本地只做了语法与数据格式层面的兼容（用 --dry-run 检查数据）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medtaco.data.training_set import count_dirty_blocks  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--train_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--num_train_epochs", type=float, default=3)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--max_length", type=int, default=16384)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--bf16", type=bool, default=True)
    p.add_argument("--gradient_checkpointing", type=bool, default=True)
    p.add_argument("--deepspeed", default=None)
    p.add_argument("--attn_implementation", default="auto",
                   choices=["auto", "flash_attention_2", "sdpa", "eager"],
                   help="auto = 装了 flash_attn 就用 FA2，否则退回 sdpa")
    p.add_argument("--use_peft", action="store_true")
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", default="all-linear")
    p.add_argument("--dry-run", action="store_true", help="只校验数据格式，不加载模型")
    # 冒烟用：跑 N 步就停、只吃前 M 条。两者都为 0 表示正常训练。
    # 没有这两个开关时，想验证"能不能跑起来"必须等完三个 epoch。
    p.add_argument("--max_steps", type=int, default=0, help=">0 时只跑这么多步")
    p.add_argument("--limit_data", type=int, default=0, help=">0 时只用前 N 条样本")
    return p.parse_args()


# ---------------------------------------------------------------- 注意力实现


def _resolve_attn(want: str) -> str:
    """auto: 装了 flash_attn 用 FA2，否则退 sdpa。

    FA2 不是正确性前提，只是省显存提速。硬写 flash_attention_2 会让没装
    flash-attn 的机器在 from_pretrained 直接抛 ImportError，而 sdpa 在
    torch>=2.1 上是原生实现，Qwen3-VL 全参 SFT 完全跑得动。
    """
    if want and want != "auto":
        return want
    try:
        import flash_attn  # noqa: F401
    except Exception:  # noqa: BLE001 - 任何导入失败都退回 sdpa
        return "sdpa"
    return "flash_attention_2"


# ---------------------------------------------------------------- 数据


def load_records(path: str, limit: int = 0):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def validate_records(rows, need_images: bool = True) -> dict:
    """数据体检：训练前必须全部为 0 才算可以开跑。"""
    stats = {"n": len(rows), "missing_messages": 0, "no_image": 0,
             "no_assistant": 0, "missing_image_file": 0, "empty_assistant": 0,
             "dirty_blocks": 0}
    for r in rows:
        msgs = r.get("messages")
        if not msgs:
            stats["missing_messages"] += 1
            continue
        stats["dirty_blocks"] += count_dirty_blocks(msgs)
        imgs = r.get("images") or []
        if need_images and not imgs:
            stats["no_image"] += 1
        for p in imgs:
            if not os.path.exists(p):
                stats["missing_image_file"] += 1
        asst = [m for m in msgs if m.get("role") == "assistant"]
        if not asst:
            stats["no_assistant"] += 1
        elif not any(_msg_text(m).strip() for m in asst):
            stats["empty_assistant"] += 1
    return stats


def _msg_text(m) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") for b in c if isinstance(b, dict))


def build_dataset(path: str, processor, max_length: int, limit: int = 0):
    """把 jsonl 转成 HF Dataset。真正的 tokenize 放到 collator 里做，避免预展开占内存。"""
    from datasets import Dataset

    from medtaco.data.training_set import clean_messages

    rows = load_records(path, limit)
    ds = Dataset.from_list(rows)

    def _tok(example):
        # 必须清洗：Dataset 的 Arrow 往返会把内容块补成并集 schema（纯文本块多出
        # {"image": None}），而 Qwen3-VL 模板用 `'image' in content` 判图 —— 只看键
        # 不看值，于是每块都被当成一张图，image_grid_thw 越界。详见 clean_messages。
        out = processor.apply_chat_template(
            clean_messages(example["messages"]), tokenize=True,
            add_generation_prompt=False, return_dict=True,
            max_length=max_length, truncation=True,
        )
        # 图片**只在这里处理一次**：apply_chat_template 内部已经调过 image_processor，
        # out 里带了 pixel_values / image_grid_thw。再手调一次 image_processor(images=...)
        # 既重复（同一张图编码两遍），又会在 fast 处理器上直接抛
        # ValueError: Only returning PyTorch tensors is currently supported
        # （它不接受 return_tensors="np"，而 datasets 又存不了 torch.Tensor）。
        #
        # input_ids 是**嵌套**的 [[t1, t2, ...]]（batch 维，哪怕只有一条），必须压平：
        # 不压平的话 collator 拿到的是二维 list，拼批时直接
        # ValueError: expected sequence of length 1453 at dim 2 (got 1224)。
        ids = out["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        while ids and isinstance(ids[0], list):
            ids = ids[0]
        out["input_ids"] = ids
        am = out.get("attention_mask")
        if hasattr(am, "tolist"):
            am = am.tolist()
        while isinstance(am, list) and am and isinstance(am[0], list):
            am = am[0]
        if am is not None:
            out["attention_mask"] = am
        # loss_mask：训练侧在 collator 里用，保留原样
        if example.get("loss_mask"):
            out["loss_mask"] = example["loss_mask"][: len(ids)]
        return out

    return ds.map(_tok, remove_columns=ds.column_names, num_proc=8)


def _open_img(p):
    from PIL import Image
    return Image.open(p).convert("RGB")


# ---------------------------------------------------------------- collator

# Qwen3-VL 的特殊 token。写死而不查 tokenizer，是因为 collator 在 dataloader 的
# worker 里跑，不想为了三个 id 再反序列化一遍 tokenizer。
IM_START = 151644
IM_END = 151645


def build_labels(input_ids, pad_token_id: int, assistant_id: int = 77091):
    """只在 assistant 段算 loss：``<|im_start|>assistant\\n ... <|im_end|>`` 全置 -100。

    为什么不用 tokenizer 自带的 ``assistant_masks``：``apply_chat_template`` 只在
    ``return_assistant_tokens_mask=True`` 时给，而它要求模板里定义了 assistant 块；
    这里手写反而更透明 —— 判据就是「im_start 后面紧跟 assistant 角色 token」。

    观测轮（工具返回的图 + 【工具返回】文本）落在 user 段，因此**天然被 mask 掉**，
    不需要额外依赖 loss_mask。导出的 loss_mask 若存在，再与之相与。
    """
    labels = [-100] * len(input_ids)
    n = len(input_ids)
    for i, t in enumerate(input_ids):
        if t != IM_START:
            continue
        if i + 1 < n and input_ids[i + 1] == assistant_id:
            # 角色 token 本身与换行不参与 loss，从内容第一个 token 开始
            j = i + 2
            while j < n and input_ids[j] != IM_END:
                labels[j] = input_ids[j]
                j += 1
            if j < n:  # 包含 <|im_end|>，让模型学会收尾
                labels[j] = input_ids[j]
    return labels


class VLMCollator:
    """Qwen3-VL 的批内拼接。

    默认 collator 会沿新维度 stack，把 ``image_grid_thw`` 变成 ``[B, n, 3]``，而模型的
    ``fast_pos_embed_interpolate`` 只接受 ``[n, 3]``（实测
    ``TypeError: linspace() ... got (int, int, Tensor)``）。所以两样图片元数据必须
    **沿第 0 维 cat**，因为它们本来就是「整个 batch 的所有图」拼在一起。
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch

        maxlen = max(len(f["input_ids"]) for f in features)
        input_ids, attn, labels = [], [], []
        for f in features:
            ids = list(f["input_ids"])
            lab = build_labels(ids, self.pad_token_id)
            if f.get("loss_mask"):
                lm = list(f["loss_mask"])[: len(ids)]
                lab = [l if (k < len(lm) and lm[k]) else -100 for k, l in enumerate(lab)]
            pad = maxlen - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            attn.append([1] * len(ids) + [0] * pad)
            labels.append(lab + [-100] * pad)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        # 图片元数据：沿 0 维拼接，保持 [总图数, ...] 的语义
        pv = [torch.as_tensor(f["pixel_values"]) for f in features if f.get("pixel_values") is not None]
        if pv:
            batch["pixel_values"] = torch.cat(pv, dim=0).float()
        thw = [torch.as_tensor(f["image_grid_thw"]) for f in features if f.get("image_grid_thw") is not None]
        if thw:
            batch["image_grid_thw"] = torch.cat([t.reshape(-1, 3) for t in thw], dim=0).long()
        return batch


# ---------------------------------------------------------------- 主流程


def main() -> int:
    args = parse_args()

    if args.dry_run:
        rows = load_records(args.train_file, limit=200)
        stats = validate_records(rows, need_images=True)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        bad = sum(v for k, v in stats.items() if k != "n" and v)
        print("数据体检:", "通过" if bad == 0 else f"有问题 {bad} 处")
        return 0 if bad == 0 else 1

    import torch
    from peft import LoraConfig
    from transformers import AutoProcessor, Trainer, TrainingArguments
    from transformers import AutoModelForVision2Seq

    from medtaco.utils.config import load_config

    cfg = load_config(args.config)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    attn_impl = _resolve_attn(args.attn_implementation)
    print(f"[sft] attn_implementation={attn_impl}", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name_or_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation=attn_impl,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # 梯度检查点与 KV cache 不能共存；不关掉会在 backward 里报
        # 「only support gradient checkpointing with use_cache=False」。
        if getattr(model, "config", None) is not None:
            model.config.use_cache = False

    if args.use_peft:
        peft_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules, task_type="CAUSAL_LM",
        )
        model.add_adapter(peft_cfg)
        print(f"[sft] LoRA r={args.lora_r}")

    train_ds = build_dataset(args.train_file, processor, args.max_length,
                             limit=args.limit_data)
    if args.limit_data:
        print(f"[sft] limit_data={args.limit_data} -> {len(train_ds)} 条", flush=True)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
        remove_unused_columns=False,
        report_to=["wandb"] if os.environ.get("WANDB_PROJECT") else [],
        ddp_find_unused_parameters=False if local_rank != -1 else None,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      data_collator=VLMCollator(processor.tokenizer.pad_token_id))
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    if cfg.sft.gate:
        print("[sft] 训练完成。请用 scripts/check_gate.py 跑 §6 的四道验收闸门")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
