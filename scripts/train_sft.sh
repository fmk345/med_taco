#!/usr/bin/env bash
# Qwen3-VL-8B SFT 冷启动。需要单机 8 卡（A100-40G 用 LoRA，H800-80G 可全参）。
#
# 用法:
#   bash scripts/train_sft.sh                      # 默认 configs/default.yaml
#   bash scripts/train_sft.sh --lora --epochs 2    # 40G 卡
#
# 前置: pip install -U transformers trl peft accelerate deepspeed
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$ROOT/configs/default.yaml}"
TRAIN_FILE="${TRAIN_FILE:-$ROOT/data/train/sft.jsonl}"
OUT="${OUT:-$ROOT/data/ckpt/sft}"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
NPROC="${NPROC:-8}"

USE_LORA=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lora) USE_LORA=1; shift ;;
    --epochs) EXTRA_ARGS+=(--num_train_epochs "$2"); shift 2 ;;
    --lr) EXTRA_ARGS+=(--learning_rate "$2"); shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "训练数据不存在: $TRAIN_FILE" >&2
  echo "先跑: python scripts/pipeline.py --stage export --for sft" >&2
  exit 1
fi

export WANDB_PROJECT="${WANDB_PROJECT:-medtaco}"
export TOKENIZERS_PARALLELISM=false

if [[ "$USE_LORA" == "1" ]]; then
  LORA_ARGS=(--use_peft --lora_r 64 --lora_alpha 128 --lora_dropout 0.05
             --lora_target_modules all-linear)
else
  LORA_ARGS=()
fi

echo "[sft] model=$MODEL data=$TRAIN_FILE out=$OUT nproc=$NPROC lora=$USE_LORA"

torchrun --nproc_per_node="$NPROC" "$ROOT/scripts/sft_trainer.py" \
  --model_name_or_path "$MODEL" \
  --train_file "$TRAIN_FILE" \
  --output_dir "$OUT" \
  --config "$CONFIG" \
  --bf16 True \
  --num_train_epochs 3 \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_length 16384 \
  --warmup_ratio 0.03 \
  --logging_steps 5 \
  --save_steps 200 \
  --save_total_limit 3 \
  --gradient_checkpointing True \
  --deepspeed "$ROOT/configs/deepspeed_zero3.json" \
  "${LORA_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
