#!/usr/bin/env bash
# TACO 式 GRPO 训练（分级前缀探针 + OGAR 分段路由）。需要 verl + 8 卡。
#
# 前置:
#   git clone https://github.com/volcengine/verl && cd verl && pip install -e .
#   export https_proxy=http://10.217.148.40:8080   # 装包时；训练时不需要
#
# 用法: bash scripts/train_rl.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$ROOT/configs/default.yaml}"
TRAIN_FILE="${TRAIN_FILE:-$ROOT/data/train/rl.jsonl}"
OUT="${OUT:-$ROOT/data/ckpt/rl}"
SFT_CKPT="${SFT_CKPT:-$ROOT/data/ckpt/sft}"
VERL_DIR="${VERL_DIR:-$ROOT/third_party/verl}"

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "RL 数据不存在: $TRAIN_FILE" >&2
  echo "先跑: python scripts/pipeline.py --stage export --for rl" >&2
  exit 1
fi
if [[ ! -d "$VERL_DIR" ]]; then
  echo "找不到 verl: $VERL_DIR" >&2
  echo "git clone https://github.com/volcengine/verl $VERL_DIR" >&2
  exit 1
fi

# 把我们的插件注册进 verl（方式取决于 verl 版本，见 verl_plugin/README.md）
export PYTHONPATH="$ROOT:$VERL_DIR:${PYTHONPATH:-}"
export MEDTACO_CONFIG="$CONFIG"

echo "[rl] init=$SFT_CKPT data=$TRAIN_FILE out=$OUT"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=taco_staged \
  algorithm.use_kl_in_reward=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.path="$SFT_CKPT" \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.agent.default_agent_loop=taco_agent \
  actor_rollout_ref.rollout.response_length=32768 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  data.train_files="$TRAIN_FILE" \
  data.train_batch_size=16 \
  data.max_prompt_length=4096 \
  data.max_response_length=28672 \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.project_name=medtaco \
  trainer.experiment_name=taco_qwen3vl8b \
  trainer.default_local_dir="$OUT" \
  trainer.total_epochs=1 \
  trainer.save_freq=100 \
  "$@"
