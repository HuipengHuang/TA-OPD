#!/usr/bin/env bash
# TA-OPD | text models | vLLM rollout and teacher | FSDP training
#
# Usage:
#   bash train.sh [--student MODEL] [--teacher MODEL] [--loss_mode MODE] [hydra overrides...]
#
# 除了 student / teacher / loss_mode, 其余超参数都固定为论文默认设置 (Table 4):
#   batch 72 / mini-batch 36 / rollout 4 / lr 1e-6 / 300 steps
#   max prompt 1024 / max response 7168 / train temp 1.0 / top-p 1.0
#   eval temp 0.7 / eval top-p 0.95

set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${ROOT_DIR}/verl:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

# ==================== 默认配置 ====================
STUDENT_MODEL=Qwen/Qwen3-1.7B
TEACHER_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
LOSS_MODE=taopd_reverse_kl_topk
TOPK=16

# 论文默认值 (Table 4), 同时用于实验命名
LR=1e-6
MAX_RESPONSE_LENGTH=7168

TRAIN_DATA=${ROOT_DIR}/training_data/dapo-math-17k-processed.parquet
VAL_DATA=${ROOT_DIR}/training_data/valid_math500.parquet
OUTPUT_DIR=${ROOT_DIR}/outputs

# 8 卡: 6 卡训练 student, 2 卡跑 teacher 推理
ACTOR_GPUS=6
TEACHER_GPUS=2
# ==================================================

usage() {
  cat <<EOF
Usage: bash $(basename "${BASH_SOURCE[0]}") [options] [hydra overrides...]

Options:
  --student MODEL      Student model ID or local path (default: ${STUDENT_MODEL})
  --teacher MODEL      Teacher model ID or local path (default: ${TEACHER_MODEL})
  --loss_mode MODE     normalized_reverse_kl_topk | taopd_reverse_kl_topk | sample_corrected_ta_opd
                       (default: ${LOSS_MODE})
  -h, --help           Show this message

Any other argument is passed through to Hydra, e.g.:
  bash train.sh --loss_mode normalized_reverse_kl_topk actor_rollout_ref.actor.optim.lr=5e-7
EOF
}

HYDRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --student)   STUDENT_MODEL=$2; shift 2 ;;
    --teacher)   TEACHER_MODEL=$2; shift 2 ;;
    --loss_mode) LOSS_MODE=$2;     shift 2 ;;
    -h|--help)   usage; exit 0 ;;
    *)           HYDRA_ARGS+=("$1"); shift ;;
  esac
done

case "${LOSS_MODE}" in
  normalized_reverse_kl_topk|taopd_reverse_kl_topk|sample_corrected_ta_opd) ;;
  *)
    echo "Unsupported --loss_mode: ${LOSS_MODE}" >&2
    echo "Choose normalized_reverse_kl_topk, taopd_reverse_kl_topk, or sample_corrected_ta_opd." >&2
    exit 2
    ;;
esac

for data_file in "${TRAIN_DATA}" "${VAL_DATA}"; do
  if [[ ! -f "${data_file}" ]]; then
    echo "Data file not found: ${data_file}" >&2
    exit 2
  fi
done

STUDENT_NAME=$(basename "${STUDENT_MODEL}")
TEACHER_NAME=$(basename "${TEACHER_MODEL}")
PROJECT_NAME=dapo17k_${STUDENT_NAME}_from_${TEACHER_NAME}
EXPERIMENT_NAME=loss_${LOSS_MODE}_topk_${TOPK}_lr_${LR}_${MAX_RESPONSE_LENGTH}

mkdir -p "${OUTPUT_DIR}/rollouts" "${OUTPUT_DIR}/validation"

echo "Student:   ${STUDENT_MODEL}"
echo "Teacher:   ${TEACHER_MODEL}"
echo "Objective: ${LOSS_MODE} (top-k=${TOPK})"
echo "GPUs:      ${ACTOR_GPUS} actor + ${TEACHER_GPUS} teacher"
echo "Project:   ${PROJECT_NAME}"
echo "Exp:       ${EXPERIMENT_NAME}"
echo "Output:    ${OUTPUT_DIR}"

python3 -m verl.trainer.main_ppo \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="['${TRAIN_DATA}']" \
  data.val_files="['${VAL_DATA}']" \
  data.train_batch_size=72 \
  data.max_prompt_length=1024 \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=True \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  \
  reward.reward_manager.name=remote \
  critic.enable=False \
  \
  actor_rollout_ref.model.path="${STUDENT_MODEL}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.enable_activation_offload=True \
  actor_rollout_ref.actor.optim.lr="${LR}" \
  actor_rollout_ref.actor.ppo_mini_batch_size=36 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.max_model_len=8192 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  \
  distillation.enabled=True \
  distillation.nnodes=1 \
  distillation.n_gpus_per_node="${TEACHER_GPUS}" \
  distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}" \
  distillation.teacher_models.teacher_model.inference.name=vllm \
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.7 \
  distillation.teacher_models.teacher_model.inference.max_model_len=8193 \
  distillation.distillation_loss.loss_mode="${LOSS_MODE}" \
  distillation.distillation_loss.topk="${TOPK}" \
  distillation.distillation_loss.use_task_rewards=False \
  distillation.distillation_loss.use_policy_gradient=False \
  \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="${ACTOR_GPUS}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.logger='["console"]' \
  trainer.balance_batch=True \
  trainer.val_before_train=True \
  trainer.total_training_steps=300 \
  trainer.save_freq=30 \
  trainer.test_freq=10 \
  trainer.rollout_data_dir="${OUTPUT_DIR}/rollouts" \
  trainer.validation_data_dir="${OUTPUT_DIR}/validation" \
  trainer.default_local_dir="${OUTPUT_DIR}/checkpoints/${EXPERIMENT_NAME}" \
  ${HYDRA_ARGS[@]+"${HYDRA_ARGS[@]}"}

echo "==== Done ===="