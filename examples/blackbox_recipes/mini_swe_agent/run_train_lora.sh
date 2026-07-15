#!/usr/bin/env bash
# FSDP + LoRA V1 local training for the blackbox mini-swe recipe.
#
# Uses verl.trainer.main_ppo with the V1 unified trainer (colocate_async mode).
# Trainer and rollout share all GPUs — vLLM sleeps during training and wakes
# up for rollout. Supports both CUDA GPUs and Ascend NPUs.
#
# Usage:
#   bash examples/blackbox_recipes/mini_swe_agent/run_train_lora.sh
#
# All configurable via environment variables (see defaults below).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${REPO_ROOT}"

# ── Model & data ────────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-${HOME}/models/Qwen3.5-9B}"
TRAIN_DATA="${TRAIN_DATA:-${HOME}/data/swe_agent/swe_rebench_filtered.parquet}"
VAL_DATA="${VAL_DATA:-${HOME}/data/swe_agent/swe_bench_verified.parquet}"

# ── V1 trainer ──────────────────────────────────────────────────────────────
TRAINER_MODE="${TRAINER_MODE:-colocate_async}"
NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-1}"
PARAMETER_SYNC_STEP="${PARAMETER_SYNC_STEP:-4}"

# ── Hardware ────────────────────────────────────────────────────────────────
NNODES="${NNODES:-1}"
PHYSICAL_DEVICES="${PHYSICAL_DEVICES:-8}"

# colocate_async: trainer and rollout share all devices via sleep/wake.
N_GPUS_PER_NODE_TRAIN="${N_GPUS_PER_NODE_TRAIN:-${PHYSICAL_DEVICES}}"
N_GPUS_PER_NODE_ROLLOUT="${N_GPUS_PER_NODE_ROLLOUT:-${PHYSICAL_DEVICES}}"

# ── Sequence lengths ────────────────────────────────────────────────────────
PROMPT_LENGTH="${PROMPT_LENGTH:-4096}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-131072}"
MAX_MODEL_LEN=$((PROMPT_LENGTH + RESPONSE_LENGTH))
PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-$((PROMPT_LENGTH + RESPONSE_LENGTH))}"

# ── Rollout parameters ──────────────────────────────────────────────────────
ENGINE="${ENGINE:-vllm}"
N="${N:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
ROLLOUT_TP="${ROLLOUT_TP:-${N_GPUS_PER_NODE_ROLLOUT}}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.7}"

# ── Algorithm ───────────────────────────────────────────────────────────────
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"

# ── FSDP training ───────────────────────────────────────────────────────────
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-True}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-True}"
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-False}"
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-True}"
ULYSSES_SP="${ULYSSES_SP:-1}"

# ── LoRA ────────────────────────────────────────────────────────────────────
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-16}"

# ── Agent parameters ────────────────────────────────────────────────────────
RUNNER="${RUNNER:-mini_swe}"
AGENT_MAX_TURNS="${AGENT_MAX_TURNS:-100}"
SWE_AGENT_TOOL_IMAGE="${SWE_AGENT_TOOL_IMAGE:-swr.cn-east-3.myhuaweicloud.com/openyuanrong/mini-swe-agent-tool:latest}"
SWE_AGENT_RUN_TIMEOUT="${SWE_AGENT_RUN_TIMEOUT:-7200}"
CONDA_ENV="${CONDA_ENV:-testbed}"
GATEWAY_COUNT="${GATEWAY_COUNT:-1}"
MAX_CONCURRENT_SESSIONS="${MAX_CONCURRENT_SESSIONS:-128}"
NUM_AGENT_WORKERS="${NUM_AGENT_WORKERS:-8}"
MAX_TURNS="${MAX_TURNS:-${AGENT_MAX_TURNS}}"

if [[ "${RUNNER}" != "mini_swe" ]]; then
    echo "Unknown RUNNER=${RUNNER}; this recipe currently supports mini_swe only" >&2
    exit 1
fi
AGENT_RUNNER_FQN="examples.blackbox_recipes.mini_swe_agent.mini_swe_agent_runner.mini_swe_agent_runner"

# ── AKernel (remote sandbox) ────────────────────────────────────────────────
AKERNEL_SERVER_ADDRESS="${AKERNEL_SERVER_ADDRESS:-}"
AKERNEL_TOKEN="${AKERNEL_TOKEN:-}"
AKERNEL_TUNNEL_SSL_VERIFY="${AKERNEL_TUNNEL_SSL_VERIFY:-0}"

# ── Logging & checkpointing ─────────────────────────────────────────────────
PROJECT_NAME="${PROJECT_NAME:-swe_agent_blackbox}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-swe_agent_lora_$(date +%Y%m%d_%H%M)}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-10}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
CKPTS_DIR="${CKPTS_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:--1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${PPO_MINI_BATCH_SIZE}}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"

# ── Environment variables ───────────────────────────────────────────────────
export AGENT_MAX_TURNS
export SWE_AGENT_EVAL_TIMEOUT="${SWE_AGENT_EVAL_TIMEOUT:-600}"
export SWE_AGENT_TOOL_IMAGE
export SWE_AGENT_RUN_TIMEOUT
export CONDA_ENV
export GATEWAY_COUNT
export AKERNEL_SERVER_ADDRESS
export AKERNEL_TOKEN
export AKERNEL_TUNNEL_SSL_VERIFY
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TRANSFER_QUEUE_ENABLE="${TRANSFER_QUEUE_ENABLE:-}"
export MAX_TURNS
export DEBUG_MODE="${DEBUG_MODE:-true}"
export SANDBOX_NAME_PREFIX="${SANDBOX_NAME_PREFIX:-mini-swe-}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/verl:${PYTHONPATH:-}"

echo "=== SWE-Agent Blackbox FSDP+LoRA Local Training ==="
echo "Model:       ${MODEL_PATH}"
echo "Train data:  ${TRAIN_DATA}"
echo "Val data:    ${VAL_DATA}"
echo "LoRA:        rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo "FSDP:        param_offload=${ACTOR_PARAM_OFFLOAD}, optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD}, torch_compile=${USE_TORCH_COMPILE}, ulysses_sp=${ULYSSES_SP}"
echo "Engine:      ${ENGINE} (rollout_tp=${ROLLOUT_TP}, gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL})"
echo "Runner:      ${RUNNER}"
echo "Turns:       agent_max_turns=${AGENT_MAX_TURNS}"
echo "Batch:       n=${N}, mini_bsz=${PPO_MINI_BATCH_SIZE}"
echo "Sequence:    prompt=${PROMPT_LENGTH}, response=${RESPONSE_LENGTH}"
echo "Trainer:     V1 ${TRAINER_MODE}"
echo "Resources:   trainer=${NNODES}x${N_GPUS_PER_NODE_TRAIN}, rollout=1x${N_GPUS_PER_NODE_ROLLOUT} (total ${PHYSICAL_DEVICES} devices)"
echo "Samples:     train_max=${TRAIN_MAX_SAMPLES}, val_max=${VAL_MAX_SAMPLES}"
echo "============================================================="

# ── Build MAIN_CMD ──────────────────────────────────────────────────────────

MAIN_CMD=(
    python3 -m verl.trainer.main_ppo
    --config-name=swe_agent_blackbox_fsdp_v1
    --config-path="${REPO_ROOT}/examples/blackbox_recipes/mini_swe_agent/config"
    hydra.searchpath="[pkg://verl.trainer.config]"
)

# Trainer
MAIN_CMD+=(
    "trainer.use_v1=True"
    "trainer.v1.trainer_mode=${TRAINER_MODE}"
    "trainer.v1.colocate_async.num_warmup_batches=${NUM_WARMUP_BATCHES}"
    "trainer.nnodes=${NNODES}"
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE_TRAIN}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "trainer.val_before_train=${VAL_BEFORE_TRAIN}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.default_local_dir=${CKPTS_DIR}"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    "transfer_queue.enable=True"
)

# Model & LoRA
MAIN_CMD+=(
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.model.lora_rank=${LORA_RANK}"
    "actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}"
)

# Data
MAIN_CMD+=(
    "data.train_files=['${TRAIN_DATA}']"
    "data.val_files=['${VAL_DATA}']"
    "data.train_max_samples=${TRAIN_MAX_SAMPLES}"
    "data.val_max_samples=${VAL_MAX_SAMPLES}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.val_batch_size=${VAL_BATCH_SIZE}"
    "data.max_prompt_length=${PROMPT_LENGTH}"
    "data.max_response_length=${RESPONSE_LENGTH}"
)

# Rollout
MAIN_CMD+=(
    "actor_rollout_ref.rollout.n=${N}"
    "actor_rollout_ref.rollout.name=${ENGINE}"
    "actor_rollout_ref.rollout.mode=async"
    "actor_rollout_ref.rollout.prompt_length=${PROMPT_LENGTH}"
    "actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH}"
    "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_MODEL_LEN}"
    "actor_rollout_ref.rollout.temperature=${TEMPERATURE}"
    "actor_rollout_ref.rollout.top_p=${TOP_P}"
    "actor_rollout_ref.rollout.top_k=${TOP_K}"
    "actor_rollout_ref.rollout.nnodes=${NNODES}"
    "actor_rollout_ref.rollout.n_gpus_per_node=${N_GPUS_PER_NODE_ROLLOUT}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}"
    "actor_rollout_ref.rollout.calculate_log_probs=true"
    "actor_rollout_ref.rollout.enable_chunked_prefill=true"
    "actor_rollout_ref.rollout.disable_log_stats=false"
    "actor_rollout_ref.rollout.multi_turn.enable=true"
    "actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1"
    "actor_rollout_ref.rollout.multi_turn.format=qwen3_coder"
    "actor_rollout_ref.rollout.agent.num_workers=${NUM_AGENT_WORKERS}"
    "actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter"
    "actor_rollout_ref.rollout.custom.agent_framework.gateway_count=${GATEWAY_COUNT}"
    "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.swe_agent.runner_fqn=${AGENT_RUNNER_FQN}"
    "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.swe_agent.dispatch_mode=ray_task"
    "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.swe_agent.max_concurrent_sessions=${MAX_CONCURRENT_SESSIONS}"
    "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.swe_agent.runner_kwargs.tool_image=${SWE_AGENT_TOOL_IMAGE}"
    "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.swe_agent.runner_kwargs.run_timeout=${SWE_AGENT_RUN_TIMEOUT}"
    "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.swe_agent.runner_kwargs.conda_env=${CONDA_ENV}"
)

# FSDP Actor
MAIN_CMD+=(
    "actor_rollout_ref.actor.strategy=fsdp"
    "actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}"
    "actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN}"
    "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}"
    "actor_rollout_ref.actor.optim.weight_decay=0.1"
    "actor_rollout_ref.actor.optim.lr_scheduler_type=constant"
    "actor_rollout_ref.actor.use_rollout_log_probs=true"
    "actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD}"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD}"
    "actor_rollout_ref.actor.fsdp_config.dtype=bfloat16"
    "actor_rollout_ref.actor.use_torch_compile=${USE_TORCH_COMPILE}"
    "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${ULYSSES_SP}"
)

# FSDP Ref
MAIN_CMD+=(
    "actor_rollout_ref.ref.strategy=fsdp"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
    "actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD}"
    "actor_rollout_ref.ref.fsdp_config.dtype=bfloat16"
    "actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${ULYSSES_SP}"
)

# Algorithm
MAIN_CMD+=(
    "algorithm.gamma=1.0"
    "algorithm.lam=1.0"
    "algorithm.adv_estimator=grpo"
    "algorithm.use_kl_in_reward=false"
    "algorithm.kl_ctrl.type=fixed"
    "algorithm.kl_ctrl.kl_coef=0.0"
    "algorithm.rollout_correction.bypass_mode=true"
)

# User overrides (pass additional hydra args via "$@")
MAIN_CMD+=("$@")

# ── Launch ───────────────────────────────────────────────────────────────────
echo "Launching: ${MAIN_CMD[*]:0:5} ..."
"${MAIN_CMD[@]}"
