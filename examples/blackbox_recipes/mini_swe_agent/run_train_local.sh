#!/usr/bin/env bash
# Megatron + V1 async training for the blackbox mini-swe recipe (single-node, local).
#
# Uses verl.trainer.main_ppo with the V1 unified trainer. The default mode is
# separate_async, which uses separate trainer and rollout GPU pools.
#
# This is the single-machine version of run_train.sh: all Ray cluster bootstrap,
# `ray job submit` and runtime-env plumbing have been removed. The trainer is
# launched directly on the current node.
#
# Usage:
#   bash examples/blackbox_recipes/mini_swe_agent/run_train_local.sh
#
# All configurable via environment variables (see defaults below).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${REPO_ROOT}"

# ── Model & data ─────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-${HOME}/models/Qwen3.5-9B}"
TRAIN_DATA="${TRAIN_DATA:-${HOME}/data/swe_agent/swe_rebench_filtered.parquet}"
VAL_DATA="${VAL_DATA:-${HOME}/data/swe_agent/swe_bench_verified.parquet}"

# ── V1 trainer ───────────────────────────────────────────────────────────
TRAINER_MODE="${TRAINER_MODE:-separate_async}"
NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-1}"
SEPARATE_NUM_WARMUP_BATCHES="${SEPARATE_NUM_WARMUP_BATCHES:-${NUM_WARMUP_BATCHES}}"
PARAMETER_SYNC_STEP="${PARAMETER_SYNC_STEP:-4}"
CONFIG_NAME="${CONFIG_NAME:-swe_agent_blackbox_megatron_v1}"

# ── Hardware ─────────────────────────────────────────────────────────────
NNODES="${NNODES:-${NNODES_TRAIN:-1}}"
PHYSICAL_GPUS_PER_NODE="${PHYSICAL_GPUS_PER_NODE:-8}"
if [[ "${TRAINER_MODE}" == "separate_async" ]]; then
    N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${TRAIN_NGPUS_PER_NODE:-4}}"
    ROLLOUT_NNODES="${ROLLOUT_NNODES:-${NNODES_ROLLOUT:-${NNODES}}}"
    ROLLOUT_NGPUS_PER_NODE="${ROLLOUT_NGPUS_PER_NODE:-${NGPUS_PER_NODE_ROLLOUT:-4}}"
else
    N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${TRAIN_NGPUS_PER_NODE:-${PHYSICAL_GPUS_PER_NODE}}}"
    ROLLOUT_NNODES="${ROLLOUT_NNODES:-${NNODES_ROLLOUT:-0}}"
    ROLLOUT_NGPUS_PER_NODE="${ROLLOUT_NGPUS_PER_NODE:-${NGPUS_PER_NODE_ROLLOUT:-${N_GPUS_PER_NODE}}}"
fi

# ── Algorithm ────────────────────────────────────────────────────────────
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
ACTOR_LR="${ACTOR_LR:-1e-6}"

# ── Sequence lengths ─────────────────────────────────────────────────────
PROMPT_LENGTH="${PROMPT_LENGTH:-4096}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-131072}"
MAX_MODEL_LEN=$((PROMPT_LENGTH + RESPONSE_LENGTH))

# ── Rollout parameters ───────────────────────────────────────────────────
ENGINE="${ENGINE:-vllm}"
if [[ "${TRAINER_MODE}" == "separate_async" ]]; then
    GEN_TP="${GEN_TP:-${TP:-${ROLLOUT_NGPUS_PER_NODE}}}"
else
    GEN_TP="${GEN_TP:-${TP:-2}}"
fi
N="${N:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.7}"
UPDATE_WEIGHTS_BUCKET_MB="${UPDATE_WEIGHTS_BUCKET_MB:-2048}"

# ── Megatron training parallelism ────────────────────────────────────────
if [[ "${TRAINER_MODE}" == "separate_async" ]]; then
    TRAIN_TP="${TRAIN_TP:-${TP:-${N_GPUS_PER_NODE}}}"
else
    TRAIN_TP="${TRAIN_TP:-${TP:-8}}"
fi
TRAIN_PP="${TRAIN_PP:-1}"
TRAIN_CP="${TRAIN_CP:-1}"
OFFLOAD="${OFFLOAD:-True}"
OPTIMIZER_OFFLOAD_FRACTION="${OFFLOAD_FRACTION:-1.0}"
USE_MBRIDGE="${USE_MBRIDGE:-True}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"

# ── LoRA ─────────────────────────────────────────────────────────────────
# Set USE_LORA=true to switch the actor (and vLLM rollout) to LoRA fine-tuning.
# When disabled (default), no LoRA Hydra overrides are emitted and the launch
# path is identical to the full-parameter recipe.
USE_LORA="${USE_LORA:-false}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
# Comma-separated module names; default targets the Qwen3 dense attention/mlp.
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
# vLLM LoRA rollout knobs (only take effect when USE_LORA=true).
ROLLOUT_MAX_LORAS="${ROLLOUT_MAX_LORAS:-1}"
ROLLOUT_MAX_CPU_LORAS="${ROLLOUT_MAX_CPU_LORAS:-1}"

# ── Agent parameters ─────────────────────────────────────────────────────
# AGENT_MAX_TURNS is the agent's turn budget inside the sandbox: it becomes the
# mini-swe-agent step_limit (read by the runner via the AGENT_MAX_TURNS env var).
# Note: the trainer's multi_turn.max_assistant_turns is NOT enforced on the
# blackbox rollout path (AgentFrameworkRolloutAdapter), so it is not exposed here.
RUNNER="${RUNNER:-mini_swe}"
AGENT_MAX_TURNS="${AGENT_MAX_TURNS:-100}"
if [[ "${RUNNER}" == "mini_swe" ]]; then
    AGENT_RUNNER_FQN="examples.blackbox_recipes.mini_swe_agent.mini_swe_agent_runner.mini_swe_agent_runner"
    SWE_AGENT_TOOL_IMAGE="${SWE_AGENT_TOOL_IMAGE:-swr.cn-east-3.myhuaweicloud.com/openyuanrong/mini-swe-agent-tool:latest}"
else
    echo "Unknown RUNNER=${RUNNER}; this recipe currently supports mini_swe only" >&2
    exit 1
fi
SWE_AGENT_RUN_TIMEOUT="${SWE_AGENT_RUN_TIMEOUT:-7200}"
CONDA_ENV="${CONDA_ENV:-testbed}"
GATEWAY_COUNT="${GATEWAY_COUNT:-1}"
MAX_CONCURRENT_SESSIONS="${MAX_CONCURRENT_SESSIONS:-128}"
NUM_AGENT_WORKERS="${NUM_AGENT_WORKERS:-8}"
RUNNER_ARGS=(
    "actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter"
    "actor_rollout_ref.rollout.custom.agent_framework.gateway_count=${GATEWAY_COUNT}"
    "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.swe_agent.runner_fqn=${AGENT_RUNNER_FQN}"
    "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.swe_agent.dispatch_mode=ray_task"
    "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.swe_agent.max_concurrent_sessions=${MAX_CONCURRENT_SESSIONS}"
    "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.swe_agent.runner_kwargs.tool_image=${SWE_AGENT_TOOL_IMAGE}"
    "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.swe_agent.runner_kwargs.run_timeout=${SWE_AGENT_RUN_TIMEOUT}"
    "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.swe_agent.runner_kwargs.conda_env=${CONDA_ENV}"
)

# LoRA Hydra overrides (only emitted when USE_LORA=true). Spread across both the
# actor (training side) and rollout (vLLM inference side) sub-configs.
LORA_ARGS=()
if [[ "${USE_LORA}" == "true" ]]; then
    LORA_ARGS=(
        actor_rollout_ref.actor.use_lora=true
        actor_rollout_ref.actor.lora_rank=${LORA_RANK}
        actor_rollout_ref.actor.lora_alpha=${LORA_ALPHA}
        actor_rollout_ref.actor.lora_dropout=${LORA_DROPOUT}
        "+actor_rollout_ref.actor.target_modules=[${LORA_TARGET_MODULES}]"
        actor_rollout_ref.rollout.enable_lora=true
        actor_rollout_ref.rollout.max_loras=${ROLLOUT_MAX_LORAS}
        actor_rollout_ref.rollout.max_lora_rank=${LORA_RANK}
        actor_rollout_ref.rollout.max_cpu_loras=${ROLLOUT_MAX_CPU_LORAS}
    )
fi

# ── AKernel (remote sandbox) ─────────────────────────────────────────────
AKERNEL_SERVER_ADDRESS="${AKERNEL_SERVER_ADDRESS:-}"
AKERNEL_TOKEN="${AKERNEL_TOKEN:-}"
AKERNEL_TUNNEL_SSL_VERIFY="${AKERNEL_TUNNEL_SSL_VERIFY:-0}"

# ── Logging & checkpointing ──────────────────────────────────────────────
PROJECT_NAME="${PROJECT_NAME:-swe_agent_blackbox}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-swe_agent_$(date +%Y%m%d_%H%M)}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-10}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
CKPTS_DIR="${CKPTS_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-${MAX_SAMPLES:--1}}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-${MAX_SAMPLES:--1}}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${PPO_MINI_BATCH_SIZE}}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"

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
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/verl:${PYTHONPATH:-}"
export TRANSFER_QUEUE_ENABLE="${TRANSFER_QUEUE_ENABLE:-}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"

echo "=== SWE-Agent Blackbox Megatron Async Training (single-node) ==="
echo "Model:       ${MODEL_PATH}"
echo "Train data:  ${TRAIN_DATA}"
echo "Val data:    ${VAL_DATA}"
echo "Engine:      ${ENGINE} (gen_tp=${GEN_TP}, train_tp=${TRAIN_TP})"
echo "Runner:      ${RUNNER}"
echo "Turns:       agent_max_turns=${AGENT_MAX_TURNS}"
echo "Batch:       n=${N}, mini_bsz=${PPO_MINI_BATCH_SIZE}"
echo "Sequence:    prompt=${PROMPT_LENGTH}, response=${RESPONSE_LENGTH}"
echo "Trainer:     V1 ${TRAINER_MODE}"
if [[ "${USE_LORA}" == "true" ]]; then
    echo "LoRA:        enabled (rank=${LORA_RANK}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT})"
    echo "LoRA targets: ${LORA_TARGET_MODULES}"
else
    echo "LoRA:        disabled (full-parameter)"
fi
if [[ "${TRAINER_MODE}" == "separate_async" ]]; then
    echo "Resources:   trainer=${NNODES}x${N_GPUS_PER_NODE}, rollout=${ROLLOUT_NNODES}x${ROLLOUT_NGPUS_PER_NODE}"
else
    echo "Resources:   colocated=${NNODES}x${N_GPUS_PER_NODE}"
fi
echo "Samples:     train_max=${TRAIN_MAX_SAMPLES}, val_max=${VAL_MAX_SAMPLES}"
echo "==================================================================="

# ── Compute derived parameters ───────────────────────────────────────────
ACTOR_PPO_MAX_TOKEN_LEN=$(( (PROMPT_LENGTH + RESPONSE_LENGTH) / TRAIN_CP ))
INFER_PPO_MAX_TOKEN_LEN=$(( (PROMPT_LENGTH + RESPONSE_LENGTH) / TRAIN_CP ))

# ── Launch (directly on the current node, no Ray job submit) ────────────
python3 -m verl.trainer.main_ppo \
    --config-name="${CONFIG_NAME}" \
    --config-path="${REPO_ROOT}/examples/blackbox_recipes/mini_swe_agent/config" \
    hydra.searchpath=[pkg://verl.trainer.config] \
    trainer.use_v1=True \
    trainer.v1.trainer_mode="${TRAINER_MODE}" \
    trainer.v1.colocate_async.num_warmup_batches=${NUM_WARMUP_BATCHES} \
    trainer.v1.separate_async.num_warmup_batches=${SEPARATE_NUM_WARMUP_BATCHES} \
    trainer.v1.separate_async.parameter_sync_step=${PARAMETER_SYNC_STEP} \
    transfer_queue.enable=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    data.train_files="['${TRAIN_DATA}']" \
    data.val_files="['${VAL_DATA}']" \
    data.train_max_samples=${TRAIN_MAX_SAMPLES} \
    data.val_max_samples=${VAL_MAX_SAMPLES} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.max_prompt_length=${PROMPT_LENGTH} \
    data.max_response_length=${RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.n=${N} \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.prompt_length=${PROMPT_LENGTH} \
    actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.top_p=${TOP_P} \
    actor_rollout_ref.rollout.top_k=${TOP_K} \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${UPDATE_WEIGHTS_BUCKET_MB} \
    actor_rollout_ref.rollout.nnodes=${ROLLOUT_NNODES} \
    actor_rollout_ref.rollout.n_gpus_per_node=${ROLLOUT_NGPUS_PER_NODE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY"' \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_cache_mode=align' \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_cpu_binding=true' \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=true' \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.agent.num_workers=${NUM_AGENT_WORKERS} \
    "${RUNNER_ARGS[@]}" \
    ${LORA_ARGS[@]+"${LORA_ARGS[@]}"} \
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN} \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.actor.megatron.param_offload=${OFFLOAD} \
    actor_rollout_ref.actor.megatron.grad_offload=${OFFLOAD} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${OFFLOAD} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TRAIN_TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${TRAIN_PP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${TRAIN_CP} \
    actor_rollout_ref.actor.megatron.use_mbridge=${USE_MBRIDGE} \
    actor_rollout_ref.ref.megatron.param_offload=${OFFLOAD} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TRAIN_TP} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${TRAIN_PP} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${TRAIN_CP} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${INFER_PPO_MAX_TOKEN_LEN} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${INFER_PPO_MAX_TOKEN_LEN} \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.default-local-dir="${CKPTS_DIR}" \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    ${TOTAL_TRAINING_STEPS:+trainer.total_training_steps=${TOTAL_TRAINING_STEPS}} \
    "$@"