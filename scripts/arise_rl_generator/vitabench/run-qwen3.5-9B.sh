#!/bin/bash
# Usage: bash scripts/arise_rl_generator/vitabench/run-qwen3.5-9B.sh 2>&1 | tee vita_generator_$(date +%m%d).log
# =============================================================================
# VitaBench curriculum agent (generator) training script - Qwen3.5-9B + RG-KL
#
# Core idea:
# - The generator (Qwen3.5-9B) explores the 100 official OTA environments through the real VitaBenchToolState tools
# - Generates instructions + rubrics (the environment is taken directly from the official data)
# - The solver (external sglang service) validates task difficulty in interactive mode
# - UserSimulator (local Qwen3.5-397B-A17B) drives the multi-turn user interaction
#
# RG-KL algorithm (identical to the solver in arise_rl_solver.vitabench):
#   Phase A: k_s student rollouts without coach memory -> generate tasks -> solver validation -> scoring
#   Phase B: the gpt-5.2 coach reviews student task quality -> produces improvement advice (memory)
#   Phase C: k_m teacher rollouts with coach memory injected, used only to estimate Δ_r
#   Training:
#     - Student guided_tokens = [p_m_prompt, τ_s_response]
#     - L = L_GRPO_student + λ(Δ_r) · D_KL + β · KL_ref_penalty
#     - λ(Δ_r) = λ_0 · gate(Δ_r) · warmup · cosine_decay
# =============================================================================

# =============================================================================
# Clean up any previous training processes
# Note: do not `pkill -9 sglang`; it would kill the external executor server (the solver on GPU 0)
# Only clean up ray and train.py; the sglang training engine is a ray actor and is torn down by ray stop
# =============================================================================
ray stop --force 2>/dev/null || true
sleep 3
pkill -9 -f "ray::" 2>/dev/null || true
sleep 2
pkill -9 -f "train\.py" 2>/dev/null || true
pkill -9 -f "slime" 2>/dev/null || true
sleep 3
pkill -9 -f "ray::" 2>/dev/null || true
pkill -9 -f "train\.py" 2>/dev/null || true
rm -rf /tmp/ray/*

set -ex

# =============================================================================
# Network settings
# =============================================================================

# =============================================================================
# Environment settings
# =============================================================================

RUN_NAME=${RUN_NAME:-Qwen3.5-9B-ARISE-Generator-VitaBench}
CKPT_DIR=${RUN_NAME}

QQR_PATH=$(pip list | grep qqr | awk '{print $NF}')
SLIME_PATH=$(pip list | grep slime | awk '{print $NF}')
MEGATRON_LM_PATH=$(pip list | grep megatron-core | awk '{print $NF}')

if [ -z "${QQR_PATH}" ]; then
  echo "QQR_PATH is not set"
  exit 1
fi

if [ -z "${SLIME_PATH}" ]; then
  echo "SLIME_PATH is not set"
  exit 1
fi

cd ${SLIME_PATH}

# =============================================================================
# GPU settings
# GPU 0 serves the executor (solver); GPUs 1-7 train the generator
# =============================================================================
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"1,2,3,4,5,6"}   # Drop GPU 7 and use 6 GPUs at TP=2 to halve the model state

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23456}
export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export NPROC_PER_NODE=${NPROC_PER_NODE:-6}
export NNODES=${WORLD_SIZE}
export NODE_RANK=${RANK}

export PYTHONBUFFERED=16
export PYTHONPATH=${QQR_PATH}:${SLIME_PATH}:${MEGATRON_LM_PATH}:${PYTHONPATH}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)

if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi

echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_NVLS_ENABLE=${HAS_NVLINK}
export NCCL_DEBUG=WARN
export TORCH_CUDA_ARCH_LIST="9.0;9.0a"
export TORCH_USE_CUDA_DSA=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=OFF
export RAY_NUM_SERVER_CALL_THREAD=1
export TOKENIZERS_PARALLELISM=false
export no_proxy="127.0.0.1,${MASTER_ADDR}"

# =============================================================================
# Ray memory management settings (avoids NAS and GPU memory fragmentation issues)
# =============================================================================
export RAY_memory_monitor_refresh_ms=0
export RAY_object_spilling_config='{"type":"filesystem","params":{"directory_path":"/tmp/ray_spill"}}'
mkdir -p /tmp/ray_spill
export RAY_object_store_memory=$((200 * 1024 * 1024 * 1024))
export RAY_BACKEND_LOG_LEVEL=warning
export RAY_PLASMA_STORE_MEMORY_FRACTION=0.3

# Measured: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is incompatible with torch_memory_saver
# (SGLangEngine raises a "TorchMemorySaver disabled" RuntimeError on startup). Other OOM mitigations are used instead:
# (1) --max-tokens-per-gpu 2048 halves training activation; (2) --sglang-mem-fraction 0.4 leaves room for Megatron

adjust_oom_score() {
    echo -500 > /proc/self/oom_score_adj 2>/dev/null || true
}
adjust_oom_score

# =============================================================================
# VitaBench settings
# =============================================================================

# scenario: ota (online travel) / delivery / instore / cross_domain
# the generator and solver prompts switch to the matching scenario automatically based on this value
export VITABENCH_DOMAIN=${VITABENCH_DOMAIN:-ota}
export VITABENCH_DATA_DIR=${VITABENCH_DATA_DIR:?"VITABENCH_DATA_DIR is required (the data/vita directory of the official VitaBench repo)"}

# =============================================================================
# Executor agent settings (solver validation service; must be started beforehand)
# =============================================================================

export EXECUTOR_API_BASE=${EXECUTOR_API_BASE:-"http://localhost:30000/v1"}
export EXECUTOR_MODEL=${EXECUTOR_MODEL:-"default"}
export EXECUTOR_NUM_TRIALS=${EXECUTOR_NUM_TRIALS:-8}   # K=8, as in the paper
export EXECUTOR_SUCCESS_THRESHOLD=${EXECUTOR_SUCCESS_THRESHOLD:-0.9}  # γ=0.9, as in the paper
export EXECUTOR_MAX_STEPS=${EXECUTOR_MAX_STEPS:-40}    # the solver gets at most 40 rounds of tool interaction, as in the paper
export EXECUTOR_CONCURRENCY_LIMIT=${EXECUTOR_CONCURRENCY_LIMIT:-8}

# UserSimulator (locally deployed Qwen3.5-397B-A17B)
export USER_SIMULATOR_API_KEY="EMPTY"
export USER_SIMULATOR_BASE_URL=${USER_SIMULATOR_BASE_URL:?"USER_SIMULATOR_BASE_URL is required (Qwen3.5-397B service endpoint)"}
export USER_SIMULATOR_MODEL="Qwen3.5-397B-A17B"
export USER_SIMULATOR_TEMPERATURE="0.0"

# Coach (gpt-5.2) produces the generator coaching memory
export COACH_MODEL="gpt-5.2-2025-12-11"
export COACH_CONCURRENCY_LIMIT="10"

# Sliding-window judge (local Qwen3.5-397B)
export LLM_JUDGE_CONCURRENCY_LIMIT="16"

# =============================================================================
# Reward-Gated Reverse KL (RG-KL) settings
# =============================================================================

export ENABLE_RG_KL=true
export K_STUDENT=8                          # number of student rollouts (these are trained on)
export K_TEACHER=8                          # number of teacher rollouts (used only for Δ_r); G = 8 + 8 = 16, as in the paper
export RG_KL_LAMBDA_0=0.5                   # λ_0=0.5, as in the paper
export RG_KL_DELTA_THRESHOLD=0.05           # τ=0.05, as in the paper
export RG_KL_GATE_TEMPERATURE=0.0125        # sigmoid gate temperature T
export RG_KL_WARMUP_ITERS=20                # cosine warm-up
export RG_KL_DECAY_ITERS=120                # cosine decay length
export RG_KL_MIN_LAMBDA_RATIO=0.1          # decays to λ_0 · 0.1 at the end
export ENABLE_GUIDED_TOKENS_FOR_STUDENT=true

# =============================================================================
# Model settings (Qwen3.5-9B)
# =============================================================================

source "${QQR_PATH}/scripts/models/qwen3.5-9B.sh"

CKPT_ARGS=(
   --hf-checkpoint Qwen/Qwen3.5-9B
   --ref-load /path/to/Qwen3.5-9B_torch_dist
   --load ${CKPT_DIR}
   --save ${CKPT_DIR}
   --save-interval 10                       # Kept at 10. The save bug always hits on the second save regardless of interval, and 10 lets each run go further
   --normalize-advantages
)

# =============================================================================
# Rollout settings
# =============================================================================

ROLLOUT_ARGS=(
   --rollout-function-path qqr.rollout.agent_rollout.generate_rollout
   --prompt-data ${TRAIN_DATA:?"TRAIN_DATA is required (task seed jsonl: one {\"query\": ..., \"metadata\": {\"task_id\": <official env id>, \"domain\": ...}} per line)"}
   --input-key query
   --rollout-shuffle
   --num-rollout 200
   --rollout-batch-size 2
   --n-samples-per-prompt 16              # G=16 (8 student + 8 teacher), as in the paper
   --rollout-max-context-len 32000
   --rollout-max-response-len 6144         # At 4096, 92.5% of outputs were truncated and rewards collapsed to 0. TP=2 frees enough memory to raise this to 6144 so the instructions JSON fits
   --rollout-temperature 0.8

   --global-batch-size 32                 # rollout_batch_size × 16
   --balance-data
   --group-rm
)

# =============================================================================
# Evaluation settings
# =============================================================================

EVAL_ARGS=(
   # Evaluation is disabled entirely: it ran the default AIME math set rather than the curriculum tasks,
   # and it consumed a lot of training time. Leaving the array empty passes no --eval-* flags and disables evaluation.
   --skip-eval-before-train
)

# =============================================================================
# Performance settings
# =============================================================================

PERF_ARGS=(
   --tensor-model-parallel-size 2          # 6 GPUs with TP=2, DP=3 halves params/grads and resolves the 9B model OOM seen at TP=1
   --sequence-parallel                     # Enable sequence parallelism when TP>1 to further reduce activation memory
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 1024               # 2048 still OOMs, but only by about 0.6GB, so halve it again
)

# =============================================================================
# RG-KL GRPO settings
# =============================================================================

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-rg-kl                             # Enable Reward-Gated Reverse KL
   --rg-kl-clip-value 10.0                 # per-token KL clipping threshold
   --use-kl-loss
   --kl-loss-coef 0.001                    # Reference-KL safety anchor
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.2
   --clip-grad 100000
)

# =============================================================================
# Optimizer settings
# =============================================================================

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --min-lr 1e-7
   --lr-decay-style cosine
   --lr-decay-iters 480
   --lr-warmup-iters 0
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --override-opt_param-scheduler
)

# =============================================================================
# Logging settings
# =============================================================================


SWANLAB_ARGS=(
   # --use-swanlab
   # --swanlab-project QQR-VitaBenchCurriculum-RG-KL
   # --swanlab-group ${RUN_NAME}
   # --swanlab-mode cloud
)

# =============================================================================
# SGLang inference engine settings
# =============================================================================

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1         # 7 GPUs are not divisible by 2, so use one GPU per engine (7 engines total)
   --sglang-mem-fraction-static 0.4        # OOM mitigation: 0.5 -> 0.4 to leave more memory for Megatron training
   --sglang-server-concurrency 128
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# =============================================================================
# Custom function paths (pointing at the arise_rl_generator.vitabench module)
# =============================================================================

CUSTOM_ARGS=(
   --custom-generate-function-path qqr.examples.arise_rl_generator.vitabench.generate
   --custom-rm-path qqr.examples.arise_rl_generator.vitabench.group_reward
   --custom-reward-post-process-path qqr.examples.arise_rl_generator.vitabench.reward_post_process
)

# =============================================================================
# API Keys
# =============================================================================

export DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY:?"DASHSCOPE_API_KEY is required"}
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

# =============================================================================
# Launch training
# =============================================================================

if [ $RANK -eq 0 ]; then

echo "Checking the executor solver service..."
if curl -s --max-time 5 ${EXECUTOR_API_BASE}/models > /dev/null 2>&1; then
    echo "Executor service is ready: ${EXECUTOR_API_BASE}"
else
    echo "Warning: the executor service did not respond (${EXECUTOR_API_BASE})"
    echo "  Make sure the executor service is running:"
    echo "  bash scripts/arise_rl_generator/vitabench/start_executor_server.sh"
    echo "  Waiting 10s before continuing..."
    sleep 10
fi

ray start --head --port 6379 --disable-usage-stats --metrics-export-port 8080

mkdir -p ${CKPT_DIR}/debug_generator_vitabench

echo "======================================"
echo "VitaBench curriculum RG-KL training configuration:"
echo "  Model: Qwen3.5-9B"
echo "  Algorithm: Reward-Gated Reverse KL"
echo "  VitaBench domain: ${VITABENCH_DOMAIN}"
echo "--------------------------------------"
echo "  Generator: real VitaBenchToolState"
echo "  Solver: executor (${EXECUTOR_API_BASE})"
echo "  UserSimulator: ${USER_SIMULATOR_MODEL} (${USER_SIMULATOR_BASE_URL})"
echo "  Coach: ${COACH_MODEL}"
echo "--------------------------------------"
echo "  RG-KL: k_s=${K_STUDENT} k_m=${K_TEACHER} λ_0=${RG_KL_LAMBDA_0}"
echo "======================================"

python3 train.py \
   --actor-num-nodes ${NNODES} \
   --actor-num-gpus-per-node ${NPROC_PER_NODE} \
   --rollout-num-gpus ${NPROC_PER_NODE} \
   --colocate \
   --save-rollout-data \
   --save-debug-rollout-data "${CKPT_DIR}/debug_generator_vitabench/rollout_{rollout_id}.pt" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${SWANLAB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   2>&1 | tee ${CKPT_DIR}/train.log

sleep 30m

else

echo "Starting Ray worker to ${MASTER_ADDR}"
ray start --block --address ${MASTER_ADDR}:6379 --disable-usage-stats --metrics-export-port 8080

fi
