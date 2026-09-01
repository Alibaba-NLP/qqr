#!/bin/bash

# =============================================================================
# Start the Travel executor agent inference service
#
# This service must be running before training the Travel curriculum agent
# The executor agent is a frozen model that answers using the AMap/Transport/WebSearch tools
#
# Default: this service uses GPU 0 and training uses GPUs 1-7
# =============================================================================

set -ex

# Executor model path
EXECUTOR_MODEL_PATH=${EXECUTOR_MODEL_PATH:-"Qwen/Qwen3.5-9B"}

# Service port
EXECUTOR_PORT=${EXECUTOR_PORT:-30000}

# GPU settings
# Defaults to GPU 0; the training script uses GPUs 1-7
EXECUTOR_GPUS=${EXECUTOR_GPUS:-"0"}

# Tensor-parallel size (number of GPUs)
# 1 for a single GPU; set to the GPU count for multi-GPU
EXECUTOR_TP_SIZE=${EXECUTOR_TP_SIZE:-1}

echo "======================================"
echo "Starting the Travel executor agent inference service"
echo "Model path: ${EXECUTOR_MODEL_PATH}"
echo "Port: ${EXECUTOR_PORT}"
echo "GPU: ${EXECUTOR_GPUS}"
echo "TP Size: ${EXECUTOR_TP_SIZE}"
echo "======================================"
echo ""
echo "The Travel executor will use the following tools:"
echo "  - AMap: POI search and route planning"
echo "  - Transport: transit lookup"
echo "  - WebSearch: web search"
echo "======================================"

# Start the inference service with SGLang
# Important: --tool-call-parser qwen is required for function calling
CUDA_VISIBLE_DEVICES=${EXECUTOR_GPUS} python -m sglang.launch_server \
    --model-path ${EXECUTOR_MODEL_PATH} \
    --port ${EXECUTOR_PORT} \
    --host 0.0.0.0 \
    --tensor-parallel-size ${EXECUTOR_TP_SIZE} \
    --mem-fraction-static 0.85 \
    --chunked-prefill-size 8192 \
    --max-running-requests 32 \
    --trust-remote-code \
    --tool-call-parser qwen

# or use vLLM:
# CUDA_VISIBLE_DEVICES=${EXECUTOR_GPUS} python -m vllm.entrypoints.openai.api_server \
#     --model ${EXECUTOR_MODEL_PATH} \
#     --port ${EXECUTOR_PORT} \
#     --host 0.0.0.0 \
#     --tensor-parallel-size ${EXECUTOR_TP_SIZE} \
#     --trust-remote-code \
#     --max-model-len 8192
