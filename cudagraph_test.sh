#!/bin/bash
set -euo pipefail

# Test: does CUDA graph mode cause the MPS overhead?
#
# The server A/B test (CUDA graphs enabled by default) shows a ~5% MPS gap.
# The offline probes (enforce_eager=True) show zero gap.
#
# This test runs 4 server conditions with --enforce-eager to control for
# CUDA graph effects:
#   1. eager_gen_only:       gen server, enforce-eager, no MPS, no embed
#   2. eager_mps_active:     gen + embed servers, enforce-eager, MPS, concurrent
#   3. eager_ts_active:      gen + embed servers, enforce-eager, time-sliced, concurrent
#   4. graphs_mps_active:    gen + embed servers, CUDA graphs (default), MPS, concurrent

VLLM_IMAGE="vllm/vllm-openai:v0.15.1"
HF_CACHE="/home/ubuntu/.cache/huggingface"
VLLM_CACHE="/home/ubuntu/.cache/vllm"
GEN_MODEL="Qwen/Qwen3-30B-A3B-FP8"
EMBED_MODEL="Qwen/Qwen3-Embedding-8B"
DIR="$(cd "$(dirname "$0")" && pwd)"

source "$DIR/venv/bin/activate"

full_cleanup() {
    docker stop vllm-gen vllm-embed mps-daemon 2>/dev/null || true
    docker rm vllm-gen vllm-embed mps-daemon 2>/dev/null || true
    docker volume rm mps-pipe 2>/dev/null || true
    pkill -f gen_benchmark || true
    pkill -f embed_benchmark_vllm || true
}
trap full_cleanup EXIT

wait_health() {
    local port=$1 label=$2
    for i in $(seq 1 180); do
        curl -s "http://localhost:$port/health" > /dev/null 2>&1 && echo "  $label ready after ${i}s" && return
        sleep 1
    done
    echo "  $label FAILED to start" && exit 1
}

start_mps() {
    docker volume create mps-pipe > /dev/null 2>&1 || true
    docker run --rm -d --name mps-daemon \
        --gpus all --ipc=host --entrypoint bash \
        -v mps-pipe:/mps "$VLLM_IMAGE" \
        -c 'CUDA_MPS_PIPE_DIRECTORY=/mps nvidia-cuda-mps-control -d && sleep infinity' \
        > /dev/null 2>&1
    sleep 2
}

start_gen() {
    local use_mps=$1
    local enforce_eager=$2
    local extra_args=""
    local gen_extra=""
    if [ "$use_mps" = "yes" ]; then
        extra_args="-v mps-pipe:/mps -e CUDA_MPS_PIPE_DIRECTORY=/mps"
    fi
    if [ "$enforce_eager" = "yes" ]; then
        gen_extra="--enforce-eager"
    fi
    docker run --rm -d --name vllm-gen \
        --gpus all --network host --ipc=host \
        -v "$HF_CACHE:/root/.cache/huggingface" \
        -v "$VLLM_CACHE:/root/.cache/vllm" \
        $extra_args \
        -e HF_HUB_OFFLINE=1 \
        "$VLLM_IMAGE" "$GEN_MODEL" \
        --port 8100 --gpu-memory-utilization 0.50 \
        --max-model-len 4096 --max-num-seqs 512 \
        --disable-log-requests $gen_extra > /dev/null 2>&1
    wait_health 8100 "Gen"
}

start_embed() {
    local use_mps=$1
    local extra_args=""
    if [ "$use_mps" = "yes" ]; then
        extra_args="-v mps-pipe:/mps -e CUDA_MPS_PIPE_DIRECTORY=/mps"
    fi
    docker run --rm -d --name vllm-embed \
        --gpus all --network host --ipc=host \
        -v "$HF_CACHE:/root/.cache/huggingface" \
        -v "$VLLM_CACHE:/root/.cache/vllm" \
        $extra_args \
        -e HF_HUB_OFFLINE=1 \
        "$VLLM_IMAGE" "$EMBED_MODEL" \
        --port 8200 --gpu-memory-utilization 0.30 \
        --max-model-len 512 --convert embed \
        --disable-log-requests > /dev/null 2>&1
    wait_health 8200 "Embed"
}

measure_gen_with_embed() {
    local label=$1
    echo "  warmup 20s (concurrent)..."
    python3 "$DIR/gen_benchmark.py" --base-url http://localhost:8100 \
        --duration 20 --concurrency 512 --max-tokens 512 --model "$GEN_MODEL" > /dev/null 2>&1 &
    python3 "$DIR/embed_benchmark_vllm.py" --base-url http://localhost:8200 \
        --duration 20 --concurrency 64 --batch-size 8 --model "$EMBED_MODEL" > /dev/null 2>&1 &
    wait

    echo "  measuring 30s (concurrent)..."
    python3 "$DIR/embed_benchmark_vllm.py" --base-url http://localhost:8200 \
        --duration 35 --concurrency 64 --batch-size 8 --model "$EMBED_MODEL" > /dev/null 2>&1 &
    local EMBED_PID=$!

    local result
    result=$(python3 "$DIR/gen_benchmark.py" --base-url http://localhost:8100 \
        --duration 30 --concurrency 512 --max-tokens 512 --model "$GEN_MODEL" 2>&1)
    echo "  $label: $(echo "$result" | grep "Throughput:")"
    echo "  $label: $(echo "$result" | grep "Decode rate")"
    wait $EMBED_PID || true
}

measure_gen() {
    local label=$1
    echo "  warmup 20s..."
    python3 "$DIR/gen_benchmark.py" --base-url http://localhost:8100 \
        --duration 20 --concurrency 512 --max-tokens 512 --model "$GEN_MODEL" > /dev/null 2>&1

    echo "  measuring gen 30s..."
    local result
    result=$(python3 "$DIR/gen_benchmark.py" --base-url http://localhost:8100 \
        --duration 30 --concurrency 512 --max-tokens 512 --model "$GEN_MODEL" 2>&1)
    echo "  $label: $(echo "$result" | grep "Throughput:")"
    echo "  $label: $(echo "$result" | grep "Decode rate")"
}

echo "=== CUDA graph test: does enforce-eager eliminate MPS gap? ==="

# 1. Eager gen-only baseline
echo ""
echo "--- 1. eager_gen_only ---"
full_cleanup; sleep 2
start_gen "no" "yes"
measure_gen "eager_gen_only"

# 2. Eager MPS active
echo ""
echo "--- 2. eager_mps_active ---"
full_cleanup; sleep 2
start_mps
start_gen "yes" "yes"
start_embed "yes"
measure_gen_with_embed "eager_mps_active"

# 3. Eager time-sliced active
echo ""
echo "--- 3. eager_ts_active ---"
full_cleanup; sleep 2
start_gen "no" "yes"
start_embed "no"
measure_gen_with_embed "eager_ts_active"

# 4. CUDA graphs MPS active (this is the condition that showed 5% gap)
echo ""
echo "--- 4. graphs_mps_active ---"
full_cleanup; sleep 2
start_mps
start_gen "yes" "no"
start_embed "yes"
measure_gen_with_embed "graphs_mps_active"

# 5. CUDA graphs time-sliced active (for comparison)
echo ""
echo "--- 5. graphs_ts_active ---"
full_cleanup; sleep 2
start_gen "no" "no"
start_embed "no"
measure_gen_with_embed "graphs_ts_active"

echo ""
echo "=== Done ==="
