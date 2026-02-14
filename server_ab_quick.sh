#!/bin/bash
set -euo pipefail

# Quick server A/B: 4 conditions, 1 rep each
# 1. gen_server_only:      gen server, no MPS, no embed
# 2. mps_gen_server_only:  gen server, MPS daemon, no embed
# 3. mps_server_active:    gen + embed servers, MPS, concurrent load
# 4. ts_server_active:     gen + embed servers, time-sliced, concurrent load

VLLM_IMAGE="vllm/vllm-openai:v0.15.1"
HF_CACHE="/home/ubuntu/.cache/huggingface"
VLLM_CACHE="/home/ubuntu/.cache/vllm"
GEN_MODEL="Qwen/Qwen3-30B-A3B-FP8"
EMBED_MODEL="Qwen/Qwen3-Embedding-8B"
DIR="$(cd "$(dirname "$0")" && pwd)"

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
    for i in $(seq 1 120); do
        curl -s "http://localhost:$port/health" > /dev/null 2>&1 && echo "  $label ready after ${i}s" && return
        sleep 1
    done
    echo "  $label FAILED to start" && exit 1
}

start_gen() {
    local use_mps=$1
    local extra_args=""
    if [ "$use_mps" = "yes" ]; then
        extra_args="-v mps-pipe:/mps -e CUDA_MPS_PIPE_DIRECTORY=/mps"
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
        --disable-log-requests > /dev/null 2>&1
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

measure_gen() {
    local label=$1
    echo "  warmup 20s..."
    python3 "$DIR/gen_benchmark.py" --base-url http://localhost:8100 \
        --duration 20 --concurrency 512 --max-tokens 512 --model "$GEN_MODEL" > /dev/null 2>&1

    echo "  measuring gen 30s..."
    local result
    result=$(python3 "$DIR/gen_benchmark.py" --base-url http://localhost:8100 \
        --duration 30 --concurrency 512 --max-tokens 512 --model "$GEN_MODEL" 2>&1)
    local throughput
    throughput=$(echo "$result" | grep "Throughput:")
    local latency
    latency=$(echo "$result" | grep "Decode rate")
    echo "  $label: $throughput"
    echo "  $label: $latency"
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
    local throughput
    throughput=$(echo "$result" | grep "Throughput:")
    local latency
    latency=$(echo "$result" | grep "Decode rate")
    echo "  $label: $throughput"
    echo "  $label: $latency"

    wait $EMBED_PID || true
}

# Activate venv for benchmark scripts
source "$DIR/venv/bin/activate"

echo "=== Server A/B quick test ==="

# 1. gen server only (no MPS)
echo ""
echo "--- 1. gen_server_only ---"
full_cleanup; sleep 2
start_gen "no"
measure_gen "gen_server_only"

# 2. gen server + MPS daemon (no embed)
echo ""
echo "--- 2. mps_gen_server_only ---"
full_cleanup; sleep 2
docker volume create mps-pipe > /dev/null 2>&1 || true
docker run --rm -d --name mps-daemon \
    --gpus all --ipc=host --entrypoint bash \
    -v mps-pipe:/mps "$VLLM_IMAGE" \
    -c 'CUDA_MPS_PIPE_DIRECTORY=/mps nvidia-cuda-mps-control -d && sleep infinity' \
    > /dev/null 2>&1
sleep 2
start_gen "yes"
measure_gen "mps_gen_server_only"

# 3. gen + embed, MPS, concurrent
echo ""
echo "--- 3. mps_server_active ---"
full_cleanup; sleep 2
docker volume create mps-pipe > /dev/null 2>&1 || true
docker run --rm -d --name mps-daemon \
    --gpus all --ipc=host --entrypoint bash \
    -v mps-pipe:/mps "$VLLM_IMAGE" \
    -c 'CUDA_MPS_PIPE_DIRECTORY=/mps nvidia-cuda-mps-control -d && sleep infinity' \
    > /dev/null 2>&1
sleep 2
start_gen "yes"
start_embed "yes"
measure_gen_with_embed "mps_server_active"

# 4. gen + embed, time-sliced, concurrent
echo ""
echo "--- 4. ts_server_active ---"
full_cleanup; sleep 2
start_gen "no"
start_embed "no"
measure_gen_with_embed "ts_server_active"

echo ""
echo "=== Done ==="
