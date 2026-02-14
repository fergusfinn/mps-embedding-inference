#!/bin/bash
set -euo pipefail

# Test what causes the MPS overhead by isolating variables:
# 1. gen_only:       gen alone, no MPS
# 2. mps_gen_only:   gen alone, MPS daemon running (no embed server)
# 3. mps_idle_embed: gen + embed server loaded but idle, MPS
# 4. mps_active:     gen + embed concurrent, MPS
# 5. ts_active:      gen + embed concurrent, time-sliced (no MPS)

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
    echo "  warmup 30s..."
    python3 "$DIR/gen_benchmark.py" --base-url http://localhost:8100 \
        --duration 30 --concurrency 512 --max-tokens 512 --model "$GEN_MODEL" > /dev/null 2>&1

    echo "  measuring gen 60s..."
    local result
    result=$(python3 "$DIR/gen_benchmark.py" --base-url http://localhost:8100 \
        --duration 60 --concurrency 512 --max-tokens 512 --model "$GEN_MODEL" 2>&1 | grep "Throughput:")
    echo "  $label: $result"
}

measure_gen_with_embed() {
    local label=$1
    echo "  warmup 30s (gen + embed concurrent)..."
    python3 "$DIR/gen_benchmark.py" --base-url http://localhost:8100 \
        --duration 30 --concurrency 512 --max-tokens 512 --model "$GEN_MODEL" > /dev/null 2>&1 &
    local WARM_GEN=$!
    python3 "$DIR/embed_benchmark_vllm.py" --base-url http://localhost:8200 \
        --duration 30 --concurrency 64 --batch-size 8 --model "$EMBED_MODEL" > /dev/null 2>&1 &
    local WARM_EMBED=$!
    wait $WARM_GEN || true
    wait $WARM_EMBED || true

    echo "  measuring 60s (gen + embed concurrent)..."
    python3 "$DIR/embed_benchmark_vllm.py" --base-url http://localhost:8200 \
        --duration 65 --concurrency 64 --batch-size 8 --model "$EMBED_MODEL" > /dev/null 2>&1 &
    local EMBED_PID=$!

    local gen_result
    gen_result=$(python3 "$DIR/gen_benchmark.py" --base-url http://localhost:8100 \
        --duration 60 --concurrency 512 --max-tokens 512 --model "$GEN_MODEL" 2>&1 | grep "Throughput:")

    wait $EMBED_PID || true
    echo "  $label: $gen_result"
}

echo "=== Isolation test: what causes MPS overhead? ==="
echo ""

# 1. Gen only, no MPS
echo "--- 1. gen_only (no MPS, no embed) ---"
full_cleanup
sleep 2
start_gen "no"
measure_gen "gen_only"
GEN_ONLY_DONE=1
echo ""

# 2. Gen only, MPS daemon running (no embed server)
echo "--- 2. mps_gen_only (MPS daemon, no embed server) ---"
full_cleanup
sleep 2
start_mps
start_gen "yes"
measure_gen "mps_gen_only"
echo ""

# 3. Gen + embed loaded but idle, MPS
echo "--- 3. mps_idle_embed (MPS, embed server loaded but no traffic) ---"
full_cleanup
sleep 2
start_mps
start_gen "yes"
start_embed "yes"
measure_gen "mps_idle_embed"
echo ""

# 4. Gen + embed concurrent, MPS
echo "--- 4. mps_active (MPS, gen + embed concurrent) ---"
full_cleanup
sleep 2
start_mps
start_gen "yes"
start_embed "yes"
measure_gen_with_embed "mps_active"
echo ""

# 5. Gen + embed concurrent, time-sliced
echo "--- 5. ts_active (time-sliced, gen + embed concurrent) ---"
full_cleanup
sleep 2
start_gen "no"
start_embed "no"
measure_gen_with_embed "ts_active"
echo ""

echo "=== Done ==="
