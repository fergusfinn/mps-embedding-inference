#!/bin/bash
set -euo pipefail

# Direct throughput comparison: offline vLLM engine at high load
# under gen_only, mps_active, and ts_active conditions.
#
# Uses throughput_probe.py (512 prompts x 512 tokens) instead of
# the HTTP server to isolate GPU-level effects from server overhead.

VLLM_IMAGE="vllm/vllm-openai:v0.15.1"
HF_CACHE="/home/ubuntu/.cache/huggingface"
VLLM_CACHE="/home/ubuntu/.cache/vllm"
EMBED_MODEL="Qwen/Qwen3-Embedding-8B"
DIR="$(cd "$(dirname "$0")" && pwd)"

full_cleanup() {
    docker stop gen-probe vllm-embed mps-daemon 2>/dev/null || true
    docker rm gen-probe vllm-embed mps-daemon 2>/dev/null || true
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

run_gen_probe() {
    local use_mps=$1
    local extra_args=""
    if [ "$use_mps" = "yes" ]; then
        extra_args="-v mps-pipe:/mps -e CUDA_MPS_PIPE_DIRECTORY=/mps"
    fi
    docker run --rm --name gen-probe \
        --gpus all --network host --ipc=host \
        -v "$HF_CACHE:/root/.cache/huggingface" \
        -v "$VLLM_CACHE:/root/.cache/vllm" \
        -v "$DIR:/workspace" \
        $extra_args \
        -e HF_HUB_OFFLINE=1 \
        --entrypoint python3 \
        "$VLLM_IMAGE" \
        /workspace/throughput_probe.py 2>&1 | grep -E "Throughput:|Generated"
}

echo "=== Offline throughput A/B test ==="

# --- 1. gen_only ---
echo ""
echo "--- 1. gen_only ---"
full_cleanup
sleep 2
echo "  Running throughput probe..."
run_gen_probe "no"

# --- 2. mps_active ---
echo ""
echo "--- 2. mps_active ---"
full_cleanup
sleep 2
docker volume create mps-pipe > /dev/null 2>&1 || true
docker run --rm -d --name mps-daemon \
    --gpus all --ipc=host --entrypoint bash \
    -v mps-pipe:/mps "$VLLM_IMAGE" \
    -c 'CUDA_MPS_PIPE_DIRECTORY=/mps nvidia-cuda-mps-control -d && sleep infinity' \
    > /dev/null 2>&1
sleep 2
start_embed "yes"
# Start embed load in background
python3 "$DIR/embed_benchmark_vllm.py" --base-url http://localhost:8200 \
    --duration 300 --concurrency 64 --batch-size 8 --model "$EMBED_MODEL" > /dev/null 2>&1 &
EMBED_PID=$!
sleep 5
echo "  Running throughput probe with active embed (MPS)..."
run_gen_probe "yes"
kill $EMBED_PID 2>/dev/null || true

# --- 3. ts_active ---
echo ""
echo "--- 3. ts_active ---"
full_cleanup
sleep 2
start_embed "no"
python3 "$DIR/embed_benchmark_vllm.py" --base-url http://localhost:8200 \
    --duration 300 --concurrency 64 --batch-size 8 --model "$EMBED_MODEL" > /dev/null 2>&1 &
EMBED_PID=$!
sleep 5
echo "  Running throughput probe with active embed (time-sliced)..."
run_gen_probe "no"
kill $EMBED_PID 2>/dev/null || true

echo ""
echo "=== Done ==="
