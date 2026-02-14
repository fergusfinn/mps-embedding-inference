#!/bin/bash
set -euo pipefail

# nsys traces for gen decode under 3 conditions:
#   1. gen_only:   no MPS, no embed (baseline)
#   2. mps_active: MPS, embed server running with active load
#   3. ts_active:  time-sliced, embed server running with active load
#
# Compares per-kernel GPU execution times to test whether
# concurrent embed load slows down individual gen kernels.

VLLM_IMAGE="vllm/vllm-openai:v0.15.1"
HF_CACHE="/home/ubuntu/.cache/huggingface"
VLLM_CACHE="/home/ubuntu/.cache/vllm"
EMBED_MODEL="Qwen/Qwen3-Embedding-8B"
DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS="$DIR/profile_results"
mkdir -p "$RESULTS"

NSYS=/opt/nsys/bin/nsys
NSYS_HOST=/opt/nvidia/nsight-systems/2024.6.2/bin/nsys

full_cleanup() {
    docker stop nsys-gen vllm-embed mps-daemon 2>/dev/null || true
    docker rm nsys-gen vllm-embed mps-daemon 2>/dev/null || true
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

start_embed_load() {
    # Run embed benchmark in background for 120s (longer than gen probe takes)
    python3 "$DIR/embed_benchmark_vllm.py" --base-url http://localhost:8200 \
        --duration 120 --concurrency 64 --batch-size 8 --model "$EMBED_MODEL" > /dev/null 2>&1 &
    EMBED_PID=$!
    echo "  Embed load started (PID $EMBED_PID)"
    sleep 5  # let embed load ramp up
}

run_gen_nsys() {
    local label=$1
    local use_mps=$2
    local extra_args=""
    if [ "$use_mps" = "yes" ]; then
        extra_args="-v mps-pipe:/mps -e CUDA_MPS_PIPE_DIRECTORY=/mps"
    fi
    docker run --rm --name nsys-gen --gpus all --network host --ipc=host --privileged \
        -v "$HF_CACHE:/root/.cache/huggingface" \
        -v "$VLLM_CACHE:/root/.cache/vllm" \
        -v "$DIR:/workspace" \
        -v /opt/nvidia/nsight-systems/2024.6.2:/opt/nsys \
        $extra_args \
        -e HF_HUB_OFFLINE=1 \
        --entrypoint bash \
        "$VLLM_IMAGE" \
        -c "$NSYS profile --trace=cuda -o /workspace/profile_results/nsys_${label} --force-overwrite true -- python3 /workspace/ncu_decode_probe.py 2>&1 && echo 'nsys done'"
}

echo "=== nsys concurrent test: gen_only vs mps_active vs ts_active ==="

# --- Condition 1: gen_only (baseline, no MPS, no embed) ---
echo ""
echo "--- Condition 1: gen_only ---"
full_cleanup
sleep 2
run_gen_nsys "gen_only" "no"
echo "  gen_only trace captured"

# --- Condition 2: mps_active (MPS, embed running with load) ---
echo ""
echo "--- Condition 2: mps_active ---"
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
start_embed_load
run_gen_nsys "mps_active" "yes"
kill $EMBED_PID 2>/dev/null || true
echo "  mps_active trace captured"

# --- Condition 3: ts_active (time-sliced, embed running with load) ---
echo ""
echo "--- Condition 3: ts_active ---"
full_cleanup
sleep 2

start_embed "no"
start_embed_load
run_gen_nsys "ts_active" "no"
kill $EMBED_PID 2>/dev/null || true
echo "  ts_active trace captured"

# --- Extract and compare stats ---
echo ""
echo "--- Extracting kernel stats ---"

for label in gen_only mps_active ts_active; do
    echo ""
    echo "=== $label: top kernels ==="
    $NSYS_HOST stats --report cuda_gpu_kern_sum "$RESULTS/nsys_${label}.nsys-rep" 2>/dev/null | head -30 || echo "stats failed for $label"
    echo ""
    echo "=== $label: cuda api sum ==="
    $NSYS_HOST stats --report cuda_api_sum "$RESULTS/nsys_${label}.nsys-rep" 2>/dev/null | head -20 || echo "api stats failed for $label"
done

echo ""
echo "=== Done ==="
