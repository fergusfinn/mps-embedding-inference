#!/bin/bash
set -euo pipefail

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
}
trap full_cleanup EXIT

wait_health() {
    local port=$1 label=$2
    for i in $(seq 1 120); do
        curl -s "http://localhost:$port/health" > /dev/null 2>&1 && echo "$label ready after ${i}s" && return
        sleep 1
    done
    echo "$label FAILED to start" && exit 1
}

run_trial() {
    local mode=$1 rep=$2
    echo "=== $mode rep $rep ==="

    full_cleanup

    if [ "$mode" = "mps" ]; then
        docker volume create mps-pipe > /dev/null 2>&1 || true
        docker run --rm -d --name mps-daemon \
            --gpus all --ipc=host --entrypoint bash \
            -v mps-pipe:/mps "$VLLM_IMAGE" \
            -c 'CUDA_MPS_PIPE_DIRECTORY=/mps nvidia-cuda-mps-control -d && sleep infinity' \
            > /dev/null 2>&1
        sleep 2

        docker run --rm -d --name vllm-gen \
            --gpus all --network host --ipc=host \
            -v "$HF_CACHE:/root/.cache/huggingface" \
            -v "$VLLM_CACHE:/root/.cache/vllm" \
            -v mps-pipe:/mps \
            -e CUDA_MPS_PIPE_DIRECTORY=/mps \
            -e HF_HUB_OFFLINE=1 \
            "$VLLM_IMAGE" "$GEN_MODEL" \
            --port 8100 --gpu-memory-utilization 0.50 \
            --max-model-len 4096 --max-num-seqs 512 \
            --disable-log-requests > /dev/null 2>&1
        wait_health 8100 "Gen"

        docker run --rm -d --name vllm-embed \
            --gpus all --network host --ipc=host \
            -v "$HF_CACHE:/root/.cache/huggingface" \
            -v "$VLLM_CACHE:/root/.cache/vllm" \
            -v mps-pipe:/mps \
            -e CUDA_MPS_PIPE_DIRECTORY=/mps \
            -e HF_HUB_OFFLINE=1 \
            "$VLLM_IMAGE" "$EMBED_MODEL" \
            --port 8200 --gpu-memory-utilization 0.30 \
            --max-model-len 512 --convert embed \
            --disable-log-requests > /dev/null 2>&1
        wait_health 8200 "Embed"
    else
        docker run --rm -d --name vllm-gen \
            --gpus all --network host --ipc=host \
            -v "$HF_CACHE:/root/.cache/huggingface" \
            -v "$VLLM_CACHE:/root/.cache/vllm" \
            -e HF_HUB_OFFLINE=1 \
            "$VLLM_IMAGE" "$GEN_MODEL" \
            --port 8100 --gpu-memory-utilization 0.50 \
            --max-model-len 4096 --max-num-seqs 512 \
            --disable-log-requests > /dev/null 2>&1
        wait_health 8100 "Gen"

        docker run --rm -d --name vllm-embed \
            --gpus all --network host --ipc=host \
            -v "$HF_CACHE:/root/.cache/huggingface" \
            -v "$VLLM_CACHE:/root/.cache/vllm" \
            -e HF_HUB_OFFLINE=1 \
            "$VLLM_IMAGE" "$EMBED_MODEL" \
            --port 8200 --gpu-memory-utilization 0.30 \
            --max-model-len 512 --convert embed \
            --disable-log-requests > /dev/null 2>&1
        wait_health 8200 "Embed"
    fi

    echo "  warmup 30s..."
    python3 "$DIR/gen_benchmark.py" --base-url http://localhost:8100 \
        --duration 30 --concurrency 512 --max-tokens 512 --model "$GEN_MODEL" > /dev/null 2>&1

    echo "  measuring gen 60s..."
    local gen_result
    gen_result=$(python3 "$DIR/gen_benchmark.py" --base-url http://localhost:8100 \
        --duration 60 --concurrency 512 --max-tokens 512 --model "$GEN_MODEL" 2>&1 | grep "Throughput:")

    echo "  measuring embed 60s..."
    local embed_result
    embed_result=$(python3 "$DIR/embed_benchmark_vllm.py" --base-url http://localhost:8200 \
        --duration 60 --concurrency 64 --batch-size 8 --model "$EMBED_MODEL" 2>&1 | grep "Throughput:")

    echo "  $mode rep $rep: GEN  $gen_result"
    echo "  $mode rep $rep: EMBED $embed_result"
    echo ""
}

echo "=== A/B test: time-sliced vs MPS, 3 reps each (alternating) ==="
echo ""

for rep in 1 2 3; do
    run_trial "timeslice" "$rep"
    run_trial "mps" "$rep"
done

echo "=== Done ==="
