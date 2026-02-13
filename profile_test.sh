#!/bin/bash
set -euo pipefail

VLLM_IMAGE="vllm/vllm-openai:v0.15.1"
HF_CACHE="/home/ubuntu/.cache/huggingface"
VLLM_CACHE="/home/ubuntu/.cache/vllm"
GEN_MODEL="Qwen/Qwen3-30B-A3B-FP8"
EMBED_MODEL="Qwen/Qwen3-Embedding-8B"
DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS="$DIR/profile_results"
mkdir -p "$RESULTS"

full_cleanup() {
    docker stop vllm-gen vllm-embed mps-daemon 2>/dev/null || true
    docker rm vllm-gen vllm-embed mps-daemon 2>/dev/null || true
    docker volume rm mps-pipe 2>/dev/null || true
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

# DCGM fields: sm_active(1002), sm_occupancy(1003), tensor_active(1004),
# dram_active(1005), fp32_active(1007), fp16_active(1008)
DCGM_FIELDS="1002,1003,1004,1005,1007,1008"

run_profiled() {
    local mode=$1
    echo ""
    echo "=========================================="
    echo "  Profiling: $mode"
    echo "=========================================="

    full_cleanup
    sleep 2

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
            -v mps-pipe:/mps -e CUDA_MPS_PIPE_DIRECTORY=/mps \
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
            -v mps-pipe:/mps -e CUDA_MPS_PIPE_DIRECTORY=/mps \
            -e HF_HUB_OFFLINE=1 \
            "$VLLM_IMAGE" "$EMBED_MODEL" \
            --port 8200 --gpu-memory-utilization 0.30 \
            --max-model-len 512 --convert embed \
            --disable-log-requests > /dev/null 2>&1
        wait_health 8200 "Embed"

    elif [ "$mode" = "timeslice" ]; then
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

    else
        # gen-only baseline
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
    fi

    # Warmup both workloads
    echo "  warmup 30s..."
    python3 "$DIR/gen_benchmark.py" --base-url http://localhost:8100 \
        --duration 30 --concurrency 512 --max-tokens 512 --model "$GEN_MODEL" > /dev/null 2>&1 &
    local GEN_WARM_PID=$!
    if [ "$mode" != "gen_only" ]; then
        python3 "$DIR/embed_benchmark_vllm.py" --base-url http://localhost:8200 \
            --duration 30 --concurrency 64 --batch-size 8 --model "$EMBED_MODEL" > /dev/null 2>&1 &
        local EMBED_WARM_PID=$!
    fi
    wait $GEN_WARM_PID || true
    if [ "$mode" != "gen_only" ]; then
        wait $EMBED_WARM_PID || true
    fi

    echo "  measuring 60s with DCGM profiling..."

    # Start DCGM dmon (1 sample/sec, 60 samples)
    dcgmi dmon -e "$DCGM_FIELDS" -d 1000 -c 60 > "$RESULTS/dcgm_${mode}.txt" 2>&1 &
    local DCGM_PID=$!

    # Run gen + embed load concurrently
    python3 "$DIR/gen_benchmark.py" --base-url http://localhost:8100 \
        --duration 60 --concurrency 512 --max-tokens 512 --model "$GEN_MODEL" 2>&1 &
    local GEN_PID=$!

    if [ "$mode" != "gen_only" ]; then
        python3 "$DIR/embed_benchmark_vllm.py" --base-url http://localhost:8200 \
            --duration 60 --concurrency 64 --batch-size 8 --model "$EMBED_MODEL" 2>&1 &
        local EMBED_PID=$!
    fi

    wait $GEN_PID || true
    echo "  gen result:"
    # gen already printed its output

    if [ "$mode" != "gen_only" ]; then
        wait $EMBED_PID || true
    fi
    wait $DCGM_PID || true

    echo "  DCGM saved to $RESULTS/dcgm_${mode}.txt"
}

echo "=== DCGM Profiling: gen_only vs timeslice vs mps ==="

run_profiled "gen_only"
run_profiled "timeslice"
run_profiled "mps"

echo ""
echo "=== Summary ==="
echo ""

for mode in gen_only timeslice mps; do
    echo "--- $mode ---"
    # Parse DCGM output: skip header lines, average the numeric fields
    # Fields: SMACT SMOCC TENSO DRAMA FP32A FP16A
    awk '
    !/^#/ && !/^Entity/ && !/^ID/ && !/^$/ && NF>=7 {
        smact+=$2; smocc+=$3; tensor+=$4; dram+=$5; fp32+=$6; fp16+=$7; n++
    }
    END {
        if(n>0) {
            printf "  SM Active:     %.3f\n", smact/n
            printf "  SM Occupancy:  %.3f\n", smocc/n
            printf "  Tensor Active: %.3f\n", tensor/n
            printf "  DRAM Active:   %.3f\n", dram/n
            printf "  FP32 Active:   %.3f\n", fp32/n
            printf "  FP16 Active:   %.3f\n", fp16/n
            printf "  (n=%d samples)\n", n
        }
    }' "$RESULTS/dcgm_${mode}.txt"
    echo ""
done
