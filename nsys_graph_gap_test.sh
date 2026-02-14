#!/bin/bash
set -euo pipefail

# nsys traces for vLLM server WITH CUDA GRAPHS to measure inter-kernel gaps.
#
# Hypothesis: under MPS, embed kernels interleave into gen's CUDA graph replays,
# widening the gaps between consecutive gen kernels.
#
# Two conditions:
#   1. mps_active:  gen + embed servers, MPS, concurrent load, CUDA graphs ON
#   2. ts_active:   gen + embed servers, time-sliced, concurrent load, CUDA graphs ON
#
# After capture, analyze_graph_gaps.py extracts inter-kernel gap distributions
# from the nsys SQLite databases and compares them.

VLLM_IMAGE="vllm/vllm-openai:v0.15.1"
HF_CACHE="/home/ubuntu/.cache/huggingface"
VLLM_CACHE="/home/ubuntu/.cache/vllm"
GEN_MODEL="Qwen/Qwen3-30B-A3B-FP8"
EMBED_MODEL="Qwen/Qwen3-Embedding-8B"
DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS="$DIR/profile_results"
mkdir -p "$RESULTS"

NSYS=/opt/nsys/bin/nsys
NSYS_HOST=/opt/nvidia/nsight-systems/2024.6.2/bin/nsys

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

start_gen_nsys() {
    local label=$1
    local use_mps=$2
    local extra_args=""
    if [ "$use_mps" = "yes" ]; then
        extra_args="-v mps-pipe:/mps -e CUDA_MPS_PIPE_DIRECTORY=/mps"
    fi
    # Wrap vLLM server with nsys. No --delay/--duration: capture everything,
    # filter by timestamp in analysis. CUDA graphs are ON (no --enforce-eager).
    docker run --rm -d --name vllm-gen \
        --gpus all --network host --ipc=host --privileged \
        -v "$HF_CACHE:/root/.cache/huggingface" \
        -v "$VLLM_CACHE:/root/.cache/vllm" \
        -v "$DIR:/workspace" \
        -v /opt/nvidia/nsight-systems/2024.6.2:/opt/nsys \
        $extra_args \
        -e HF_HUB_OFFLINE=1 \
        --entrypoint "$NSYS" \
        "$VLLM_IMAGE" \
        profile --trace=cuda \
        -o "/workspace/profile_results/nsys_graphs_${label}" --force-overwrite true \
        -- python3 -m vllm.entrypoints.openai.api_server \
        --model "$GEN_MODEL" --port 8100 \
        --gpu-memory-utilization 0.50 --max-model-len 4096 --max-num-seqs 512 \
        --disable-log-requests \
        > /dev/null 2>&1
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

run_condition() {
    local label=$1
    local use_mps=$2

    echo ""
    echo "--- Condition: $label ---"
    full_cleanup
    sleep 2

    if [ "$use_mps" = "yes" ]; then
        start_mps
    fi

    start_gen_nsys "$label" "$use_mps"
    wait_health 8100 "Gen"

    start_embed "$use_mps"

    # Warmup gen + embed for 20s
    echo "  warmup 20s (concurrent)..."
    python3 "$DIR/gen_benchmark.py" --base-url http://localhost:8100 \
        --duration 20 --concurrency 512 --max-tokens 512 --model "$GEN_MODEL" > /dev/null 2>&1 &
    python3 "$DIR/embed_benchmark_vllm.py" --base-url http://localhost:8200 \
        --duration 20 --concurrency 64 --batch-size 8 --model "$EMBED_MODEL" > /dev/null 2>&1 &
    wait

    # Measurement: 30s gen + 35s embed (embed runs longer to ensure overlap)
    echo "  measuring 30s (concurrent, nsys capturing)..."
    python3 "$DIR/embed_benchmark_vllm.py" --base-url http://localhost:8200 \
        --duration 35 --concurrency 64 --batch-size 8 --model "$EMBED_MODEL" > /dev/null 2>&1 &
    EMBED_PID=$!

    local gen_result
    gen_result=$(python3 "$DIR/gen_benchmark.py" --base-url http://localhost:8100 \
        --duration 30 --concurrency 512 --max-tokens 512 --model "$GEN_MODEL" 2>&1)
    echo "  $label: $(echo "$gen_result" | grep "Throughput:")"

    wait $EMBED_PID || true

    # Send SIGINT to nsys inside the container to stop collection and write trace.
    # nsys is PID 1 in the container (it's the entrypoint).
    echo "  signaling nsys to stop collection..."
    docker kill --signal=SIGINT vllm-gen 2>/dev/null || true

    # Wait for nsys to finalize the trace (it writes on SIGINT, then exits)
    echo "  waiting for nsys to finalize trace..."
    for i in $(seq 1 120); do
        docker inspect vllm-gen > /dev/null 2>&1 || break
        sleep 1
    done

    # Give a moment for file system sync
    sleep 2

    if [ -f "$RESULTS/nsys_graphs_${label}.nsys-rep" ]; then
        local size
        size=$(stat -c%s "$RESULTS/nsys_graphs_${label}.nsys-rep")
        if [ "$size" -gt 0 ]; then
            echo "  trace captured: nsys_graphs_${label}.nsys-rep ($(du -h "$RESULTS/nsys_graphs_${label}.nsys-rep" | cut -f1))"
        else
            echo "  WARNING: trace file is empty (0 bytes)"
        fi
    else
        echo "  WARNING: trace file not found!"
    fi
}

echo "=== nsys CUDA graph gap test: MPS vs time-sliced ==="

run_condition "mps_active" "yes"
run_condition "ts_active" "no"

# --- Analyze gaps ---
echo ""
echo "--- Analyzing inter-kernel gaps ---"

# Export to SQLite for analysis
for label in mps_active ts_active; do
    REP="$RESULTS/nsys_graphs_${label}.nsys-rep"
    DB="$RESULTS/nsys_graphs_${label}.sqlite"
    if [ -f "$REP" ]; then
        echo "  Exporting $label to SQLite..."
        $NSYS_HOST export --type sqlite --force-overwrite true -o "$DB" "$REP" 2>/dev/null || \
        $NSYS_HOST stats --force-export=true "$REP" > /dev/null 2>/dev/null || true
    fi
done

# Run gap analysis
python3 "$DIR/analyze_graph_gaps.py" \
    "$RESULTS/nsys_graphs_mps_active.sqlite" \
    "$RESULTS/nsys_graphs_ts_active.sqlite"

echo ""
echo "=== Done ==="
