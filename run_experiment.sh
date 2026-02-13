#!/bin/bash
set -euo pipefail

# GPU co-scheduling experiment: gen + embed, all via vLLM in Docker.
#
# Conditions:
#   1. Full GPU: gen only
#   2. Full GPU: gen + embed, MPS
#   3. Full GPU: gen + embed, time-sliced (no MPS)
#   4. MIG 4g+3g: gen only (4g partition)
#   5. MIG 4g+3g: gen + embed (4g gen, 3g embed)
#
# All models served via vLLM in Docker containers.
# MIG containers see their partition as a plain GPU (Docker handles UUID mapping).

VLLM_IMAGE="vllm/vllm-openai:v0.15.1"
HF_CACHE="/home/ubuntu/.cache/huggingface"
VLLM_CACHE="/home/ubuntu/.cache/vllm"

GEN_MODEL="Qwen/Qwen3-30B-A3B-FP8"
EMBED_MODEL="Qwen/Qwen3-Embedding-8B"
GEN_PORT=8100
EMBED_PORT=8200
GEN_MEM_UTIL=0.50
EMBED_MEM_UTIL=0.30
MIG_GEN_MEM_UTIL=0.85
GEN_CONCURRENCY=1024
GEN_MAX_TOKENS=512
EMBED_CONCURRENCY=64
EMBED_BATCH_SIZE=8
WARMUP_DURATION=30
MEASURE_DURATION=60

DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$DIR/results/experiment_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "=== GPU Co-scheduling Experiment ==="
echo "Gen:   $GEN_MODEL (vLLM, C=$GEN_CONCURRENCY)"
echo "Embed: $EMBED_MODEL (vLLM, --convert embed, C=$EMBED_CONCURRENCY, batch=$EMBED_BATCH_SIZE)"
echo "Image: $VLLM_IMAGE"
echo "Warmup: ${WARMUP_DURATION}s, Measure: ${MEASURE_DURATION}s"
echo "Results: $RESULTS_DIR"
echo ""

cleanup() {
    echo "Cleaning up..."
    docker stop vllm-gen vllm-embed mps-daemon 2>/dev/null || true
    docker rm vllm-gen vllm-embed mps-daemon 2>/dev/null || true
    docker volume rm mps-pipe 2>/dev/null || true
    if nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader 2>/dev/null | grep -q Enabled; then
        sudo nvidia-smi mig -i 0 -dci 2>/dev/null || true
        sudo nvidia-smi mig -i 0 -dgi 2>/dev/null || true
        sudo nvidia-smi -i 0 -mig 0 2>/dev/null || true
    fi
}
trap cleanup EXIT

wait_for_server() {
    local port=$1
    local name=$2
    local container=$3
    for i in $(seq 1 600); do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            echo "$name ready after ${i}s"
            return 0
        fi
        if ! docker ps -q --filter "name=$container" | grep -q .; then
            echo "ERROR: $name container exited"
            docker logs "$container" 2>&1 | tail -20
            return 1
        fi
        sleep 1
    done
    echo "ERROR: $name failed to start (timeout)"
    docker logs "$container" 2>&1 | tail -20
    return 1
}

start_gen() {
    local tag=$1
    local mem_util=$2
    shift 2
    docker run --rm -d --name vllm-gen \
        "$@" \
        --network host \
        --ipc=host \
        -v "$HF_CACHE:/root/.cache/huggingface" \
        -v "$VLLM_CACHE:/root/.cache/vllm" \
        -e HF_HUB_OFFLINE=1 \
        "$VLLM_IMAGE" \
        "$GEN_MODEL" \
        --port "$GEN_PORT" \
        --gpu-memory-utilization "$mem_util" \
        --max-model-len 4096 \
        --max-num-seqs "$GEN_CONCURRENCY" \
        --disable-log-requests \
        > "$RESULTS_DIR/docker_gen_${tag}.log" 2>&1
    wait_for_server "$GEN_PORT" "Gen" "vllm-gen"
}

stop_gen() {
    docker stop vllm-gen 2>/dev/null || true
    docker rm vllm-gen 2>/dev/null || true
    sleep 2
}

start_embed() {
    local tag=$1
    local mem_util=$2
    shift 2
    docker run --rm -d --name vllm-embed \
        "$@" \
        --network host \
        --ipc=host \
        -v "$HF_CACHE:/root/.cache/huggingface" \
        -v "$VLLM_CACHE:/root/.cache/vllm" \
        -e HF_HUB_OFFLINE=1 \
        "$VLLM_IMAGE" \
        "$EMBED_MODEL" \
        --convert embed \
        --port "$EMBED_PORT" \
        --gpu-memory-utilization "$mem_util" \
        --max-model-len 512 \
        --disable-log-requests \
        > "$RESULTS_DIR/docker_embed_${tag}.log" 2>&1
    wait_for_server "$EMBED_PORT" "Embed" "vllm-embed"
}

stop_embed() {
    docker stop vllm-embed 2>/dev/null || true
    docker rm vllm-embed 2>/dev/null || true
    sleep 2
}

warmup_gen() {
    echo "Warming up gen (${WARMUP_DURATION}s)..."
    python3 "$DIR/gen_benchmark.py" --base-url "http://localhost:$GEN_PORT" \
        --duration "$WARMUP_DURATION" --concurrency "$GEN_CONCURRENCY" \
        --max-tokens "$GEN_MAX_TOKENS" --model "$GEN_MODEL" > /dev/null 2>&1
}

measure_gen() {
    local tag=$1
    echo "Measuring gen for ${MEASURE_DURATION}s..."
    python3 "$DIR/gen_benchmark.py" --base-url "http://localhost:$GEN_PORT" \
        --duration "$MEASURE_DURATION" --concurrency "$GEN_CONCURRENCY" \
        --max-tokens "$GEN_MAX_TOKENS" --model "$GEN_MODEL" \
        > "$RESULTS_DIR/bench_${tag}_gen.txt" 2>&1
}

measure_embed() {
    local tag=$1
    echo "Measuring embed for ${MEASURE_DURATION}s..."
    python3 "$DIR/embed_benchmark_vllm.py" --base-url "http://localhost:$EMBED_PORT" \
        --duration "$MEASURE_DURATION" --concurrency "$EMBED_CONCURRENCY" \
        --batch-size "$EMBED_BATCH_SIZE" --model "$EMBED_MODEL" \
        > "$RESULTS_DIR/bench_${tag}_embed.txt" 2>&1
}

print_results() {
    local tag=$1
    local gen="$RESULTS_DIR/bench_${tag}_gen.txt"
    local embed="$RESULTS_DIR/bench_${tag}_embed.txt"
    echo "--- $tag ---"
    [ -f "$gen" ] && grep -E "Throughput:|p50:" "$gen" || true
    [ -f "$embed" ] && grep -E "Throughput:|p50:" "$embed" || true
    echo ""
}

# Activate venv for benchmark clients (they use aiohttp)
source "$DIR/venv/bin/activate"

# =========================================
# STEP 1: Full GPU, gen only
# =========================================
echo "=========================================="
echo "STEP 1: Full GPU, gen only"
echo "=========================================="
start_gen "step1" "$GEN_MEM_UTIL" --gpus all
warmup_gen
measure_gen "step1"
print_results "step1"
stop_gen

# =========================================
# STEP 2: Full GPU, gen + embed, MPS
# =========================================
echo "=========================================="
echo "STEP 2: Full GPU, gen + embed (MPS)"
echo "=========================================="
# MPS requires containers to share a CUDA context via the MPS daemon.
# Docker containers can't connect to a host MPS daemon (credential/namespace mismatch),
# so we run MPS inside a sidecar container and share the pipe dir via a Docker volume.
docker volume create mps-pipe > /dev/null 2>&1
docker run --rm -d --name mps-daemon \
    --gpus all \
    --ipc=host \
    --entrypoint bash \
    -v mps-pipe:/mps \
    "$VLLM_IMAGE" \
    -c 'CUDA_MPS_PIPE_DIRECTORY=/mps nvidia-cuda-mps-control -d && sleep infinity'
sleep 3

MPS_ARGS=(-v mps-pipe:/mps -e "CUDA_MPS_PIPE_DIRECTORY=/mps")
start_gen "step2" "$GEN_MEM_UTIL" --gpus all "${MPS_ARGS[@]}"
start_embed "step2" "$EMBED_MEM_UTIL" --gpus all "${MPS_ARGS[@]}"
warmup_gen

measure_embed "step2" &
EMBED_BENCH_PID=$!
measure_gen "step2"
wait "$EMBED_BENCH_PID"

print_results "step2"
stop_gen
stop_embed
docker stop mps-daemon 2>/dev/null || true
docker rm mps-daemon 2>/dev/null || true
docker volume rm mps-pipe 2>/dev/null || true
sleep 2

# =========================================
# STEP 3: Full GPU, gen + embed, no MPS
# =========================================
echo "=========================================="
echo "STEP 3: Full GPU, gen + embed (time-sliced)"
echo "=========================================="
start_gen "step3" "$GEN_MEM_UTIL" --gpus all
start_embed "step3" "$EMBED_MEM_UTIL" --gpus all
warmup_gen

measure_embed "step3" &
EMBED_BENCH_PID=$!
measure_gen "step3"
wait "$EMBED_BENCH_PID"

print_results "step3"
stop_gen
stop_embed

# =========================================
# STEP 4: MIG, gen only (4g.71gb)
# =========================================
echo "=========================================="
echo "STEP 4: MIG, gen only (4g.71gb)"
echo "=========================================="
sudo nvidia-smi -i 0 -mig 1
sudo nvidia-smi mig -i 0 -cgi 5,9
sudo nvidia-smi mig -i 0 -gi 1 -cci 0
sudo nvidia-smi mig -i 0 -gi 2 -cci 0

MIG_4G=$(nvidia-smi -L | grep '1c.4g.71gb' | grep -oP 'MIG-[a-f0-9-]+')
MIG_3G=$(nvidia-smi -L | grep '1c.3g.71gb' | grep -oP 'MIG-[a-f0-9-]+')
echo "Gen partition:   4g.71gb ($MIG_4G)"
echo "Embed partition: 3g.71gb ($MIG_3G)"

start_gen "step4" "$MIG_GEN_MEM_UTIL" --gpus "\"device=${MIG_4G}\""
warmup_gen
measure_gen "step4"
print_results "step4"
stop_gen

# =========================================
# STEP 5: MIG, gen + embed (4g + 3g)
# =========================================
echo "=========================================="
echo "STEP 5: MIG, gen (4g) + embed (3g)"
echo "=========================================="
start_gen "step5" "$MIG_GEN_MEM_UTIL" --gpus "\"device=${MIG_4G}\""
start_embed "step5" "$MIG_GEN_MEM_UTIL" --gpus "\"device=${MIG_3G}\""
warmup_gen

measure_embed "step5" &
EMBED_BENCH_PID=$!
measure_gen "step5"
wait "$EMBED_BENCH_PID"

print_results "step5"
stop_gen
stop_embed

echo "Disabling MIG..."
sudo nvidia-smi mig -i 0 -dci
sudo nvidia-smi mig -i 0 -dgi
sudo nvidia-smi -i 0 -mig 0

# =========================================
# FINAL SUMMARY
# =========================================
echo ""
echo "=========================================="
echo "FINAL SUMMARY"
echo "=========================================="
echo ""

extract() { grep -oP 'Throughput: \K[0-9.]+' "$1" 2>/dev/null || echo "0"; }

GEN1=$(extract "$RESULTS_DIR/bench_step1_gen.txt")
GEN2=$(extract "$RESULTS_DIR/bench_step2_gen.txt")
GEN3=$(extract "$RESULTS_DIR/bench_step3_gen.txt")
GEN4=$(extract "$RESULTS_DIR/bench_step4_gen.txt")
GEN5=$(extract "$RESULTS_DIR/bench_step5_gen.txt")
EMB2=$(extract "$RESULTS_DIR/bench_step2_embed.txt")
EMB3=$(extract "$RESULTS_DIR/bench_step3_embed.txt")
EMB5=$(extract "$RESULTS_DIR/bench_step5_embed.txt")

delta() {
    if [ "$1" != "0" ] && [ "$2" != "0" ]; then
        python3 -c "print(f'{($2/$1 - 1)*100:+.1f}%')"
    else
        echo "-"
    fi
}

printf "%-45s %15s %15s %10s\n" "" "Gen (tok/s)" "Embed (seq/s)" "Gen delta"
printf "%-45s %15s %15s %10s\n" "1. Full GPU: gen only"            "$GEN1" "-"    "baseline"
printf "%-45s %15s %15s %10s\n" "2. Full GPU: gen+embed (MPS)"     "$GEN2" "$EMB2" "$(delta $GEN1 $GEN2)"
printf "%-45s %15s %15s %10s\n" "3. Full GPU: gen+embed (no MPS)"  "$GEN3" "$EMB3" "$(delta $GEN1 $GEN3)"
printf "%-45s %15s %15s %10s\n" "4. MIG 4g: gen only"              "$GEN4" "-"    "$(delta $GEN1 $GEN4)"
printf "%-45s %15s %15s %10s\n" "5. MIG 4g+3g: gen+embed"          "$GEN5" "$EMB5" "$(delta $GEN1 $GEN5)"

echo ""
echo "Full results in: $RESULTS_DIR"
