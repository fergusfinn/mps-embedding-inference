#!/bin/bash
set -euo pipefail

# dmon sweep WITH CUDA GRAPHS (no --enforce-eager)
# Compare to run_dmon_sweep.sh which uses --enforce-eager
# Uses DCGM profiling counters for tensor core / DRAM utilization + nvidia-smi dmon for power/temp

GPU_ID=0
VLLM_PORT=8100
MODEL="${1:-Qwen/Qwen3-30B-A3B}"
GPU_MEM_UTIL="${2:-0.85}"
GEN_MAX_TOKENS=512
MEASURE_DURATION=30  # seconds of steady-state measurement
CONCURRENCY_LEVELS="${3:-1 256 1024 2048 4096}"

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/venv/bin/activate"

# Create results dir with model short name
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
RESULTS_DIR="$DIR/results/cudagraphs_${MODEL_SHORT}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

source "$VENV"
export CUDA_VISIBLE_DEVICES=$GPU_ID
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# Use CUDA 12.8 nvcc for flashinfer FP8 block scaling support
export PATH=/usr/local/cuda-12.8/bin:$PATH

echo "=== FLOP Utilization Sweep (CUDA GRAPHS) ==="
echo "Model: $MODEL"
echo "GPU mem util: $GPU_MEM_UTIL"
echo "Request config: input ~30 tokens, output max $GEN_MAX_TOKENS tokens"
echo "Concurrency levels: $CONCURRENCY_LEVELS"
echo "Mode: CUDA graphs (default, no --enforce-eager)"
echo "Results: $RESULTS_DIR"
echo ""

VLLM_PID=""
DMON_PID=""
DCGM_PID=""

cleanup() {
    [ -n "${DMON_PID:-}" ] && kill "$DMON_PID" 2>/dev/null || true
    [ -n "${DCGM_PID:-}" ] && kill "$DCGM_PID" 2>/dev/null || true
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null && wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

wait_for_vllm() {
    for i in $(seq 1 300); do
        if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
            echo "vLLM ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "ERROR: vLLM failed to start"
    return 1
}

stop_vllm() {
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null && wait "$VLLM_PID" 2>/dev/null || true
    VLLM_PID=""
    sleep 2
}

for C in $CONCURRENCY_LEVELS; do
    echo "=========================================="
    echo "Testing C=$C (max-num-seqs=$C) — CUDA GRAPHS"
    echo "=========================================="

    # Start vLLM WITHOUT --enforce-eager (uses CUDA graphs)
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --port "$VLLM_PORT" \
        --gpu-memory-utilization $GPU_MEM_UTIL --max-model-len 4096 \
        --max-num-seqs "$C" \
        --quantization fp8 \
        --disable-log-requests \
        > "$RESULTS_DIR/vllm_C${C}.log" 2>&1 &
    VLLM_PID=$!
    wait_for_vllm

    # Warmup (longer for CUDA graphs — graph capture happens on first few batches)
    echo "Warming up (includes CUDA graph capture)..."
    python3 "$DIR/gen_benchmark.py" --base-url "http://localhost:$VLLM_PORT" \
        --duration 15 --concurrency "$C" --max-tokens 64 --model "$MODEL" > /dev/null 2>&1

    # Start nvidia-smi dmon (power, temp, clocks)
    nvidia-smi dmon -i $GPU_ID -s ucp -d 1 -c $((MEASURE_DURATION + 5)) \
        > "$RESULTS_DIR/dmon_C${C}.txt" 2>&1 &
    DMON_PID=$!

    # Start DCGM profiling (tensor core, SM, DRAM utilization)
    # Fields: sm_active(1002), sm_occupancy(1003), tensor_active(1004),
    #         dram_active(1005), fp32_active(1007), fp16_active(1008),
    #         tensor_imma_active(1013)
    dcgmi dmon -i 0 -e 1002,1003,1004,1005,1007,1008,1013 \
        -d 1000 -c $((MEASURE_DURATION + 5)) \
        > "$RESULTS_DIR/dcgm_C${C}.txt" 2>&1 &
    DCGM_PID=$!

    # Run benchmark for measurement period
    echo "Measuring for ${MEASURE_DURATION}s at C=$C..."
    python3 "$DIR/gen_benchmark.py" --base-url "http://localhost:$VLLM_PORT" \
        --duration "$MEASURE_DURATION" --concurrency "$C" --max-tokens "$GEN_MAX_TOKENS" \
        --model "$MODEL" > "$RESULTS_DIR/bench_C${C}.txt" 2>&1

    # Stop monitors
    kill "$DMON_PID" 2>/dev/null || true
    wait "$DMON_PID" 2>/dev/null || true
    DMON_PID=""
    kill "$DCGM_PID" 2>/dev/null || true
    wait "$DCGM_PID" 2>/dev/null || true
    DCGM_PID=""

    # Print DCGM summary
    echo ""
    echo "--- DCGM profiling for C=$C ---"
    tail -12 "$RESULTS_DIR/dcgm_C${C}.txt" | head -10
    echo ""

    # Print benchmark result
    echo "--- Benchmark C=$C ---"
    grep -E "(Throughput|tok/s)" "$RESULTS_DIR/bench_C${C}.txt" || tail -5 "$RESULTS_DIR/bench_C${C}.txt"
    echo ""

    stop_vllm
done

echo ""
echo "=== Summary (CUDA GRAPHS) ==="
echo "Model: $MODEL"
echo "Request I/O: input ~30 tokens (fixed prompt), output max $GEN_MAX_TOKENS tokens"
echo "Mode: CUDA graphs (no --enforce-eager)"
echo ""
printf "%-8s %10s %8s %8s %8s %8s %8s %8s %6s %6s\n" \
    "C" "tok/s" "sm_act" "sm_occ" "tensor" "dram" "fp8_tc" "pwr" "temp" "KV%"
for C in $CONCURRENCY_LEVELS; do
    TOKS="-"
    SM_ACT="-" ; SM_OCC="-" ; TENSOR="-" ; DRAM="-" ; TIMMA="-"
    PWR="-" ; TEMP="-" ; KV="-"

    # Extract throughput
    if [ -f "$RESULTS_DIR/bench_C${C}.txt" ]; then
        T=$(grep -oP 'Throughput: \K[0-9.]+' "$RESULTS_DIR/bench_C${C}.txt" 2>/dev/null || echo "")
        [ -n "$T" ] && TOKS=$(printf "%.0f" "$T")
    fi

    # Extract DCGM profiling averages (skip header lines starting with #)
    if [ -f "$RESULTS_DIR/dcgm_C${C}.txt" ]; then
        # Data lines start with "GPU 0" (2 fields), so SMACT=$3, SMOCC=$4, TENSO=$5, DRAMA=$6, FP32A=$7, FP16A=$8, TIMMA=$9
        DATA=$(grep "^GPU 0" "$RESULTS_DIR/dcgm_C${C}.txt" | tail -10)
        if [ -n "$DATA" ]; then
            SM_ACT=$(echo "$DATA" | awk '{sum+=$3; n++} END {if(n>0) printf "%.1f%%", sum/n*100; else print "-"}')
            SM_OCC=$(echo "$DATA" | awk '{sum+=$4; n++} END {if(n>0) printf "%.1f%%", sum/n*100; else print "-"}')
            TENSOR=$(echo "$DATA" | awk '{sum+=$5; n++} END {if(n>0) printf "%.1f%%", sum/n*100; else print "-"}')
            DRAM=$(echo "$DATA"   | awk '{sum+=$6; n++} END {if(n>0) printf "%.1f%%", sum/n*100; else print "-"}')
            TIMMA=$(echo "$DATA"  | awk '{sum+=$9; n++} END {if(n>0) printf "%.1f%%", sum/n*100; else print "-"}')
        fi
    fi

    # Extract power/temp from nvidia-smi dmon
    if [ -f "$RESULTS_DIR/dmon_C${C}.txt" ]; then
        DATA=$(grep -E "^\s+0" "$RESULTS_DIR/dmon_C${C}.txt" | tail -10)
        if [ -n "$DATA" ]; then
            PWR=$(echo "$DATA" | awk '{sum+=$10; n++} END {printf "%.0fW", sum/n}')
            TEMP=$(echo "$DATA" | awk '{sum+=$11; n++} END {printf "%.0fC", sum/n}')
        fi
    fi

    # Extract KV cache usage from vLLM log
    if [ -f "$RESULTS_DIR/vllm_C${C}.log" ]; then
        KV=$(grep -oP 'GPU KV cache usage: \K[0-9.]+' "$RESULTS_DIR/vllm_C${C}.log" 2>/dev/null | tail -1 || echo "")
        [ -n "$KV" ] && KV="${KV}%"
    fi
    [ -z "$KV" ] && KV="-"

    printf "%-8s %10s %8s %8s %8s %8s %8s %8s %6s %6s\n" \
        "C=$C" "$TOKS" "$SM_ACT" "$SM_OCC" "$TENSOR" "$DRAM" "$TIMMA" "$PWR" "$TEMP" "$KV"
done

echo ""
echo "Column legend:"
echo "  sm_act   = SM active fraction (DCGM 1002)"
echo "  sm_occ   = SM warp occupancy (DCGM 1003)"
echo "  tensor   = Any tensor core active (DCGM 1004)"
echo "  dram     = DRAM bandwidth utilization (DCGM 1005)"
echo "  fp8_tc   = INT8/FP8 tensor core (IMMA) active (DCGM 1013)"
echo "  pwr/temp = Power draw and GPU temp (nvidia-smi dmon)"
echo "  KV%      = KV cache utilization (vLLM log, snapshot)"
echo ""
echo "Full results in: $RESULTS_DIR"
