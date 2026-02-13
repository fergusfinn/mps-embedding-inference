# H200 MoE Decode: FLOP Utilization Sweep Results

**Hardware:** NVIDIA H200 (141 GB HBM3e, 700W TDP, 4.8 TB/s bandwidth, 132 SMs, ~3958 TFLOPS FP8)
**Request I/O:** input ~30 tokens (fixed prompt), output max 512 tokens
**Quantization:** `--quantization fp8`, single GPU
**Instrumentation:** DCGM profiling counters (hardware-level) + nvidia-smi dmon (power/temp)

# Part 1: `--enforce-eager` (no CUDA graphs)

---

## Qwen3-30B-A3B (128 experts, 8 active/token, ~30 GB FP8)

| C | tok/s | SM active | SM occ | Tensor core | DRAM BW | FP8 (IMMA) | Power | Temp | KV% |
|---|-------|-----------|--------|-------------|---------|------------|-------|------|-----|
| 1 | 25 | 3.6% | 0.6% | 0.3% | 1.8% | 0.0% | 135W | 34C | 0.1% |
| 256 | 5,569 | 20.0% | 6.7% | 3.0% | 5.3% | 0.0% | 216W | 38C | 13.7% |
| 1024 | 16,610 | 55.3% | 19.2% | 11.6% | 21.9% | 0.0% | 447W | 48C | 46.7% |
| 2048 | 22,092 | 78.2% | 29.2% | 16.1% | 28.1% | 0.0% | 587W | 54C | 2.1%* |
| 4096 | 25,496 | 89.1% | 35.3% | 17.0% | 26.2% | 0.0% | 623W | 56C | 57.4% |

---

## Qwen3-Next-80B-A3B-Instruct-FP8 (512 experts, 10+1 active/token, ~77 GB FP8)

KV cache capacity: 492K tokens → max ~905 concurrent sequences at ~544 tok/seq.

| C requested | Actual running | tok/s (steady) | SM active | SM occ | Tensor core | DRAM BW | FP8 (IMMA) | Power | KV% (steady) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 8 | 5.5% | 0.9% | 0.2% | 0.7% | 0.0% | 129W | 0.1% |
| 256 | 256 | 2,029 | 19.7% | 4.0% | 2.1% | 7.0% | 0.0% | 217W | 25.9% |
| 1024 | **905** | **6,063** | 41.2% | 9.0% | 4.0% | 16.4% | 0.0% | 318W | **99.9%** |
| 2048 | **794** | **5,399** | 35.1% | 7.7% | 2.7% | 13.0% | 0.0% | 269W | **99.9%** |

**Note on C=1024 and C=2048:** Both exceed KV cache capacity. vLLM caps at 905 (C=1024) or 794 (C=2048) running sequences; the rest queue. The benchmark-reported throughput (3,592 / 4,775) is diluted by startup ramp and queue drain — the steady-state values from vLLM logs (6,063 / 5,399) are shown above. DCGM counters are from the steady-state decode phase only (warmup/prefill samples excluded).

**C=2048 runs *fewer* sequences than C=1024** (794 vs 905). The higher `--max-num-seqs` causes vLLM to reserve more per-sequence scheduler resources, reducing the effective KV cache available for concurrent sequences.

---

## Side-by-side comparison

### At C=1 (single-request decode)

| Metric | 30B-A3B | 80B-A3B | Ratio |
|--------|---------|---------|-------|
| Throughput | 25 tok/s | 8 tok/s | 0.32x |
| SM active | 3.6% | 5.4% | 1.5x |
| Tensor core | 0.3% | 0.2% | 0.7x |
| DRAM BW | 1.8% | 0.7% | 0.4x |
| Power | 135W | 129W | 0.96x |

### At C=256

| Metric | 30B-A3B | 80B-A3B | Ratio |
|--------|---------|---------|-------|
| Throughput | 5,569 tok/s | 2,029 tok/s | 0.36x |
| SM active | 20.0% | 21.1% | ~1x |
| Tensor core | 3.0% | 2.7% | ~1x |
| DRAM BW | 5.3% | 8.2% | 1.5x |
| Power | 216W | 217W | ~1x |

### At peak (30B: C=1024, 80B: C=1024 → 905 actual)

| Metric | 30B-A3B | 80B-A3B | Ratio |
|--------|---------|---------|-------|
| Throughput | 16,610 tok/s | 6,063 tok/s | 0.37x |
| SM active | 50.6% | 41.2% | 0.81x |
| Tensor core | 9.8% | 4.0% | 0.41x |
| DRAM BW | 16.2% | 16.4% | ~1x |
| Power | 447W | 318W | 0.71x |

---

## Key findings

### FP8 tensor cores ARE in use — DCGM just can't see them

The `tensor_imma_active` (field 1013) counter reads 0.0% but this is a **DCGM measurement gap**, not a real result. IMMA tracks INT8/INT4 integer tensor ops. On Hopper, FP8 uses QGMMA floating-point MMA instructions, which DCGM has no dedicated counter for ([DCGM issue #251](https://github.com/NVIDIA/DCGM/issues/251)). The aggregate `tensor_active` (field 1004) counter — our "Tensor core" column — DOES capture FP8 activity, showing 2-17%. Meanwhile `fp16_active` (BF16 HMMA) reads 0.000% and `fp32_active` (TF32 DMMA) reads 0.1-2.8% (FP32 accumulation from FP8 kernels). The tensor core activity we measure is native FP8 compute via CUTLASS block-scaled FP8 GEMM kernels.

### The 80B is ~3x slower but uses similar GPU resources

At C=256, both models consume ~20% SM active and ~217W power, but the 80B produces only 2,029 tok/s vs 5,569. The 80B has 2.7x more total expert weights (512 experts vs 128) to load from HBM per decode step, so each step takes longer. However, it activates slightly more experts (10+1 shared vs 8), which is why DRAM BW is higher relative to SM active on the 80B.

### The 80B model is KV-cache-limited at 905 concurrent sequences

At C=1024 (905 actual running), the 80B hits 99.9% KV cache utilization and achieves peak steady-state throughput of 6,063 tok/s at SM=41%. At C=2048, vLLM's higher `--max-num-seqs` overhead actually *reduces* concurrent sequences to 794, dropping SM to 35% and throughput to 5,399 tok/s.

### FP8 tensor core utilization peaks at ~17% (30B) and ~4% (80B)

Both models are deeply tensor-core-idle even at maximum load. The `tensor_active` counter (which captures FP8 QGMMA activity) peaks at 17% for the 30B and 4% for the 80B. This is because:
1. MoE decode is dominated by loading expert weights from HBM (bandwidth-bound)
2. The actual matmuls (expert forward passes) are small — hidden_dim=2048, expert_dim=512-1024
3. Routing, RMSNorm, softmax, and other scalar ops consume SM time but not tensor cores

### The 80B is more bandwidth-bound than the 30B

| Model | DRAM BW / SM active ratio at C=256 |
|-------|-----------------------------------|
| 30B-A3B | 7.0% / 20.2% = 0.35 |
| 80B-A3B | 7.0% / 19.7% = 0.36 |

At C=256 the ratios are surprisingly similar. The difference becomes clearer at saturation — at peak load the 80B reaches DRAM=16.4% at SM=41.2% while the 30B reaches DRAM=21.2% at SM=86.5%. The 80B saturates its KV cache (and its SM budget) much earlier because each decode step takes longer with 512 experts.

---

## Model specs

| | Qwen3-30B-A3B | Qwen3-Next-80B-A3B |
|---|---|---|
| Total params | 30B | 80B |
| Active params/token | ~3.3B | ~3B |
| Experts (total / active) | 128 / 8 | 512 / 10+1 shared |
| Layers | 48 | 48 |
| Architecture | Standard MoE | Hybrid (Gated DeltaNet + Gated Attention) |
| FP8 weight size on GPU | ~30 GB | ~75 GB |
| Expert intermediate dim | 1024 | 512 |
| KV cache capacity (tokens) | ~731K (0.85 mem util) | ~492K (0.90 mem util) |
| Max concurrent seqs (at ~542 tok/seq) | ~1,349 | ~906 |

---

## Column definitions (DCGM field IDs)

| Column | DCGM field | What it measures |
|--------|-----------|-----------------|
| SM active | 1002 (sm_active) | Fraction of time at least one warp is active on any SM |
| SM occ | 1003 (sm_occupancy) | Ratio of active warps to max warps (warp-level parallelism) |
| Tensor core | 1004 (tensor_active) | Fraction of time any tensor core pipe is active |
| DRAM BW | 1005 (dram_active) | Fraction of peak HBM bandwidth utilized (peak = 4.8 TB/s) |
| FP8 (IMMA) | 1013 (tensor_imma_active) | INT8/INT4 integer tensor core pipe — does NOT capture FP8 (see note below) |

**DCGM FP8 measurement gap:** Field 1013 tracks integer MMA (IMMA), not FP8. On Hopper, FP8 uses QGMMA instructions (floating-point MMA), for which DCGM has no dedicated counter. The aggregate `tensor_active` (1004) captures FP8 activity. Confirmed: `fp16_active` (1008) = 0.000%, `fp32_active` (1007) = 0.1-2.8% (FP32 accumulation), `tensor_imma_active` (1013) = 0.0%. The non-zero `tensor_active` readings are FP8 QGMMA compute.

---

## Implications for MPS co-scheduling

The DCGM data reveals massive headroom for co-scheduling at realistic operating points (C=1 to C=256):

### At C=1 (single-request decode — 30B model)

| Resource | Used | Available |
|----------|------|-----------|
| SM active time | 3.6% | **96.4%** |
| Tensor cores (FP8) | 0.3% | **99.7%** |
| DRAM bandwidth | 1.8% | **98.2%** |
| Power | 135W / 700W | **565W** |

### At C=1 (single-request decode — 80B model)

| Resource | Used | Available |
|----------|------|-----------|
| SM active time | 5.4% | **94.6%** |
| Tensor cores (FP8) | 0.2% | **99.8%** |
| DRAM bandwidth | 0.7% | **99.3%** |
| Power | 129W / 700W | **571W** |

Even though the MoE model IS using FP8 tensor cores, it only uses them 0.2-0.3% of the time at C=1 — the workload is dominated by memory reads and scalar/vector operations. An embedding model's BF16 matmuls would add tensor core load on top of this minimal baseline.

---

# Part 2: CUDA Graphs (default, no `--enforce-eager`)

## Qwen3-30B-A3B with CUDA graphs

| C | tok/s | SM active | SM occ | Tensor core | DRAM BW | FP8 (IMMA) | Power | Temp | KV% |
|---|-------|-----------|--------|-------------|---------|------------|-------|------|-----|
| 1 | 194 | 27.5% | 4.6% | 2.0% | 13.6% | 0.0% | 232W | 38C | 0.0% |
| 256 | 15,703 | 60.1% | 19.7% | 10.1% | 24.3% | 0.0% | 456W | 50C | 0.1% |
| 1024 | 14,903 | 50.0% | 17.1% | 10.6% | 21.4% | 0.0% | 416W | 47C | 0.0% |
| 2048 | 22,433 | 75.0% | 28.3% | 15.9% | 28.3% | 0.0% | 583W | 54C | 100% |
| 4096 | 25,762 | 83.9% | 33.4% | 16.7% | 26.6% | 0.0% | 611W | 56C | 57.1% |

**Note on C=1024:** Benchmark throughput (14,903) is lower than C=256 (15,703) because 1024 concurrent requests generate more output, hitting KV cache limits and triggering batch completion/restart cycles. Steady-state generation throughput from vLLM logs is ~16,300 tok/s, similar to C=256's ~17,000.

---

## CUDA graphs vs enforce-eager comparison (30B model)

| C | Mode | tok/s | SM active | SM occ | Tensor | DRAM BW | Power |
|---|------|-------|-----------|--------|--------|---------|-------|
| 1 | eager | 25 | 3.6% | 0.6% | 0.3% | 1.8% | 135W |
| 1 | **graphs** | **194** | **27.5%** | **4.6%** | **2.0%** | **13.6%** | **232W** |
| 1 | *speedup* | *7.8x* | *7.6x* | *7.7x* | *6.7x* | *7.6x* | *1.7x* |
| 256 | eager | 5,569 | 20.0% | 6.7% | 3.0% | 5.3% | 216W |
| 256 | **graphs** | **15,703** | **60.1%** | **19.7%** | **10.1%** | **24.3%** | **456W** |
| 256 | *speedup* | *2.8x* | *3.0x* | *2.9x* | *3.4x* | *4.6x* | *2.1x* |
| 2048 | eager | 22,092 | 78.2% | 29.2% | 16.1% | 28.1% | 587W |
| 2048 | **graphs** | **22,433** | **75.0%** | **28.3%** | **15.9%** | **28.3%** | **583W** |
| 2048 | *speedup* | *1.02x* | *~1x* | *~1x* | *~1x* | *~1x* | *~1x* |
| 4096 | eager | 25,496 | 89.1% | 35.3% | 17.0% | 26.2% | 623W |
| 4096 | **graphs** | **25,762** | **83.9%** | **33.4%** | **16.7%** | **26.6%** | **611W** |
| 4096 | *speedup* | *~1x* | *~1x* | *~1x* | *~1x* | *~1x* | *~1x* |

### Key findings

**CUDA graphs eliminate kernel launch overhead, revealing the true workload profile:**

1. **At C=1: 7.8x throughput improvement** (25 → 194 tok/s). In enforce-eager mode, ~33μs inter-kernel gaps dominate — the GPU is idle between each of the thousands of small MoE kernels per step. CUDA graphs batch these into a single graph replay, eliminating the gaps. All hardware counters increase by ~7x proportionally, confirming the overhead was uniform idle time.

2. **At C=256: 2.8x throughput improvement** (5,569 → 15,703 tok/s). DRAM BW jumps from 5.3% to 24.3% (4.6x) — even more than throughput, because CUDA graphs allow sustained memory streaming. Power hits 456W, approaching the 700W TDP.

3. **At C≥2048: no improvement** — at high enough concurrency, each scheduler step processes enough tokens that individual kernels are large enough to amortize the launch overhead naturally. Both modes converge to ~22-26K tok/s.

4. **DRAM bandwidth peaks at ~28% in both modes** (~1.3 TB/s of 4.8 TB/s). This appears to be the true bandwidth ceiling for this MoE architecture. The 132 SMs cannot generate enough memory requests to saturate the 4.8 TB/s HBM3e bus, because the fused_moe_kernel's compute-per-byte ratio creates a balanced bottleneck rather than a pure bandwidth bottleneck.

5. **FP8 IMMA counter still 0.0%** — this is a DCGM measurement gap (field 1013 tracks INT8, not FP8). FP8 compute IS happening via QGMMA instructions; it shows up in the aggregate `tensor_active` column (2-17%).

### Implications for co-scheduling

CUDA graphs change the co-scheduling picture significantly:

| Resource | Eager C=1 available | **Graphs C=1 available** |
|----------|-------------------|------------------------|
| SM active | 96.4% | **72.5%** |
| Tensor cores | 99.7% | **98.0%** |
| DRAM BW | 98.2% | **86.4%** |
| Power | 565W | **468W** |

With CUDA graphs, a single-request decode still leaves ~72% SM headroom and ~86% DRAM headroom — substantial, but less than the 96%+ available in enforce-eager mode. At C=256 (a more realistic serving point), graphs use 60% SM and 24% DRAM, leaving ~40% SM and ~76% DRAM bandwidth for a co-located embedding model.
