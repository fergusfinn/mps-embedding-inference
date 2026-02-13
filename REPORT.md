# Co-scheduling Generative and Embedding Models on a Single GPU via CUDA MPS

## 1. The Idea

LLM decode is memory-bandwidth-bound. Each decode step reads the full model weights from HBM to generate one token per sequence. The arithmetic intensity is low: the GPU's SMs spend most of their time waiting on memory. This is especially true for Mixture-of-Experts (MoE) models, which load expert weights from HBM but only compute through a small active subset (e.g. 8 of 128 experts per token).

Embedding inference is compute-bound. It runs dense matrix multiplications over full input sequences in a single forward pass, with high arithmetic intensity and high SM utilization per kernel.

These two workloads have complementary resource profiles. If we can run them simultaneously, the embedding model could use SM cycles that the generative model leaves idle, without competing for the same bottleneck resource.

## 2. The Deployment Question

In production, embedding and generative models are typically served on separate GPUs. This is simple but wasteful: the generative GPU's compute units sit partially idle during decode, and the embedding GPU may be underutilized between bursts of indexing traffic.

An alternative: co-locate both models on the same hardware. But on a single GPU, this is strictly worse than having two separate GPUs -- you're splitting one GPU's resources between two workloads, and each gets less than it would standalone. There's no free lunch on 1 GPU.

The interesting case is **two GPUs with TP=2 for both models**. Here the baseline is meaningful:

- **Baseline**: 2 GPUs, gen on GPU 0 (TP=1), embed on GPU 1 (TP=1)
- **Experiment**: 2 GPUs, gen across both (TP=2) + embed across both (TP=2) via MPS

With TP=2, gen gets 2x memory bandwidth (9.6 TB/s vs 4.8 TB/s), directly helping the bandwidth-bound MoE decode. It also gets 2x VRAM for KV cache (more concurrent sequences) and 2x SMs. If the per-GPU MPS interference from embed is small enough, the TP=2 gen throughput could exceed the TP=1 baseline -- a net win on gen *and* embed from the same 2 GPUs.

The single-GPU experiment below is a proof of concept: validating that MPS interference is low enough to justify the TP=2 experiment, and characterizing how much overhead co-location actually introduces.

### VRAM budget (single GPU)

| Component | VRAM |
|---|---|
| Qwen3-30B-A3B (FP8 weights) | ~30 GB |
| KV cache (gpu_mem_util=0.50) | ~40 GB |
| Qwen3-Embedding-8B (BF16 weights) | ~15 GB |
| **Total** | **~85 GB / 141 GB** |

## 3. Proof of Concept

### Hardware

- NVIDIA H200 SXM (141 GB HBM3e, 4.8 TB/s bandwidth, 700W TDP, 132 SMs)

### Models

- **Generative**: Qwen/Qwen3-30B-A3B (MoE, 128 experts, 8 active/token, FP8, served via vLLM)
- **Embedding**: Qwen/Qwen3-Embedding-8B (dense, BF16, direct transformers inference)

### Measurement

GPU metrics were collected via DCGM profiling counters (1-second resolution) and nvidia-smi dmon:

| Metric | DCGM Field | What it measures |
|---|---|---|
| SM Active | 1002 | Fraction of time any warp is scheduled on an SM |
| SM Occupancy | 1003 | Fraction of warp slots filled across all SMs |
| Tensor Core Active | 1004 | Fraction of time tensor cores are executing |
| DRAM Bandwidth | 1005 | Fraction of peak HBM bandwidth utilized |
| Power | dmon | GPU power draw (W) |

Note: DCGM field 1013 (tensor_imma_active) tracks INT8 IMMA, not FP8. Hopper FP8 uses QGMMA instructions which DCGM has no dedicated counter for. The aggregate tensor_active (field 1004) captures FP8 activity.

### 3.1 Resource Profiles: Gen vs Embed in Isolation

Each model was run alone with `gpu_mem_util=0.50` (half the GPU memory each). Gen was run at C=1024 with 30s warmup to saturate KV cache.

| Metric | Gen Only | Embed Only | Sum |
|---|---|---|---|
| Throughput | 15,204 tok/s | 348 seq/s | - |
| SM Active | 49% | 33% | 82% |
| SM Occupancy | 18% | 10% | 28% |
| Tensor Core | 10% | 11% | 21% |
| DRAM Bandwidth | 18% | 14% | 32% |
| Power | ~400W | ~320W | ~720W |

The sum of SM Active (82%) is under 100%, and DRAM bandwidth (32%) has large headroom. Power sums to ~720W, slightly above the 700W TDP -- a potential concern, though actual co-located power may differ from the arithmetic sum due to different thermal and power delivery dynamics.

### 3.2 Co-location Results

Four conditions were tested, each with 60s measurement after 30s KV-cache warmup:

| Condition | Gen (tok/s) | Gen delta | Embed (seq/s) | Embed delta |
|---|---|---|---|---|
| Gen only | 15,190 | baseline | - | - |
| Gen only (MPS daemon on) | 15,226 | +0.2% | - | - |
| Gen + Embed (MPS) | 14,818 | **-2.4%** | 248 | -29% vs standalone |
| Gen + Embed (no MPS) | 13,049 | **-14.1%** | 244 | -30% vs standalone |

### 3.3 Latency Impact

| Condition | p50 | p95 | p99 |
|---|---|---|---|
| Gen only | 32.4s | 38.6s | 40.7s |
| Gen only (MPS on) | 32.4s | 38.3s | 41.0s |
| Gen + Embed (MPS) | 33.2s | 40.3s | 42.4s |
| Gen + Embed (no MPS) | 37.9s | 47.4s | 50.1s |

These are end-to-end request latencies at C=1024 with max_tokens=512 (each request generates ~512 tokens).

### 3.4 GPU Metrics During Co-location

| Condition | SM Active | SM Occ | Tensor | DRAM BW | Power | Temp |
|---|---|---|---|---|---|---|
| Gen only | 50.9% | 19.3% | 10.0% | 17.4% | 473W | 61C |
| Gen only (MPS on) | 50.9% | 19.2% | 10.0% | 17.5% | 469W | 61C |
| Gen + Embed (MPS) | 73.0% | 26.2% | 17.3% | 27.1% | 587W | 66C |
| Gen + Embed (no MPS) | 65.5% | 22.8% | 15.2% | 23.4% | 508W | 63C |

### 3.5 Interpretation

**MPS has zero overhead.** Gen-only with and without the MPS daemon running are identical: 15,190 vs 15,226 tok/s, same latencies, same DCGM profile. The entire 2.4% gen throughput loss under MPS is from embedding contention, not MPS proxy overhead.

**MPS enables true parallel execution.** With MPS, combined SM Active reaches 73% (vs 51% gen-only) -- the embedding model's kernels genuinely run alongside gen kernels on different SMs. Without MPS, time-slicing gives only 65.5% SM Active with 14% gen throughput loss, because context switching between the two CUDA contexts is expensive.

**Power stays under TDP.** Co-located power with MPS is 587W, well under the 700W TDP. No thermal throttling observed (66C vs 61C baseline).

**The embedding model takes a larger relative hit.** Embed throughput drops ~30% from standalone (348 -> 248 seq/s) regardless of MPS vs time-slicing. This is expected: the gen model is the primary consumer of both SM time and memory bandwidth, and the embed model gets whatever resources remain.

### 3.6 Concurrency Sweep (Gen Only)

The MoE model's resource utilization scales with concurrent decode sequences. At low concurrency, the GPU is deeply underutilized:

| C | Throughput (tok/s) | SM Active | SM Occ | DRAM BW | Power |
|---|---|---|---|---|---|
| 1 | 194 | 27.5% | 4.6% | 13.6% | 232W |
| 256 | 15,703 | 60.1% | 19.7% | 24.3% | 456W |
| 1024 | 14,903 | 50.0% | 17.1% | 21.4% | 416W |
| 2048 | 22,433 | 75.0% | 28.3% | 28.3% | 583W |
| 4096 | 25,762 | 83.9% | 33.4% | 26.6% | 611W |

(CUDA graphs mode, `gpu_mem_util=0.85`)

DRAM bandwidth plateaus at ~28% of the H200's 4.8 TB/s peak. Even at C=4096, over 70% of memory bandwidth is unused, confirming the model is not bandwidth-saturated -- it is limited by other factors (KV cache capacity, scheduler overhead, expert routing).

## 4. TP=2 Experiment

Not yet conducted. This is where the real value proposition lives.

### The baseline

2 GPUs, each dedicated to one workload:
- GPU 0: gen model (TP=1), full 141 GB for weights + KV cache, 4.8 TB/s bandwidth
- GPU 1: embed model (TP=1), using ~15 GB of 141 GB

This is how you'd deploy today. GPU 1 is massively underutilized.

### The experiment

Same 2 GPUs, both workloads on both GPUs via TP=2 + MPS:
- Gen model: TP=2 across both GPUs, ~15 GB weights per GPU, KV cache split across both
- Embed model: TP=2 across both GPUs, ~7.5 GB weights per GPU
- MPS on each GPU enables parallel kernel execution between gen and embed

### Why this could be a net positive

Gen with TP=2 gets:
- **2x memory bandwidth** (9.6 TB/s total) -- directly helps the bandwidth-bound MoE decode
- **2x VRAM for KV cache** -- more concurrent sequences before saturation
- **2x SMs** (264 total) -- more parallel decode capacity

From the single-GPU experiment, MPS co-location costs gen ~2.4% throughput. If TP=2 gen throughput is more than 2.4% higher than TP=1 (which is likely, given the model is bandwidth-bound), the co-located TP=2 setup beats the dedicated-GPU baseline on gen throughput *while also serving embeddings*.

### What to measure

| Condition | GPUs | Gen | Embed |
|---|---|---|---|
| Baseline: separate GPUs | 2 | TP=1 on GPU 0 | TP=1 on GPU 1 |
| Co-located TP=2 + MPS | 2 | TP=2 on both | TP=2 on both |

Success: co-located gen throughput >= baseline gen throughput, with embed throughput as bonus.

## 5. Conclusion

Not yet written. Pending TP=2 results.
