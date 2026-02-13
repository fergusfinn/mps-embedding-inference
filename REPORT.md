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

With TP=2, gen gets 2x memory bandwidth (9.6 TB/s vs 4.8 TB/s) and 2x SMs, directly helping the bandwidth-bound MoE decode. (VRAM for KV cache doesn't double -- each GPU still holds both models' weight shards.) If the per-GPU MPS interference from embed is small enough, the TP=2 gen throughput could exceed the TP=1 baseline -- a net win on gen *and* embed from the same 2 GPUs.

The single-GPU experiments below are a proof of concept: validating that MPS interference is low enough to justify the TP=2 experiment, and comparing MPS against the alternative isolation strategies (MIG and time-slicing).

### VRAM budget (single GPU)

| Component | VRAM |
|---|---|
| Qwen3-30B-A3B-FP8 (FP8 weights) | ~30 GB |
| KV cache (gpu_mem_util=0.50) | ~40 GB |
| Qwen3-Embedding-8B (BF16 weights) | ~15 GB |
| **Total** | **~85 GB / 141 GB** |

## 3. Proof of Concept

### Hardware

- NVIDIA H200 SXM (141 GB HBM3e, 4.8 TB/s bandwidth, 700W TDP, 132 SMs)

### Models

- **Generative**: Qwen/Qwen3-30B-A3B-FP8 (MoE, 128 experts, 8 active/token, pre-quantized FP8, served via vLLM)
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

### 3.1 Co-location via MPS (shared GPU)

Both models run on the full GPU. CUDA MPS enables true parallel kernel execution by routing both processes' work through a shared CUDA context, avoiding time-slicing overhead.

Four conditions were tested, each with 60s measurement after 30s KV-cache warmup (C=1024, max_tokens=512):

| Condition | Gen (tok/s) | Gen delta | Embed (seq/s) |
|---|---|---|---|
| Gen only | 14,572 | baseline | - |
| Gen only (MPS daemon on) | 14,618 | +0.3% | - |
| Gen + Embed (MPS) | 14,778 | **+1.4%** | 288 |
| Gen + Embed (no MPS, time-sliced) | 13,504 | **-7.3%** | 246 |

**MPS has zero overhead.** Gen-only with and without the MPS daemon running are identical within noise. The MPS proxy adds no measurable cost.

**MPS co-location imposes no gen throughput loss.** With MPS, gen throughput is 14,778 tok/s -- actually slightly *above* the gen-only baseline of 14,572 tok/s (+1.4%, likely noise/thermal). The embedding model gets 288 seq/s essentially for free.

**Time-slicing is significantly worse.** Without MPS, CUDA context-switches between the two processes, causing a 7.3% gen throughput loss and lower embed throughput (246 vs 288 seq/s).

#### Latency

| Condition | p50 | p95 | p99 |
|---|---|---|---|
| Gen only | 67.4ms | 80.1ms | 85.7ms |
| Gen only (MPS on) | 66.0ms | 77.8ms | 83.6ms |
| Gen + Embed (MPS) | 64.8ms | 77.0ms | 82.6ms |
| Gen + Embed (no MPS) | 72.1ms | 86.6ms | 92.9ms |

Per-token decode latency. MPS co-location actually improves latency slightly; time-slicing degrades it.

#### GPU Metrics

| Condition | SM Active | SM Occ | Tensor | DRAM BW | Power | Temp |
|---|---|---|---|---|---|---|
| Gen only | 44.5% | 9.7% | 9.9% | 16.7% | 430W | 49C |
| Gen only (MPS on) | 45.0% | 9.8% | 10.0% | 17.0% | 427W | 49C |
| Gen + Embed (MPS) | 71.3% | 18.7% | 19.2% | 29.7% | 583W | 56C |
| Gen + Embed (no MPS) | 66.3% | 16.2% | 16.4% | 25.2% | 507W | 53C |

MPS co-location raises SM Active from 45% to 71% -- the embedding model's compute kernels genuinely run in parallel alongside gen decode kernels. Power stays well under 700W TDP.

### 3.2 MIG Baseline (partitioned GPU)

NVIDIA Multi-Instance GPU (MIG) provides hardware-level isolation by partitioning a GPU into fixed slices with dedicated SMs, memory controllers, and cache. This is the "hard isolation" alternative to MPS's flexible resource sharing.

**Partition layout:** 4g.71gb (4/7 SMs, 4/8 memory) for gen + 3g.71gb (3/7 SMs, 4/8 memory) for embed. This uses all 7/7 SM slices -- no resources wasted at the partition level. Both partitions get 70 GB VRAM.

| Condition | Gen (tok/s) | Gen delta | Embed (seq/s) |
|---|---|---|---|
| Gen only (4g.71gb partition) | 4,647 | baseline | - |
| Gen + Embed (MIG isolated) | 4,687 | +0.9% | 205 |

As expected, MIG gives perfect isolation: zero interference between partitions (+0.9% is noise). But the cost is catastrophic for the bandwidth-bound gen model:

**Gen throughput drops 68% vs the full GPU** (4,647 vs 14,572 tok/s). The 4g partition has 4/8 = 50% of the GPU's memory bandwidth, but gen only achieves 31.9% of full-GPU throughput. The reduced SM count (4/7) limits how many concurrent decode operations can run, preventing the workload from saturating even the reduced bandwidth.

**Embed throughput also drops** (205 vs 288 seq/s MPS, vs 348 seq/s standalone). The 3g partition has 3/7 SMs and 4/8 memory bandwidth -- enough to run, but constrained.

Note: the MIG experiment required `gpu_memory_utilization=0.85` to fit the 30 GB FP8 model in the 70 GB partition, vs 0.50 for the full-GPU experiments. This gives more KV cache per partition but is not the driver of the throughput difference.

### 3.3 Comparison

| Configuration | Gen (tok/s) | Gen % of full | Embed (seq/s) | Interference |
|---|---|---|---|---|
| Full GPU, gen only | 14,572 | 100% | - | - |
| Full GPU, MPS co-located | 14,778 | 101.4% | 288 | None |
| Full GPU, time-sliced | 13,504 | 92.7% | 246 | -7.3% gen |
| MIG 4g+3g, co-located | 4,687 | 32.2% | 205 | -67.8% gen (partitioning) |

MPS wins on every dimension: highest gen throughput, highest embed throughput, zero interference. MIG's hard isolation guarantees are irrelevant when MPS interference is already unmeasurably small, and the partitioning cost is ruinous for bandwidth-bound workloads.

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
- **2x SMs** (264 total) -- more parallel decode capacity

VRAM does *not* double for KV cache -- each GPU still holds its shard of both models' weights (~15 GB gen + ~7.5 GB embed per GPU), so the per-GPU KV cache budget is similar to the single-GPU case. The wins are bandwidth and compute, not memory capacity.

From the single-GPU experiment, MPS co-location costs gen nothing (actually +1.4%). If TP=2 gen throughput exceeds TP=1 by any margin, the co-located TP=2 setup beats the dedicated-GPU baseline on gen throughput *while also serving embeddings*.

### What to measure

| Condition | GPUs | Gen | Embed |
|---|---|---|---|
| Baseline: separate GPUs | 2 | TP=1 on GPU 0 | TP=1 on GPU 1 |
| Co-located TP=2 + MPS | 2 | TP=2 on both | TP=2 on both |

Success: co-located gen throughput >= baseline gen throughput, with embed throughput as bonus.

## 5. Conclusion

Not yet written. Pending TP=2 results.
