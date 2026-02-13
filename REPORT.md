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
| Qwen3-Embedding-8B (BF16 weights) | ~17 GB |
| **Total** | **~87 GB / 141 GB** |

## 3. Proof of Concept

### Hardware

- NVIDIA H200 SXM (141 GB HBM3e, 4.8 TB/s bandwidth, 700W TDP, 132 SMs)

### Models

- **Generative**: Qwen/Qwen3-30B-A3B-FP8 (MoE, 128 experts, 8 active/token, pre-quantized FP8, served via vLLM v0.15.1)
- **Embedding**: Qwen/Qwen3-Embedding-8B (dense, BF16, served via vLLM v0.15.1 with `--convert embed`)

### Setup

Both models served via `vllm/vllm-openai:v0.15.1` Docker containers. All experiments use the same configuration:

- Gen: `--gpu-memory-utilization 0.50 --max-model-len 4096 --max-num-seqs 1024`
- Embed: `--gpu-memory-utilization 0.30 --max-model-len 512 --convert embed`
- Benchmark: 30s KV-cache warmup, 60s measurement
- Gen load: C=1024 concurrent requests, max_tokens=512
- Embed load: C=64 concurrent requests, batch_size=8

For MPS, a sidecar Docker container runs the MPS daemon with a shared Docker volume for the pipe directory. All containers use `--ipc=host` for shared memory access.

### 3.1 Co-location via MPS (shared GPU)

Both models run on the full GPU. CUDA MPS enables true parallel kernel execution by routing both processes' work through a shared CUDA context, avoiding time-slicing overhead.

| Condition | Gen (tok/s) | Gen delta | Embed (seq/s) |
|---|---|---|---|
| Gen only | 16,091 | baseline | - |
| Embed only | - | - | 475 |
| Gen + Embed (MPS) | 16,212 | **+0.8%** | 479 |
| Gen + Embed (no MPS, time-sliced) | 15,934 | **-1.0%** | 472 |

**MPS co-location imposes no gen throughput loss.** Gen throughput is 16,212 tok/s with MPS co-location vs 16,091 tok/s gen-only (+0.8%, within noise). The embedding model gets 479 seq/s essentially for free.

**Time-slicing has a small but measurable cost.** Without MPS, CUDA time-slices between the two processes, causing a 1.0% gen throughput loss and slightly lower embed throughput (472 vs 479 seq/s).

#### Per-token decode latency

| Condition | p50 | p95 | p99 |
|---|---|---|---|
| Gen only | 58.7ms | 69.3ms | 74.7ms |
| Gen + Embed (MPS) | 58.9ms | 70.4ms | 75.7ms |
| Gen + Embed (no MPS) | 60.2ms | 72.3ms | 76.7ms |

MPS adds no latency. Time-slicing adds ~1.5ms at p50.

### 3.2 MIG baseline (partitioned GPU)

NVIDIA Multi-Instance GPU (MIG) provides hardware-level isolation by partitioning a GPU into fixed slices with dedicated SMs, memory controllers, and cache. This is the "hard isolation" alternative to MPS's flexible resource sharing.

**Partition layout:** 4g.71gb (4/7 SMs, 4/8 memory) for gen + 3g.71gb (3/7 SMs, 4/8 memory) for embed. This uses all 7/7 SM slices. Both partitions get ~70 GB VRAM.

MIG partitions are passed to Docker containers via `--gpus "device=MIG-<uuid>"`. Each container sees its partition as a plain GPU. Gen uses `--gpu-memory-utilization 0.85` to fit the 30 GB model in the 70 GB partition.

| Condition | Gen (tok/s) | Gen delta | Embed (seq/s) |
|---|---|---|---|
| Gen only (4g.71gb partition) | 4,641 | -71.2% vs full | - |
| Embed only (3g.71gb partition) | - | - | 479 |
| Gen + Embed (MIG isolated) | 4,640 | -71.2% vs full | 475 |

As expected, MIG gives **perfect isolation**: zero interference between partitions (4,641 vs 4,640 tok/s). But the cost is catastrophic for the bandwidth-bound gen model:

**Gen throughput drops 71% vs the full GPU** (4,641 vs 16,091 tok/s). The 4g partition has 4/8 = 50% of the GPU's memory bandwidth and 4/7 = 57% of SMs, but gen only achieves 28.8% of full-GPU throughput. The reduced SM count limits how many concurrent decode operations can run, preventing the workload from saturating even the reduced bandwidth.

**Embed throughput is identical across all conditions** (~475 seq/s standalone, 479 MPS, 475 MIG). The 3g MIG partition has the same throughput as the full GPU, indicating that at C=64 the embedding workload is bottlenecked by vLLM scheduling overhead rather than GPU resources.

#### Per-token decode latency (MIG)

| Condition | p50 | p95 | p99 |
|---|---|---|---|
| MIG gen only (4g) | 170.3ms | 219.6ms | 220.3ms |
| MIG gen + embed (4g+3g) | 170.5ms | 219.5ms | 220.3ms |

MIG partitioning triples per-token latency (170ms vs 59ms) due to the reduced compute capacity.

### 3.3 Comparison

| Configuration | Gen (tok/s) | Gen % of full | Embed (seq/s) | Gen interference |
|---|---|---|---|---|
| Full GPU, gen only | 16,091 | 100% | - | - |
| Full GPU, embed only | - | - | 475 | - |
| Full GPU, MPS co-located | 16,212 | 100.8% | 479 | None |
| Full GPU, time-sliced | 15,934 | 99.0% | 472 | -1.0% |
| MIG 4g, gen only | 4,641 | 28.8% | - | -71.2% (partitioning) |
| MIG 3g, embed only | - | - | 479 | - |
| MIG 4g+3g, co-located | 4,640 | 28.8% | 475 | -71.2% (partitioning) |

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

From the single-GPU experiment, MPS co-location costs gen nothing (+0.8%). If TP=2 gen throughput exceeds TP=1 by any margin, the co-located TP=2 setup beats the dedicated-GPU baseline on gen throughput *while also serving embeddings*.

### What to measure

| Condition | GPUs | Gen | Embed |
|---|---|---|---|
| Baseline: separate GPUs | 2 | TP=1 on GPU 0 | TP=1 on GPU 1 |
| Co-located TP=2 + MPS | 2 | TP=2 on both | TP=2 on both |

Success: co-located gen throughput >= baseline gen throughput, with embed throughput as bonus.

## 5. Conclusion

Not yet written. Pending TP=2 results.
