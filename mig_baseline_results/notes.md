# MIG Baseline Benchmark
Date: Sat Feb 14 15:18:19 UTC 2026

## GPU Configuration
MIG 4g.71gb (MIG-4f79e892-6606-5a13-81f5-89bec3a53ca6) — gen model
MIG 3g.71gb (MIG-8de4c802-0943-5681-bf2b-654efd76103d) — embed model

## Part 1: Gen Model (Qwen3-30B-A3B-FP8) on MIG 4g.71gb

### Gen test: default_c64
  vLLM args: --max-num-seqs 256
  concurrency=64, max_tokens=512, duration=60s
  **Result: 5832.8 tok/s, decode p50: tok/s/request tok/s/req**

### Gen test: default_c256
  vLLM args: --max-num-seqs 256
  concurrency=256, max_tokens=512, duration=60s
  **Result: 12016.7 tok/s, decode p50: tok/s/request tok/s/req**

### Gen test: default_c512
  vLLM args: --max-num-seqs 512
  concurrency=512, max_tokens=512, duration=60s
  **Result: 15007.4 tok/s, decode p50: tok/s/request tok/s/req**

### Gen test: eager_c256
  vLLM args: --max-num-seqs 256 --enforce-eager
  concurrency=256, max_tokens=512, duration=60s
  **Result: 5721.0 tok/s, decode p50: tok/s/request tok/s/req**

### Gen test: default_c256_tok256
  vLLM args: --max-num-seqs 256
  concurrency=256, max_tokens=256, duration=60s
  **Result: 11826.0 tok/s, decode p50: tok/s/request tok/s/req**

## Part 2: Embed Model (Qwen3-Embedding-8B) on MIG 3g.71gb

### Embed test: c64_b8
  vLLM args: --max-num-seqs 256
  concurrency=64, batch_size=8, duration=60s
  **Result: 480.4 seq/s**

### Embed test: c64_b32
  vLLM args: --max-num-seqs 256
  concurrency=64, batch_size=32, duration=60s
  **Result: 504.9 seq/s**

### Embed test: c128_b8
  vLLM args: --max-num-seqs 256
  concurrency=128, batch_size=8, duration=60s
  **Result: 488.8 seq/s**

### Embed test: c128_b32
  vLLM args: --max-num-seqs 256
  concurrency=128, batch_size=32, duration=60s
  **Result: 506.5 seq/s**

### Embed test: c256_b8
  vLLM args: --max-num-seqs 256
  concurrency=256, batch_size=8, duration=60s
  **Result: 488.7 seq/s**

### Embed test: eager_c128_b8
  vLLM args: --max-num-seqs 256 --enforce-eager
  concurrency=128, batch_size=8, duration=60s
  **Result: 493.5 seq/s**

## Summary
See individual result files in /home/ubuntu/mps-embedding-inference/mig_baseline_results/
