"""
Embedding model benchmark.
Runs Qwen3-Embedding-8B inference in a tight loop and reports throughput.

Usage:
    CUDA_VISIBLE_DEVICES=0 python embed_benchmark.py --duration 60 --batch-size 8
"""
import argparse
import signal
import sys
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Global state for signal handler
_benchmark_state = {"total_sequences": 0, "start": 0, "batch_size": 0}

def _signal_handler(signum, frame):
    """Print partial results on SIGTERM/SIGINT."""
    s = _benchmark_state
    if s["start"] > 0 and s["total_sequences"] > 0:
        wall_time = time.monotonic() - s["start"]
        print(f"\n=== Embedding Benchmark Results (interrupted after {wall_time:.1f}s) ===", flush=True)
        print(f"Total sequences: {s['total_sequences']}", flush=True)
        print(f"Throughput: {s['total_sequences'] / wall_time:.1f} sequences/sec", flush=True)
        print(f"Batch size: {s['batch_size']}", flush=True)
    sys.exit(0)

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


def last_token_pool(last_hidden_states, attention_mask):
    """Pool embeddings from the last non-padding token (decoder-style models)."""
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def run_benchmark(duration, batch_size, seq_length, model_name, dtype):
    device = torch.device("cuda")
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    print(f"Loading {model_name} ({dtype})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    )
    print(f"Moving model to GPU...", flush=True)
    model = model.to(device).eval()

    # Pre-tokenize a batch of dummy inputs
    texts = [
        f"Instruct: Retrieve relevant documents\nQuery: This is sample sentence number {i} for embedding benchmark."
        for i in range(batch_size)
    ]
    inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=seq_length, return_tensors="pt"
    ).to(device)

    vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"Model loaded. VRAM used: {vram_mb:.0f} MB", flush=True)
    print(f"Input shape: {inputs['input_ids'].shape}", flush=True)

    # Warm up
    print("Warming up...", flush=True)
    for _ in range(5):
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
    torch.cuda.synchronize()

    print(f"Benchmarking for {duration}s, batch_size={batch_size}, seq_length={seq_length}...", flush=True)
    total_sequences = 0
    start = time.monotonic()
    _benchmark_state.update({"start": start, "batch_size": batch_size})

    while time.monotonic() - start < duration:
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        torch.cuda.synchronize()
        total_sequences += batch_size
        _benchmark_state["total_sequences"] = total_sequences

    wall_time = time.monotonic() - start
    print(f"\n=== Embedding Benchmark Results ===")
    print(f"Duration: {wall_time:.1f}s")
    print(f"Total sequences: {total_sequences}")
    print(f"Throughput: {total_sequences / wall_time:.1f} sequences/sec")
    print(f"Batch size: {batch_size}, Seq length: {seq_length}")
    print(f"VRAM used: {torch.cuda.memory_allocated() / 1024 / 1024:.0f} MB")
    print(f"VRAM reserved: {torch.cuda.memory_reserved() / 1024 / 1024:.0f} MB")
    return total_sequences / wall_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    args = parser.parse_args()
    run_benchmark(args.duration, args.batch_size, args.seq_length, args.model, args.dtype)
