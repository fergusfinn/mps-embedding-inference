"""
Embedding model benchmark.
Runs e5-large-v2 inference in a tight loop on a specified GPU and reports throughput.

Usage:
    CUDA_VISIBLE_DEVICES=1 python embed_benchmark.py --duration 60 --batch-size 32
"""
import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModel


def run_benchmark(duration, batch_size, seq_length):
    device = torch.device("cuda")

    print(f"Loading intfloat/e5-large-v2...")
    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-large-v2")
    model = AutoModel.from_pretrained("intfloat/e5-large-v2").to(device).eval()

    # Pre-tokenize a batch of dummy inputs
    texts = [f"query: This is a sample sentence number {i} for embedding benchmark testing purposes." for i in range(batch_size)]
    inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=seq_length, return_tensors="pt").to(device)

    vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"Model loaded. VRAM used: {vram_mb:.0f} MB")

    # Warm up
    print("Warming up...")
    for _ in range(5):
        with torch.no_grad():
            model(**inputs)
    torch.cuda.synchronize()

    print(f"Benchmarking for {duration}s, batch_size={batch_size}, seq_length={seq_length}...")
    total_sequences = 0
    start = time.monotonic()

    while time.monotonic() - start < duration:
        with torch.no_grad():
            model(**inputs)
        torch.cuda.synchronize()
        total_sequences += batch_size

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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-length", type=int, default=128)
    args = parser.parse_args()
    run_benchmark(args.duration, args.batch_size, args.seq_length)
