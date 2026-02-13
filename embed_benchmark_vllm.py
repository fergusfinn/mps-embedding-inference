"""
Embedding model benchmark client.
Sends concurrent requests to a vLLM embedding server and measures throughput.

Usage:
    python embed_benchmark_vllm.py --base-url http://localhost:8200 --duration 60
"""
import argparse
import asyncio
import time
import aiohttp


async def embed_request(session, base_url, texts, model):
    """Send a single embedding request and return sequence count + duration."""
    payload = {
        "model": model,
        "input": texts,
    }
    t0 = time.monotonic()
    async with session.post(f"{base_url}/v1/embeddings", json=payload) as resp:
        data = await resp.json()
        elapsed = time.monotonic() - t0
        n_sequences = len(data.get("data", []))
        return n_sequences, elapsed


async def run_benchmark(base_url, duration, concurrency, batch_size, model):
    """Run continuous requests for `duration` seconds with `concurrency` workers."""
    texts = [
        f"Instruct: Retrieve relevant documents\nQuery: This is sample sentence number {i} for embedding benchmark."
        for i in range(batch_size)
    ]

    total_sequences = 0
    total_requests = 0
    request_stats = []
    start = time.monotonic()

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warm up
        print("Warming up...")
        await embed_request(session, base_url, texts[:1], model)

        print(f"Benchmarking for {duration}s with {concurrency} concurrent requests, batch_size={batch_size}...")
        start = time.monotonic()

        async def worker():
            nonlocal total_sequences, total_requests
            while time.monotonic() - start < duration:
                seqs, elapsed = await embed_request(session, base_url, texts, model)
                total_sequences += seqs
                total_requests += 1
                request_stats.append((seqs, elapsed))

        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await asyncio.gather(*workers)

    wall_time = time.monotonic() - start
    print(f"\n=== Embedding Benchmark Results ===")
    print(f"Duration: {wall_time:.1f}s")
    print(f"Total requests: {total_requests}")
    print(f"Total sequences: {total_sequences}")
    print(f"Throughput: {total_sequences / wall_time:.1f} sequences/sec")
    print(f"Batch size: {batch_size}")

    if request_stats:
        latencies = sorted([e for _, e in request_stats])
        n = len(latencies)
        print(f"\n--- Per-request latency ---")
        print(f"  p50: {latencies[n//2]*1000:.1f}ms  p95: {latencies[int(n*0.95)]*1000:.1f}ms  p99: {latencies[int(n*0.99)]*1000:.1f}ms")

    return total_sequences / wall_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8200")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.base_url, args.duration, args.concurrency, args.batch_size, args.model))
