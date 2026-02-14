"""
Analyze inter-kernel gaps from nsys SQLite traces.

Compares gap distributions between two conditions (e.g., MPS vs time-sliced)
to test whether CUDA graph replay is interrupted by interleaved embed kernels.

Usage:
    python analyze_graph_gaps.py mps_trace.sqlite ts_trace.sqlite
"""
import sqlite3
import sys
import numpy as np


def get_kernel_gaps(db_path, steady_state_fraction=0.5):
    """Extract inter-kernel gaps from an nsys SQLite database.

    Returns gaps (in nanoseconds) between consecutive GPU kernels on the same device,
    filtered to the last `steady_state_fraction` of the trace (skip warmup/loading).
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    # Get all kernel start/end times, ordered by start time
    # CUPTI_ACTIVITY_KIND_KERNEL has: start, end, deviceId, demangledName
    try:
        rows = conn.execute("""
            SELECT start, end, deviceId, demangledName
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            ORDER BY start
        """).fetchall()
    except Exception as e:
        print(f"  Error querying {db_path}: {e}")
        # Try alternative table name
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        print(f"  Available tables: {[t[0] for t in tables]}")
        conn.close()
        return None, None, None

    conn.close()

    if not rows:
        print(f"  No kernels found in {db_path}")
        return None, None, None

    print(f"  {db_path}: {len(rows)} total kernels")

    # Filter to steady state (last fraction of trace by time)
    all_starts = [r[0] for r in rows]
    t_min, t_max = min(all_starts), max(all_starts)
    t_range = t_max - t_min
    t_cutoff = t_min + t_range * (1.0 - steady_state_fraction)

    steady_rows = [r for r in rows if r[0] >= t_cutoff]
    print(f"  Steady-state kernels (last {steady_state_fraction*100:.0f}%): {len(steady_rows)}")

    # Compute gaps between consecutive kernels on device 0
    device_rows = [r for r in steady_rows if r[2] == 0]
    device_rows.sort(key=lambda r: r[0])

    gaps = []
    for i in range(1, len(device_rows)):
        prev_end = device_rows[i-1][1]
        curr_start = device_rows[i][0]
        gap = curr_start - prev_end
        if gap >= 0:  # skip overlapping kernels (concurrent streams)
            gaps.append(gap)

    gaps = np.array(gaps)
    kernel_names = [r[3] for r in device_rows]
    durations = np.array([r[1] - r[0] for r in device_rows])

    return gaps, kernel_names, durations


def print_gap_stats(label, gaps):
    """Print gap distribution statistics."""
    if gaps is None or len(gaps) == 0:
        print(f"  {label}: no gaps to analyze")
        return

    print(f"\n  {label}: {len(gaps)} inter-kernel gaps")
    print(f"    Mean:   {np.mean(gaps):.0f} ns ({np.mean(gaps)/1000:.1f} us)")
    print(f"    Median: {np.median(gaps):.0f} ns ({np.median(gaps)/1000:.1f} us)")
    print(f"    p95:    {np.percentile(gaps, 95):.0f} ns ({np.percentile(gaps, 95)/1000:.1f} us)")
    print(f"    p99:    {np.percentile(gaps, 99):.0f} ns ({np.percentile(gaps, 99)/1000:.1f} us)")
    print(f"    Max:    {np.max(gaps):.0f} ns ({np.max(gaps)/1000:.1f} us)")
    print(f"    Total gap time: {np.sum(gaps)/1e6:.1f} ms")

    # Distribution buckets
    buckets = [
        ("0-1 us", 0, 1000),
        ("1-10 us", 1000, 10000),
        ("10-100 us", 10000, 100000),
        ("100us-1ms", 100000, 1000000),
        (">1 ms", 1000000, float('inf')),
    ]
    print(f"    Distribution:")
    for name, lo, hi in buckets:
        count = np.sum((gaps >= lo) & (gaps < hi))
        pct = count / len(gaps) * 100
        time_ns = np.sum(gaps[(gaps >= lo) & (gaps < hi)])
        print(f"      {name:>12s}: {count:>8d} ({pct:>5.1f}%)  total: {time_ns/1e6:>8.1f} ms")


def print_kernel_stats(label, durations):
    """Print kernel duration statistics."""
    if durations is None or len(durations) == 0:
        return

    print(f"\n  {label}: kernel duration stats")
    print(f"    Count:  {len(durations)}")
    print(f"    Mean:   {np.mean(durations)/1000:.1f} us")
    print(f"    Median: {np.median(durations)/1000:.1f} us")
    print(f"    Total:  {np.sum(durations)/1e6:.1f} ms")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <mps_trace.sqlite> <ts_trace.sqlite>")
        sys.exit(1)

    mps_db = sys.argv[1]
    ts_db = sys.argv[2]

    print("=== Inter-kernel gap analysis: MPS vs time-sliced (CUDA graphs ON) ===\n")

    print("Loading MPS trace...")
    mps_gaps, mps_names, mps_durations = get_kernel_gaps(mps_db)

    print("\nLoading time-sliced trace...")
    ts_gaps, ts_names, ts_durations = get_kernel_gaps(ts_db)

    print("\n" + "="*70)
    print_gap_stats("MPS", mps_gaps)
    print_gap_stats("Time-sliced", ts_gaps)

    print_kernel_stats("MPS", mps_durations)
    print_kernel_stats("Time-sliced", ts_durations)

    # Direct comparison
    if mps_gaps is not None and ts_gaps is not None and len(mps_gaps) > 0 and len(ts_gaps) > 0:
        print(f"\n{'='*70}")
        print(f"=== COMPARISON ===")
        mps_mean = np.mean(mps_gaps)
        ts_mean = np.mean(ts_gaps)
        print(f"  Mean gap: MPS {mps_mean:.0f} ns vs TS {ts_mean:.0f} ns  ({(mps_mean/ts_mean - 1)*100:+.1f}%)")

        mps_median = np.median(mps_gaps)
        ts_median = np.median(ts_gaps)
        print(f"  Median gap: MPS {mps_median:.0f} ns vs TS {ts_median:.0f} ns  ({(mps_median/ts_median - 1)*100:+.1f}%)")

        mps_total = np.sum(mps_gaps)
        ts_total = np.sum(ts_gaps)
        print(f"  Total gap time: MPS {mps_total/1e6:.1f} ms vs TS {ts_total/1e6:.1f} ms  ({(mps_total/ts_total - 1)*100:+.1f}%)")

        mps_p95 = np.percentile(mps_gaps, 95)
        ts_p95 = np.percentile(ts_gaps, 95)
        print(f"  p95 gap: MPS {mps_p95:.0f} ns vs TS {ts_p95:.0f} ns  ({(mps_p95/ts_p95 - 1)*100:+.1f}%)")

        if mps_durations is not None and ts_durations is not None:
            mps_kern_total = np.sum(mps_durations)
            ts_kern_total = np.sum(ts_durations)
            print(f"\n  Total kernel time: MPS {mps_kern_total/1e6:.1f} ms vs TS {ts_kern_total/1e6:.1f} ms  ({(mps_kern_total/ts_kern_total - 1)*100:+.1f}%)")

            mps_overhead = mps_total / (mps_kern_total + mps_total) * 100
            ts_overhead = ts_total / (ts_kern_total + ts_total) * 100
            print(f"  Gap as % of wall time: MPS {mps_overhead:.1f}% vs TS {ts_overhead:.1f}%")


if __name__ == "__main__":
    main()
