import pandas as pd
import glob
import os
import numpy as np

RESULTS_DIR = "benchmarks/real_world/results"

def analyze_run(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return None

    if df.empty:
        return None

    # Filter Success only
    df = df[df["status"] == "SUCCESS"]
    
    stats = {}
    
    # Latency Metrics
    stats["count"] = len(df)
    stats["latency_mean"] = df["latency_ms"].mean()
    stats["latency_std"] = df["latency_ms"].std()
    stats["latency_p50"] = df["latency_ms"].median()
    stats["latency_p90"] = df["latency_ms"].quantile(0.90)
    stats["latency_p99"] = df["latency_ms"].quantile(0.99)
    
    # TTFT Metrics
    stats["ttft_mean"] = df["ttft_ms"].mean()
    stats["ttft_p99"] = df["ttft_ms"].quantile(0.99)
    
    # Throughput (TPS)
    duration = df["timestamp"].max() - df["timestamp"].min()
    if duration > 0:
        stats["rps"] = len(df) / duration
        stats["tps"] = df["tokens_used"].sum() / duration
    else:
        stats["rps"] = 0
        stats["tps"] = 0

    # SLO Attainment (<500ms for Small, <5s for Large)
    # Assuming mixed workload, we check based on Tier column
    slo_met = 0
    for _, row in df.iterrows():
        limit = 5000 if row["tier"] == "large" else 500
        if row["latency_ms"] < limit:
            slo_met += 1
    stats["slo_attainment"] = (slo_met / len(df)) * 100 if len(df) > 0 else 0

    # Cost Proxy (Fraction of requests on Large Tier)
    # We infer "Large Tier" by worker_id if possible, or just request tier
    # Here we calculate routing adherence: Did large requests go to T4?
    # (Requires knowing T4 worker ID pattern, e.g., 'colab')
    
    return stats

def main():
    files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    print(f"Found {len(files)} log files.")
    
    all_stats = []
    for f in files:
        # Skip the aggregate report itself if it exists
        if "final_report" in f: continue
        
        s = analyze_run(f)
        if s:
            s["run_id"] = os.path.basename(f)
            # Extract Config ID (e.g., A1_Baseline_Small)
            s["config"] = "_".join(s["run_id"].split("_")[:-1])
            all_stats.append(s)
            
    if not all_stats:
        print("No valid data found.")
        return

    final_df = pd.DataFrame(all_stats)
    output_file = os.path.join(RESULTS_DIR, "final_report.csv")
    final_df.to_csv(output_file, index=False)
    
    print("\n=== Aggregate Results (Grouped by Config) ===")
    # Group by config and show mean of p99 and SLO
    summary = final_df.groupby("config")[["latency_p99", "slo_attainment", "rps"]].agg(["mean", "std"])
    print(summary)
    print(f"\n📄 Detailed report saved to {output_file}")

if __name__ == "__main__":
    main()