"""
Statistical analysis of experiment results.

Loads results JSON files and computes:
  - Per-condition cooperation rate shifts with confidence intervals
  - Detection accuracy (explicit probe)
  - Fisher's exact test: sham vs each real-swap condition
  - Round-by-round cooperation trajectories
"""

import json
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Placeholder — will implement once we have real data from Modal runs.

def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    args = parser.parse_args()

    summary_path = os.path.join(args.results_dir, "experiment_results.json")
    if not os.path.exists(summary_path):
        print(f"No results found at {summary_path}. Run experiments first.")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    print("\nLoaded results for conditions:", list(summary.keys()))
    for cond, data in summary.items():
        stats = data.get("stats", {})
        print(f"\n  {cond}:")
        print(f"    n_games:          {stats.get('n_games', 0)}")
        print(f"    detection_rate:   {stats.get('explicit_detection_rate', 0):.2%}")
        print(f"    B coop shift:     {stats.get('agent_b_coop_shift_mean', 0):+.3f}")
        print(f"    A coop shift:     {stats.get('agent_a_coop_shift_mean', 0):+.3f}")


if __name__ == "__main__":
    main()
