"""
Statistical analysis of experiment results.

Loads per-condition summary/game-log JSONs and computes:
  - Per-condition cooperation rate shifts with confidence intervals
  - Detection accuracy (explicit probe)
  - Fisher's exact test: sham vs each real-swap condition
  - Round-by-round cooperation trajectories
"""

import json
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(results_dir: str) -> dict:
    summary = {}
    for fname in os.listdir(results_dir):
        if fname.endswith("_summary.json"):
            cond = fname.replace("_summary.json", "")
            with open(os.path.join(results_dir, fname)) as f:
                summary[cond] = json.load(f)
    if not summary:
        exp_path = os.path.join(results_dir, "experiment_results.json")
        if os.path.exists(exp_path):
            with open(exp_path) as f:
                summary = json.load(f)
    return summary


def load_game_logs(results_dir: str, condition: str) -> list:
    path = os.path.join(results_dir, f"{condition}_games.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def round_by_round_coop(game_logs: list, agent_key: str = "choice_b") -> dict:
    """Compute mean cooperation rate per round across all games."""
    if not game_logs:
        return {}
    max_round = max(r["round"] for log in game_logs for r in log["rounds"])
    per_round = {}
    for rd in range(max_round + 1):
        choices = []
        for log in game_logs:
            for r in log["rounds"]:
                if r["round"] == rd:
                    choices.append(1.0 if r[agent_key] == "COOPERATE" else 0.0)
        if choices:
            per_round[rd] = {
                "mean": float(np.mean(choices)),
                "std": float(np.std(choices)),
                "n": len(choices),
            }
    return per_round


def fishers_exact_detection(sham_detections, swap_detections):
    """Fisher's exact test on detection rates."""
    try:
        from scipy.stats import fisher_exact
    except ImportError:
        return None

    sham_yes = sum(1 for d in sham_detections if d.get("explicitly_detected"))
    sham_no = len(sham_detections) - sham_yes
    swap_yes = sum(1 for d in swap_detections if d.get("explicitly_detected"))
    swap_no = len(swap_detections) - swap_yes

    table = [[sham_yes, sham_no], [swap_yes, swap_no]]
    odds, p = fisher_exact(table)
    return {"odds_ratio": float(odds), "p_value": float(p), "table": table}


def analyze(results_dir: str):
    summary = load_results(results_dir)
    if not summary:
        print("No results found. Run experiments first.")
        return

    print("\n" + "=" * 65)
    print("STATISTICAL ANALYSIS")
    print("=" * 65)

    conditions = list(summary.keys())
    print(f"\nConditions: {conditions}")

    print(f"\n{'Condition':<30} {'N':>3} {'Detect%':>8} {'B_shift':>8} {'A_shift':>8} {'Conf':>5}")
    print("-" * 65)
    for cond in conditions:
        s = summary[cond]["stats"]
        n = s["n_games"]
        det = s["explicit_detection_rate"]
        bs = s["agent_b_coop_shift_mean"]
        ashift = s["agent_a_coop_shift_mean"]
        conf = s.get("mean_confidence_same")
        conf_str = f"{conf:.1f}" if conf else "N/A"
        print(f"{cond:<30} {n:>3} {det:>7.0%} {bs:>+7.3f} {ashift:>+7.3f} {conf_str:>5}")

    # Round-by-round analysis
    print("\n\nROUND-BY-ROUND COOPERATION (Agent B — swapped agent)")
    print("-" * 65)
    all_trajectories = {}
    for cond in conditions:
        logs = load_game_logs(results_dir, cond)
        if logs:
            traj = round_by_round_coop(logs, "choice_b")
            all_trajectories[cond] = traj
            swap_round = summary[cond].get("swap_round", 10)
            rounds = sorted(traj.keys())
            pre_vals = [traj[r]["mean"] for r in rounds if r < swap_round]
            post_vals = [traj[r]["mean"] for r in rounds if r >= swap_round]
            print(f"\n  {cond}:")
            print(f"    Pre-swap mean:  {np.mean(pre_vals):.3f} (rounds 0-{swap_round-1})")
            print(f"    Post-swap mean: {np.mean(post_vals):.3f} (rounds {swap_round}+)")

    # Fisher's exact test
    if "sham" in summary:
        sham_det = summary["sham"]["stats"].get("per_game_detections", [])
        print("\n\nFISHER'S EXACT TEST (sham vs each swap condition)")
        print("-" * 65)
        for cond in conditions:
            if cond == "sham":
                continue
            swap_det = summary[cond]["stats"].get("per_game_detections", [])
            result = fishers_exact_detection(sham_det, swap_det)
            if result:
                sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else "ns"
                print(f"  sham vs {cond}: OR={result['odds_ratio']:.2f}, p={result['p_value']:.4f} {sig}")
            else:
                print(f"  sham vs {cond}: (scipy not available)")

    # Detection probe details
    print("\n\nDETECTION PROBE DETAILS")
    print("-" * 65)
    for cond in conditions:
        dets = summary[cond]["stats"].get("per_game_detections", [])
        print(f"\n  {cond}:")
        for i, d in enumerate(dets):
            det_str = "YES" if d.get("explicitly_detected") else "NO" if d.get("explicitly_detected") is False else "?"
            conf = d.get("confidence_same_entity", "?")
            cr = d.get("reported_change_round", "-")
            reason = d.get("reasoning", "")[:80]
            print(f"    Game {i}: detected={det_str}, conf={conf}/5, change_round={cr}")
            if reason:
                print(f"             {reason}")

    # Save analysis
    analysis_out = {
        "conditions": conditions,
        "summary_table": {},
        "trajectories": {},
    }
    for cond in conditions:
        s = summary[cond]["stats"]
        analysis_out["summary_table"][cond] = {
            "n_games": s["n_games"],
            "detection_rate": s["explicit_detection_rate"],
            "b_coop_shift": s["agent_b_coop_shift_mean"],
            "a_coop_shift": s["agent_a_coop_shift_mean"],
            "mean_confidence": s.get("mean_confidence_same"),
        }
    for cond, traj in all_trajectories.items():
        analysis_out["trajectories"][cond] = {
            str(r): v for r, v in traj.items()
        }

    out_path = os.path.join(results_dir, "analysis_output.json")
    with open(out_path, "w") as f:
        json.dump(analysis_out, f, indent=2)
    print(f"\n\nAnalysis saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    args = parser.parse_args()
    analyze(args.results_dir)
