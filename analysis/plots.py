"""
Generate publication-quality figures for the peer-swap experiment report.

Figures:
  1. Cooperation rate trajectories (round-by-round, per condition)
  2. Bar chart: detection rates across conditions
  3. Bar chart: cooperation shifts (pre vs post swap)
  4. Heatmap or table: detection probe confidence scores
"""

import json
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)


CONDITION_LABELS = {
    "sham": "Sham (control)",
    "same_model_fresh": "Same model\n(fresh instance)",
    "same_family_different_size": "Same family\n(diff. size)",
    "different_provider": "Different\nprovider",
}

COLORS = {
    "sham": "#4C72B0",
    "same_model_fresh": "#55A868",
    "same_family_different_size": "#C44E52",
    "different_provider": "#8172B3",
}


def load_results(results_dir):
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


def load_game_logs(results_dir, condition):
    path = os.path.join(results_dir, f"{condition}_games.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def plot_cooperation_trajectories(results_dir, summary, fig_dir):
    """Round-by-round cooperation rates for Agent B (swapped), all conditions overlaid."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for cond in summary:
        logs = load_game_logs(results_dir, cond)
        if not logs:
            continue

        max_round = max(r["round"] for log in logs for r in log["rounds"])
        rounds = list(range(max_round + 1))
        means = []
        stds = []
        for rd in rounds:
            vals = []
            for log in logs:
                for r in log["rounds"]:
                    if r["round"] == rd:
                        vals.append(1.0 if r["choice_b"] == "COOPERATE" else 0.0)
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)

        means = np.array(means)
        stds = np.array(stds)
        label = CONDITION_LABELS.get(cond, cond)
        color = COLORS.get(cond, "#333333")
        ax.plot(rounds, means, marker="o", markersize=4, label=label, color=color, linewidth=1.5)
        ax.fill_between(rounds, means - stds, means + stds, alpha=0.15, color=color)

    swap_round = summary[list(summary.keys())[0]].get("swap_round", 10)
    ax.axvline(x=swap_round, color="red", linestyle="--", alpha=0.7, label=f"Swap point (round {swap_round})")

    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Cooperation Rate (Agent B)", fontsize=11)
    ax.set_title("Agent B Cooperation Rate Over Time", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(fig_dir, "cooperation_trajectories.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_detection_rates(summary, fig_dir):
    """Bar chart of explicit detection rates per condition."""
    conditions = list(summary.keys())
    rates = [summary[c]["stats"]["explicit_detection_rate"] for c in conditions]
    colors = [COLORS.get(c, "#333333") for c in conditions]
    labels = [CONDITION_LABELS.get(c, c) for c in conditions]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(len(conditions)), rates, color=colors, edgecolor="white", linewidth=0.5)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{rate:.0%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Explicit Detection Rate", fontsize=11)
    ax.set_title("Did Agent A Detect the Swap?", fontsize=13)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(fig_dir, "detection_rates.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cooperation_shifts(summary, fig_dir):
    """Grouped bar chart: pre vs post cooperation rates for both agents."""
    conditions = list(summary.keys())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for idx, (agent_label, shift_key, shifts_key) in enumerate([
        ("Agent B (swapped)", "agent_b_coop_shift_mean", "per_game_b_shifts"),
        ("Agent A (observer)", "agent_a_coop_shift_mean", "per_game_a_shifts"),
    ]):
        ax = axes[idx]
        x = np.arange(len(conditions))
        width = 0.35

        pre_rates = []
        post_rates = []
        for c in conditions:
            shifts = summary[c]["stats"].get(shifts_key, [])
            if shifts:
                pre_rates.append(np.mean([s["pre_swap_coop_rate"] for s in shifts]))
                post_rates.append(np.mean([s["post_swap_coop_rate"] for s in shifts]))
            else:
                pre_rates.append(0)
                post_rates.append(0)

        bars1 = ax.bar(x - width / 2, pre_rates, width, label="Pre-swap", color="#66c2a5", edgecolor="white")
        bars2 = ax.bar(x + width / 2, post_rates, width, label="Post-swap", color="#fc8d62", edgecolor="white")

        labels = [CONDITION_LABELS.get(c, c) for c in conditions]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Cooperation Rate" if idx == 0 else "")
        ax.set_title(agent_label, fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Cooperation Rate Before vs After Swap", fontsize=13, y=1.02)
    plt.tight_layout()

    path = os.path.join(fig_dir, "cooperation_shifts.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_agent_a_trajectory(results_dir, summary, fig_dir):
    """Round-by-round cooperation rates for Agent A (the observer)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for cond in summary:
        logs = load_game_logs(results_dir, cond)
        if not logs:
            continue

        max_round = max(r["round"] for log in logs for r in log["rounds"])
        rounds = list(range(max_round + 1))
        means = []
        for rd in rounds:
            vals = []
            for log in logs:
                for r in log["rounds"]:
                    if r["round"] == rd:
                        vals.append(1.0 if r["choice_a"] == "COOPERATE" else 0.0)
            means.append(np.mean(vals) if vals else 0)

        label = CONDITION_LABELS.get(cond, cond)
        color = COLORS.get(cond, "#333333")
        ax.plot(rounds, means, marker="s", markersize=4, label=label, color=color, linewidth=1.5)

    swap_round = summary[list(summary.keys())[0]].get("swap_round", 10)
    ax.axvline(x=swap_round, color="red", linestyle="--", alpha=0.7, label=f"Swap point (round {swap_round})")

    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Cooperation Rate (Agent A)", fontsize=11)
    ax.set_title("Observer Agent Cooperation Rate Over Time", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(fig_dir, "agent_a_trajectories.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate experiment figures")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--fig-dir", default="figures")
    args = parser.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)

    summary = load_results(args.results_dir)
    if not summary:
        print("No results found.")
        return

    print(f"\nGenerating figures for conditions: {list(summary.keys())}")

    plot_cooperation_trajectories(args.results_dir, summary, args.fig_dir)
    plot_detection_rates(summary, args.fig_dir)
    plot_cooperation_shifts(summary, args.fig_dir)
    plot_agent_a_trajectory(args.results_dir, summary, args.fig_dir)

    print(f"\nAll figures saved to: {args.fig_dir}/")


if __name__ == "__main__":
    main()
