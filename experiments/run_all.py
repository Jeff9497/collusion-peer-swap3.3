"""
Run the full 4-condition experimental matrix.

For each condition (sham, same_model_fresh, same_family_different_size,
different_provider), runs N games and collects per-game logs + aggregated
statistics.

Usage:
    python experiments/run_all.py --seeds 5 --rounds 20 --swap-round 10
    python experiments/run_all.py --seeds 30 --rounds 25 --swap-round 12 --backend modal
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents import Agent
from src.game import run_game
from src.swap_hook import make_sham_swap, make_model_swap, make_same_model_swap
from src.metrics import aggregate_condition_results
from src.config import CONDITIONS, MODELS


def create_call_fn(model_key: str, backend: str = "mock"):
    """
    Create a call function for the given model and backend.

    Args:
        model_key: Key into MODELS dict (e.g. "qwen_14b")
        backend: "mock" for local testing, "modal" for real inference
    """
    if backend == "mock":
        from src.mock_llm import MockTitForTat, MockAlwaysDefect, MockGrimTrigger
        strategy_map = {
            "qwen_14b": MockTitForTat(),
            "qwen_7b": MockAlwaysDefect(),
            "mistral_7b": MockAlwaysDefect(),
            "llama_8b": MockGrimTrigger(),
        }
        return strategy_map.get(model_key, MockTitForTat())
    elif backend == "modal":
        raise NotImplementedError(
            "Modal backend not yet implemented. "
            "Use --backend mock for local testing."
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def run_condition(
    condition_name: str,
    condition_cfg: dict,
    n_seeds: int,
    total_rounds: int,
    swap_round: int,
    backend: str,
) -> dict:
    """Run all games for one experimental condition."""
    print(f"\n--- Condition: {condition_name} ---")
    print(f"    {condition_cfg['description']}")

    game_logs = []

    for seed in range(n_seeds):
        # Create agents
        a_key = condition_cfg["agent_a"]
        b_pre_key = condition_cfg["agent_b_pre"]
        b_post_key = condition_cfg["agent_b_post"]

        agent_a = Agent(
            name="Alice",
            model_id=MODELS.get(a_key, a_key),
            call_fn=create_call_fn(a_key, backend),
            role="observer",
        )
        agent_b = Agent(
            name="Bob",
            model_id=MODELS.get(b_pre_key, b_pre_key),
            call_fn=create_call_fn(b_pre_key, backend),
            role="swapped",
        )

        # Create swap function based on condition
        if condition_name == "sham":
            swap_fn = make_sham_swap()
        elif condition_name == "same_model_fresh":
            swap_fn = make_same_model_swap(
                MODELS.get(b_post_key, b_post_key),
                create_call_fn(b_post_key, backend),
            )
        else:
            swap_fn = make_model_swap(
                MODELS.get(b_post_key, b_post_key),
                create_call_fn(b_post_key, backend),
            )

        game_log = run_game(
            agent_a=agent_a,
            agent_b=agent_b,
            total_rounds=total_rounds,
            swap_round=swap_round,
            swap_fn=swap_fn,
            seed=seed,
        )
        game_logs.append(game_log)

    # Aggregate results
    stats = aggregate_condition_results(game_logs, swap_round)

    return {
        "condition": condition_name,
        "config": condition_cfg,
        "n_seeds": n_seeds,
        "total_rounds": total_rounds,
        "swap_round": swap_round,
        "stats": stats,
        "game_logs": game_logs,
    }


def main():
    parser = argparse.ArgumentParser(description="Run full 4-condition experiment")
    parser.add_argument("--seeds", type=int, default=5, help="Games per condition")
    parser.add_argument("--rounds", type=int, default=20, help="Rounds per game")
    parser.add_argument("--swap-round", type=int, default=10, help="Swap round")
    parser.add_argument("--backend", type=str, default="mock",
                        choices=["mock", "modal"], help="LLM backend")
    parser.add_argument("--conditions", type=str, nargs="*", default=None,
                        help="Specific conditions to run (default: all)")
    args = parser.parse_args()

    conditions = args.conditions or list(CONDITIONS.keys())

    print(f"\n{'='*60}")
    print("COLLUSION PEER SWAP EXPERIMENT")
    print(f"{'='*60}")
    print(f"  Conditions:  {', '.join(conditions)}")
    print(f"  Seeds:       {args.seeds} per condition")
    print(f"  Rounds:      {args.rounds}")
    print(f"  Swap round:  {args.swap_round}")
    print(f"  Backend:     {args.backend}")
    print(f"{'='*60}")

    t_start = time.time()
    all_results = {}

    for cond_name in conditions:
        cond_cfg = CONDITIONS[cond_name]
        result = run_condition(
            cond_name, cond_cfg, args.seeds, args.rounds, args.swap_round, args.backend
        )
        all_results[cond_name] = result

        # Print condition summary
        stats = result["stats"]
        print(f"\n    Results ({cond_name}):")
        print(f"      Detection rate:     {stats['explicit_detection_rate']:.2f}")
        print(f"      Agent B coop shift: {stats['agent_b_coop_shift_mean']:+.2f}")
        print(f"      Agent A coop shift: {stats['agent_a_coop_shift_mean']:+.2f}")
        if stats["mean_confidence_same"] is not None:
            print(f"      Mean confidence:    {stats['mean_confidence_same']:.1f}/5")

    elapsed = time.time() - t_start

    # --- Summary table ---
    print(f"\n\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"{'Condition':<28} {'Detect%':>8} {'B shift':>8} {'A shift':>8} {'Conf':>5}")
    print("-" * 60)

    for cond_name, result in all_results.items():
        s = result["stats"]
        conf = f"{s['mean_confidence_same']:.1f}" if s["mean_confidence_same"] else "N/A"
        print(
            f"{cond_name:<28} {s['explicit_detection_rate']:>7.0%} "
            f"{s['agent_b_coop_shift_mean']:>+7.2f} "
            f"{s['agent_a_coop_shift_mean']:>+7.2f} "
            f"{conf:>5}"
        )

    print(f"\nTotal time: {elapsed:.1f}s")

    # --- Save results ---
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "experiment_results.json")

    # Strip game_logs for summary file (they're big)
    summary = {}
    for cond_name, result in all_results.items():
        summary[cond_name] = {
            k: v for k, v in result.items() if k != "game_logs"
        }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to: {out_path}")

    # Save full logs per condition
    for cond_name, result in all_results.items():
        cond_path = os.path.join("results", f"{cond_name}_games.json")
        with open(cond_path, "w") as f:
            json.dump(result["game_logs"], f, indent=2, default=str)
        print(f"  {cond_name} logs: {cond_path}")


if __name__ == "__main__":
    main()
