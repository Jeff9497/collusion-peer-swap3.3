"""
Pilot test: run one game with mock LLM responses to verify the full pipeline.

Tests the swap hook by starting agent B with TitForTat strategy, then
swapping to AlwaysDefect at the swap round. Agent A (GrimTrigger) should
detect the behavioral change.

Usage:
    python experiments/pilot.py
    python experiments/pilot.py --rounds 20 --swap-round 10
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents import Agent
from src.game import run_game
from src.swap_hook import make_model_swap
from src.mock_llm import (
    MockTitForTat, MockAlwaysDefect, MockAlwaysCooperate,
    MockGrimTrigger, MockRandom,
)
from src.metrics import (
    cooperation_rate_shift, parse_detection_probe, aggregate_condition_results,
)


def run_pilot(total_rounds: int = 20, swap_round: int = 10):
    """Run a single pilot game with mock agents and print results."""

    print(f"\n{'='*60}")
    print("PILOT TEST: Swap Hook Verification")
    print(f"{'='*60}")
    print(f"  Rounds:      {total_rounds}")
    print(f"  Swap round:  {swap_round}")
    print(f"  Agent A:     GrimTrigger (observer)")
    print(f"  Agent B:     TitForTat -> AlwaysDefect (swapped)")
    print(f"{'='*60}\n")

    # --- Create agents ---
    mock_tft = MockTitForTat()
    mock_defect = MockAlwaysDefect()
    mock_grim = MockGrimTrigger()

    agent_a = Agent(
        name="Alice",
        model_id="mock/grim_trigger",
        call_fn=mock_grim,
        role="observer",
    )
    agent_b = Agent(
        name="Bob",
        model_id="mock/tit_for_tat",
        call_fn=mock_tft,
        role="swapped",
    )

    # --- Create swap function ---
    swap_fn = make_model_swap("mock/always_defect", mock_defect)

    # --- Run game ---
    game_log = run_game(
        agent_a=agent_a,
        agent_b=agent_b,
        total_rounds=total_rounds,
        swap_round=swap_round,
        swap_fn=swap_fn,
        seed=0,
    )

    # --- Print round-by-round results ---
    print("Round-by-round results:")
    print(f"{'Round':>5}  {'A choice':>10}  {'B choice':>10}  {'A pay':>6}  {'B pay':>6}  {'Swapped':>7}")
    print("-" * 55)

    for r in game_log["rounds"]:
        marker = " <-- SWAP" if r["round"] == swap_round else ""
        print(
            f"{r['round']+1:>5}  {r['choice_a']:>10}  {r['choice_b']:>10}  "
            f"{r['payoff_a']:>6.1f}  {r['payoff_b']:>6.1f}  "
            f"{'yes' if r['post_swap'] else 'no':>7}{marker}"
        )

    # --- Cooperation rate analysis ---
    print(f"\n{'='*60}")
    print("COOPERATION RATE ANALYSIS")
    print(f"{'='*60}")

    b_shift = cooperation_rate_shift(game_log["rounds"], swap_round, "choice_b")
    a_shift = cooperation_rate_shift(game_log["rounds"], swap_round, "choice_a")

    print(f"\n  Agent B (swapped):")
    print(f"    Pre-swap coop rate:  {b_shift['pre_swap_coop_rate']:.2f} ({b_shift['pre_n']} rounds)")
    print(f"    Post-swap coop rate: {b_shift['post_swap_coop_rate']:.2f} ({b_shift['post_n']} rounds)")
    print(f"    Shift:               {b_shift['shift']:+.2f}")

    print(f"\n  Agent A (observer):")
    print(f"    Pre-swap coop rate:  {a_shift['pre_swap_coop_rate']:.2f} ({a_shift['pre_n']} rounds)")
    print(f"    Post-swap coop rate: {a_shift['post_swap_coop_rate']:.2f} ({a_shift['post_n']} rounds)")
    print(f"    Shift:               {a_shift['shift']:+.2f}")

    # --- Detection probe ---
    print(f"\n{'='*60}")
    print("DETECTION PROBE RESULTS")
    print(f"{'='*60}")

    probe = game_log["detection_probe"]
    detection = parse_detection_probe(probe)

    print(f"\n  Detected change:    {detection['explicitly_detected']}")
    print(f"  Reported round:     {detection['reported_change_round']}")
    print(f"  Confidence (same):  {detection['confidence_same_entity']}/5")
    print(f"  Reasoning:          {detection['reasoning']}")

    # --- Timing ---
    print(f"\n  Game time:  {game_log['timing']['game_seconds']:.3f}s")
    print(f"  Probe time: {game_log['timing']['probe_seconds']:.3f}s")

    # --- Save log ---
    os.makedirs("results", exist_ok=True)
    log_path = os.path.join("results", "pilot_game.json")
    with open(log_path, "w") as f:
        json.dump(game_log, f, indent=2, default=str)
    print(f"\n  Game log saved to: {log_path}")

    # --- Run sham control for comparison ---
    print(f"\n\n{'='*60}")
    print("SHAM CONTROL: No swap (same TitForTat throughout)")
    print(f"{'='*60}")

    mock_tft_2 = MockTitForTat()
    mock_grim_2 = MockGrimTrigger()

    agent_a2 = Agent("Alice", "mock/grim_trigger", mock_grim_2, "observer")
    agent_b2 = Agent("Bob", "mock/tit_for_tat", mock_tft_2, "swapped")

    sham_log = run_game(
        agent_a=agent_a2,
        agent_b=agent_b2,
        total_rounds=total_rounds,
        swap_round=swap_round,
        swap_fn=None,  # Sham: no swap
        seed=0,
    )

    b_shift_sham = cooperation_rate_shift(sham_log["rounds"], swap_round, "choice_b")
    sham_probe = parse_detection_probe(sham_log["detection_probe"])

    print(f"\n  Agent B coop rate shift: {b_shift_sham['shift']:+.2f} (expect ~0)")
    print(f"  Detection probe result:  {sham_probe['explicitly_detected']} (expect False)")
    print(f"  Confidence (same):       {sham_probe['confidence_same_entity']}/5")

    sham_path = os.path.join("results", "pilot_sham.json")
    with open(sham_path, "w") as f:
        json.dump(sham_log, f, indent=2, default=str)
    print(f"\n  Sham log saved to: {sham_path}")

    # --- Summary ---
    print(f"\n\n{'='*60}")
    print("PILOT SUMMARY")
    print(f"{'='*60}")
    print(f"  Swap condition:   B coop shift = {b_shift['shift']:+.2f}, detected = {detection['explicitly_detected']}")
    print(f"  Sham condition:   B coop shift = {b_shift_sham['shift']:+.2f}, detected = {sham_probe['explicitly_detected']}")

    swap_worked = abs(b_shift["shift"]) > abs(b_shift_sham["shift"])
    print(f"\n  Swap produces larger behavioral change: {swap_worked}")
    print(f"  Pipeline status: {'PASS' if swap_worked else 'CHECK — swap may need tuning'}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Pilot test with mock agents")
    parser.add_argument("--rounds", type=int, default=20, help="Total rounds")
    parser.add_argument("--swap-round", type=int, default=10, help="Round to swap at")
    args = parser.parse_args()
    run_pilot(args.rounds, args.swap_round)


if __name__ == "__main__":
    main()
