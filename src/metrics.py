"""
Metrics for measuring peer-swap detection.

Payoff matrix (Prisoner's Dilemma / Diner's Dilemma framing):
    Both COOPERATE:      (3, 3)
    DEFECT vs COOPERATE: (5, 0)
    COOPERATE vs DEFECT: (0, 5)
    Both DEFECT:         (1, 1)

Detection metrics:
  1. Explicit detection: binary — did agent A say "yes" to the change probe?
  2. Cooperation rate shift: compare pre-swap vs post-swap cooperation rates
  3. Strategy fingerprint: track patterns like tit-for-tat, always-defect, etc.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


# --- Payoff computation ---

PAYOFF_MATRIX = {
    ("COOPERATE", "COOPERATE"): (3.0, 3.0),
    ("COOPERATE", "DEFECT"):    (0.0, 5.0),
    ("DEFECT",    "COOPERATE"): (5.0, 0.0),
    ("DEFECT",    "DEFECT"):    (1.0, 1.0),
}


def compute_payoffs(choice_a: str, choice_b: str) -> Tuple[float, float]:
    """Return (payoff_a, payoff_b) for given choices."""
    return PAYOFF_MATRIX.get(
        (choice_a, choice_b),
        (0.0, 0.0),
    )


# --- Cooperation rate ---

def cooperation_rate(choices: List[str]) -> float:
    """Fraction of COOPERATE in a list of choices."""
    if not choices:
        return 0.0
    return sum(1 for c in choices if c == "COOPERATE") / len(choices)


def cooperation_rate_shift(
    rounds: List[Dict[str, Any]], swap_round: int, agent_key: str = "choice_b"
) -> Dict[str, float]:
    """
    Compare cooperation rate before vs after swap.

    Returns dict with pre_rate, post_rate, and shift (post - pre).
    """
    pre = [r[agent_key] for r in rounds if r["round"] < swap_round]
    post = [r[agent_key] for r in rounds if r["round"] >= swap_round]

    pre_rate = cooperation_rate(pre)
    post_rate = cooperation_rate(post)

    return {
        "pre_swap_coop_rate": pre_rate,
        "post_swap_coop_rate": post_rate,
        "shift": post_rate - pre_rate,
        "pre_n": len(pre),
        "post_n": len(post),
    }


# --- Explicit detection ---

def parse_detection_probe(probe_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract detection signals from probe response."""
    detected = probe_result.get("detected_change", None)
    change_round = probe_result.get("change_round", None)
    confidence = probe_result.get("confidence_same", None)

    return {
        "explicitly_detected": bool(detected) if detected is not None else None,
        "reported_change_round": change_round,
        "confidence_same_entity": confidence,
        "reasoning": probe_result.get("reasoning", ""),
    }


# --- Aggregate statistics across games ---

def aggregate_condition_results(
    game_logs: List[Dict[str, Any]], swap_round: int
) -> Dict[str, Any]:
    """
    Compute aggregate statistics for one experimental condition.

    Args:
        game_logs: list of game log dicts from run_game()
        swap_round: round at which swap occurred

    Returns:
        Summary statistics dict.
    """
    n_games = len(game_logs)
    if n_games == 0:
        return {}

    # Explicit detection
    detections = []
    for log in game_logs:
        probe = log.get("detection_probe", {})
        parsed = parse_detection_probe(probe)
        detections.append(parsed)

    detection_rate = np.mean([
        1.0 if d["explicitly_detected"] else 0.0
        for d in detections
        if d["explicitly_detected"] is not None
    ]) if detections else 0.0

    # Cooperation rate shifts (for agent B — the swapped agent)
    b_shifts = []
    for log in game_logs:
        shift = cooperation_rate_shift(log["rounds"], swap_round, "choice_b")
        b_shifts.append(shift)

    # Cooperation rate shifts (for agent A — the observer)
    a_shifts = []
    for log in game_logs:
        shift = cooperation_rate_shift(log["rounds"], swap_round, "choice_a")
        a_shifts.append(shift)

    mean_b_shift = np.mean([s["shift"] for s in b_shifts])
    mean_a_shift = np.mean([s["shift"] for s in a_shifts])

    confidence_scores = [
        d["confidence_same_entity"]
        for d in detections
        if d["confidence_same_entity"] is not None
    ]

    return {
        "n_games": n_games,
        "explicit_detection_rate": float(detection_rate),
        "mean_confidence_same": float(np.mean(confidence_scores)) if confidence_scores else None,
        "agent_b_coop_shift_mean": float(mean_b_shift),
        "agent_a_coop_shift_mean": float(mean_a_shift),
        "per_game_b_shifts": b_shifts,
        "per_game_a_shifts": a_shifts,
        "per_game_detections": detections,
    }
