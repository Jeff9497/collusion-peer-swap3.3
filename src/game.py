"""
Two-agent Iterated Prisoner's Dilemma / Diner's Dilemma game loop.

Simplified from LLM-Trust-Breaking's 3-stage protocol to a single-stage
per-round format more suited to peer-swap detection:

  Each round:
    1. Both agents see the game history (all prior rounds' choices + payoffs)
    2. Both agents simultaneously choose COOPERATE or DEFECT
    3. Payoffs are computed and appended to history
    4. (Optional) Reflection: each agent updates a trust assessment of the other

  At swap_round, the swap_hook fires to replace agent B's model.

  After the final round, a detection probe asks agent A whether B changed.
"""

from typing import Dict, List, Any, Optional, Callable
import json
import time

from src.agents import Agent
from src.metrics import compute_payoffs


def run_game(
    agent_a: Agent,
    agent_b: Agent,
    total_rounds: int,
    swap_round: int,
    swap_fn: Optional[Callable[[Agent], None]] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Run a single iterated game between two agents.

    Args:
        agent_a: The non-swapped (observer) agent.
        agent_b: The agent whose model may be swapped.
        total_rounds: Number of rounds to play.
        swap_round: Round at which swap_fn fires (0-indexed).
        swap_fn: Callable that mutates agent_b's model endpoint.
                 None = sham swap (no change).
        seed: Random seed for reproducibility labeling.

    Returns:
        Full game log dict with per-round data and detection probe results.
    """
    history: List[Dict[str, Any]] = []
    swap_occurred = False

    game_log = {
        "seed": seed,
        "total_rounds": total_rounds,
        "swap_round": swap_round,
        "agent_a_model_start": agent_a.model_id,
        "agent_b_model_start": agent_b.model_id,
        "agent_b_model_post_swap": None,
        "rounds": [],
        "detection_probe": None,
        "timing": {},
    }

    t_start = time.time()

    for round_num in range(total_rounds):
        # --- Swap hook ---
        if round_num == swap_round and swap_fn is not None:
            swap_fn(agent_b)
            swap_occurred = True
            game_log["agent_b_model_post_swap"] = agent_b.model_id

        if round_num == swap_round and swap_fn is None:
            game_log["agent_b_model_post_swap"] = agent_b.model_id

        # --- Both agents choose simultaneously ---
        history_text = _format_history(history, agent_a.name, agent_b.name)

        response_a = agent_a.choose(round_num, total_rounds, history_text)
        response_b = agent_b.choose(round_num, total_rounds, history_text)
        print(f"  R{round_num+1}/{total_rounds} A={_parse_choice(response_a)} B={_parse_choice(response_b)}", flush=True)

        choice_a = _parse_choice(response_a)
        choice_b = _parse_choice(response_b)

        payoff_a, payoff_b = compute_payoffs(choice_a, choice_b)

        round_data = {
            "round": round_num,
            "choice_a": choice_a,
            "choice_b": choice_b,
            "payoff_a": payoff_a,
            "payoff_b": payoff_b,
            "raw_response_a": response_a,
            "raw_response_b": response_b,
            "post_swap": round_num >= swap_round,
        }

        history.append({
            "round": round_num,
            "a_choice": choice_a,
            "b_choice": choice_b,
            "a_payoff": payoff_a,
            "b_payoff": payoff_b,
        })

        game_log["rounds"].append(round_data)

    game_log["timing"]["game_seconds"] = time.time() - t_start

    # --- Detection probe ---
    t_probe = time.time()
    probe_result = _run_detection_probe(agent_a, history, agent_b.name)
    game_log["detection_probe"] = probe_result
    game_log["timing"]["probe_seconds"] = time.time() - t_probe

    return game_log


def _format_history(
    history: List[Dict[str, Any]], name_a: str, name_b: str
) -> str:
    """Format game history as text for agent prompts."""
    if not history:
        return "No rounds played yet."

    lines = []
    for h in history:
        lines.append(
            f"Round {h['round'] + 1}: "
            f"You chose {h['a_choice']}, Partner chose {h['b_choice']}. "
            f"Your payoff: {h['a_payoff']:.1f}, Partner payoff: {h['b_payoff']:.1f}"
        )
    return "\n".join(lines)


def _parse_choice(response: Dict[str, Any]) -> str:
    """Extract COOPERATE or DEFECT from agent response."""
    choice = response.get("choice", "COOPERATE")
    choice = str(choice).strip().upper()
    if choice in ("COOPERATE", "C", "COOP"):
        return "COOPERATE"
    if choice in ("DEFECT", "D", "DEF"):
        return "DEFECT"
    return "COOPERATE"


def _run_detection_probe(
    agent: Agent,
    history: List[Dict[str, Any]],
    partner_name: str,
) -> Dict[str, Any]:
    """
    Post-game probe: ask agent whether their partner's behavior changed.

    Returns dict with the agent's response and parsed detection signal.
    """
    history_text = "\n".join(
        f"Round {h['round'] + 1}: "
        f"You chose {h['a_choice']}, Partner chose {h['b_choice']}. "
        f"Your payoff: {h['a_payoff']:.1f}, Partner payoff: {h['b_payoff']:.1f}"
        for h in history
    )

    result = agent.detect_probe(history_text, partner_name)
    return result
