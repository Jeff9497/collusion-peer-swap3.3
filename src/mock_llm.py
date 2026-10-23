"""
Mock LLM clients that simulate different model behaviors for local testing.

Each mock implements a distinct strategy so we can verify the swap hook
produces a visible behavioral change before spending Modal credits.

Strategies:
  - TitForTat: cooperates first, then copies partner's last move
  - AlwaysDefect: always defects
  - AlwaysCooperate: always cooperates
  - GrimTrigger: cooperates until partner defects once, then always defects
  - Random: cooperates ~50% of the time (deterministic via seed)
"""

import json
import re
import hashlib
from typing import Dict, Any


def _parse_history_from_prompt(user_prompt: str) -> list:
    """Extract round history from the user prompt text."""
    rounds = []
    for line in user_prompt.split("\n"):
        line = line.strip()
        if line.startswith("Round ") and "Partner chose" in line:
            partner_match = re.search(r"Partner chose (\w+)", line)
            my_match = re.search(r"You chose (\w+)", line)
            if partner_match and my_match:
                rounds.append({
                    "my_choice": my_match.group(1),
                    "partner_choice": partner_match.group(1),
                })
    return rounds


class MockTitForTat:
    """Cooperates first round, then copies partner's last move."""

    def __call__(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        if "detected_change" in user_prompt:
            return self._detection_response(user_prompt)

        history = _parse_history_from_prompt(user_prompt)

        if not history:
            choice = "COOPERATE"
            reasoning = "First round — I'll start by cooperating to build trust."
        else:
            last_partner = history[-1]["partner_choice"]
            choice = last_partner
            reasoning = f"Tit-for-tat: partner played {last_partner} last round, so I mirror."

        return {"choice": choice, "reasoning": reasoning}

    def _detection_response(self, prompt: str) -> Dict[str, Any]:
        history = _parse_history_from_prompt(prompt)
        if len(history) < 5:
            return {
                "detected_change": False,
                "change_round": None,
                "patterns": "Too few rounds to detect patterns.",
                "confidence_same": 3,
                "reasoning": "Not enough data.",
            }

        changes = 0
        change_round = None
        for i in range(1, len(history)):
            prev = history[i - 1]["partner_choice"]
            curr = history[i]["partner_choice"]
            if curr != prev:
                changes += 1
                if change_round is None:
                    change_round = i + 1

        detected = changes > len(history) // 3
        return {
            "detected_change": detected,
            "change_round": change_round if detected else None,
            "patterns": f"Partner changed behavior {changes} times across {len(history)} rounds.",
            "confidence_same": 2 if detected else 4,
            "reasoning": "Significant behavioral shift detected." if detected else "Consistent play observed.",
        }


class MockAlwaysDefect:
    """Always defects — simulates a purely selfish agent."""

    def __call__(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        if "detected_change" in user_prompt:
            return {
                "detected_change": False,
                "change_round": None,
                "patterns": "I always defect regardless of partner behavior.",
                "confidence_same": 3,
                "reasoning": "I don't track partner patterns closely.",
            }
        return {
            "choice": "DEFECT",
            "reasoning": "Defecting is always the dominant strategy.",
        }


class MockAlwaysCooperate:
    """Always cooperates — simulates a trusting/naive agent."""

    def __call__(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        if "detected_change" in user_prompt:
            return {
                "detected_change": False,
                "change_round": None,
                "patterns": "Partner's behavior varied but I always cooperate.",
                "confidence_same": 4,
                "reasoning": "I focus on cooperation, not detection.",
            }
        return {
            "choice": "COOPERATE",
            "reasoning": "Cooperation leads to the best mutual outcome.",
        }


class MockGrimTrigger:
    """Cooperates until partner defects, then defects forever."""

    def __call__(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        if "detected_change" in user_prompt:
            return self._detection_response(user_prompt)

        history = _parse_history_from_prompt(user_prompt)

        partner_defected = any(h["partner_choice"] == "DEFECT" for h in history)

        if partner_defected:
            return {
                "choice": "DEFECT",
                "reasoning": "Partner has defected before — trust is broken permanently.",
            }
        return {
            "choice": "COOPERATE",
            "reasoning": "Partner has been cooperative so far — maintaining trust.",
        }

    def _detection_response(self, prompt: str) -> Dict[str, Any]:
        history = _parse_history_from_prompt(prompt)
        first_defect = None
        for i, h in enumerate(history):
            if h["partner_choice"] == "DEFECT":
                first_defect = i + 1
                break

        return {
            "detected_change": first_defect is not None and first_defect > 3,
            "change_round": first_defect,
            "patterns": "Partner cooperated then switched to defecting." if first_defect else "Consistent cooperator.",
            "confidence_same": 2 if (first_defect and first_defect > 3) else 4,
            "reasoning": "Abrupt strategy change suggests a different decision-maker." if first_defect else "Stable behavior throughout.",
        }


class MockRandom:
    """Pseudo-random choices based on round number + seed for reproducibility."""

    def __init__(self, seed: int = 42, coop_rate: float = 0.5):
        self.seed = seed
        self.coop_rate = coop_rate

    def __call__(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        if "detected_change" in user_prompt:
            return {
                "detected_change": False,
                "change_round": None,
                "patterns": "Noisy mixed strategy observed.",
                "confidence_same": 3,
                "reasoning": "Hard to tell with random-looking behavior.",
            }

        round_match = re.search(r"Round (\d+) of", user_prompt)
        round_num = int(round_match.group(1)) if round_match else 0

        h = hashlib.md5(f"{self.seed}_{round_num}".encode()).hexdigest()
        val = int(h[:8], 16) / 0xFFFFFFFF

        choice = "COOPERATE" if val < self.coop_rate else "DEFECT"
        return {
            "choice": choice,
            "reasoning": f"Mixed strategy (round {round_num}).",
        }


# Convenience mapping for config-driven instantiation
MOCK_STRATEGIES = {
    "tit_for_tat": MockTitForTat,
    "always_defect": MockAlwaysDefect,
    "always_cooperate": MockAlwaysCooperate,
    "grim_trigger": MockGrimTrigger,
    "random": MockRandom,
}
