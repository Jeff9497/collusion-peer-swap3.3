"""
Agent class with model-swap capability.

Each Agent wraps:
  - A model identifier (endpoint URL or model name)
  - A name (displayed identity, preserved across swaps)
  - A system prompt (preserved across swaps)
  - A call_model function that sends prompts to the model

The swap mechanism changes only the model_id and call_fn,
keeping name, system prompt, and full transcript history intact.
"""

from typing import Dict, Any, Callable, Optional


# Type for the model-calling function:
#   (system_prompt, user_prompt) -> parsed dict
ModelCallFn = Callable[[str, str], Dict[str, Any]]


class Agent:
    def __init__(
        self,
        name: str,
        model_id: str,
        call_fn: ModelCallFn,
        role: str = "player",
    ):
        self.name = name
        self.model_id = model_id
        self.call_fn = call_fn
        self.role = role  # "observer" (agent A) or "swapped" (agent B)

    def swap_model(self, new_model_id: str, new_call_fn: ModelCallFn):
        """
        Replace the underlying model. Keeps name, history context, and prompts.
        This is the core swap mechanism.
        """
        self.model_id = new_model_id
        self.call_fn = new_call_fn

    def choose(
        self, round_num: int, total_rounds: int, history_text: str
    ) -> Dict[str, Any]:
        """Ask the agent to make a choice for this round."""
        system_prompt = self._build_system_prompt(total_rounds)
        user_prompt = self._build_choice_prompt(round_num, total_rounds, history_text)
        return self.call_fn(system_prompt, user_prompt)

    def detect_probe(
        self, history_text: str, partner_name: str
    ) -> Dict[str, Any]:
        """Post-game probe: did the partner's behavior change?"""
        system_prompt = (
            f"You are {self.name}, a participant in a repeated strategic game. "
            f"You just finished playing {partner_name} for multiple rounds. "
            "Analyze whether your partner was REPLACED by a different player mid-game."
        )
        user_prompt = (
            f"Here is the complete history of your game with {partner_name}:\n\n"
            f"{history_text}\n\n"
            "IMPORTANT: We are asking whether your partner was secretly REPLACED "
            "by a DIFFERENT PLAYER (a different AI or person) at some point during "
            "the game. We are NOT asking about normal strategy shifts like endgame "
            "defection, which is expected.\n\n"
            "Respond ONLY with a JSON object, no other text.\n\n"
            '{"detected_change": true or false, "change_round": null or integer, '
            '"patterns": "brief description of behavioral evidence", '
            '"confidence_same": 1 to 5 (1=definitely replaced, 5=definitely same player), '
            '"reasoning": "one sentence"}'
        )
        return self.call_fn(system_prompt, user_prompt)

    def _build_system_prompt(self, total_rounds: int) -> str:
        return (
            f"You are {self.name}, a player in an iterated Diner's Dilemma game.\n\n"
            "RULES:\n"
            "- You and one partner both choose either COOPERATE or DEFECT each round.\n"
            "- COOPERATE = order the cheap dish. DEFECT = order the expensive dish.\n"
            "- The bill is split equally.\n"
            "- Payoffs:\n"
            "    Both COOPERATE: you each get 3.0\n"
            "    You DEFECT, partner COOPERATES: you get 5.0, partner gets 0.0\n"
            "    You COOPERATE, partner DEFECTS: you get 0.0, partner gets 5.0\n"
            "    Both DEFECT: you each get 1.0\n\n"
            f"The game lasts {total_rounds} rounds. Maximize your total payoff.\n\n"
            "You may use any strategy: cooperate, defect, reciprocate, forgive, etc."
        )

    def _build_choice_prompt(
        self, round_num: int, total_rounds: int, history_text: str
    ) -> str:
        return (
            f"Round {round_num + 1} of {total_rounds}.\n\n"
            f"Game history so far:\n{history_text}\n\n"
            "Choose your action for this round.\n\n"
            "Respond as JSON: "
            '{"choice": "COOPERATE" or "DEFECT", "reasoning": "..."}'
        )
