"""
The swap mechanism: functions that create swap callables for each condition.

Each function returns a Callable[[Agent], None] that, when called, mutates
the target agent's model endpoint while preserving its name and history.
"""

from typing import Callable, Optional
from src.agents import Agent, ModelCallFn


def make_sham_swap() -> Optional[Callable[[Agent], None]]:
    """Sham swap: no actual change. Returns None (game loop skips swap)."""
    return None


def make_same_model_swap(
    model_id: str, new_call_fn: ModelCallFn
) -> Callable[[Agent], None]:
    """
    Swap to a fresh instance of the same model.
    The new_call_fn should point to a freshly-initialized endpoint.
    """
    def swap(agent: Agent):
        agent.swap_model(model_id, new_call_fn)
    return swap


def make_model_swap(
    new_model_id: str, new_call_fn: ModelCallFn
) -> Callable[[Agent], None]:
    """
    Swap to a different model entirely.
    Works for both same-family-different-size and different-provider conditions.
    """
    def swap(agent: Agent):
        agent.swap_model(new_model_id, new_call_fn)
    return swap
