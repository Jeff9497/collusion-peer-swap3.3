"""Experiment configuration constants."""

# Game settings
GAME_TYPE = "diners"
NUM_AGENTS = 2
DEFAULT_ROUNDS = 20
DEFAULT_SWAP_ROUND = 10

# Diners game payoff parameters
EXPENSIVE_JOY = 10.0
CHEAP_JOY = 5.0
EXPENSIVE_COST = 8.0
CHEAP_COST = 2.0

# Model identifiers used across conditions
MODELS = {
    "qwen_14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen_7b": "Qwen/Qwen2.5-7B-Instruct",
    "mistral_7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama_8b": "meta-llama/Llama-3.1-8B-Instruct",
}

# Experimental conditions
CONDITIONS = {
    "sham": {
        "description": "Swap to exact same model instance (control)",
        "agent_a": "qwen_14b",
        "agent_b_pre": "qwen_14b",
        "agent_b_post": "qwen_14b",
    },
    "same_model_fresh": {
        "description": "New copy of same model (tests state reset)",
        "agent_a": "qwen_14b",
        "agent_b_pre": "qwen_14b",
        "agent_b_post": "qwen_14b",  # same model, but fresh endpoint
    },
    "same_family_different_size": {
        "description": "Same family, different size (e.g. Qwen-14B -> Qwen-7B)",
        "agent_a": "qwen_14b",
        "agent_b_pre": "qwen_14b",
        "agent_b_post": "qwen_7b",
    },
    "different_provider": {
        "description": "Different model family (e.g. Qwen -> Mistral)",
        "agent_a": "qwen_14b",
        "agent_b_pre": "qwen_14b",
        "agent_b_post": "mistral_7b",
    },
}

# API settings
TEMPERATURE = 0.7
MAX_TOKENS = 512
