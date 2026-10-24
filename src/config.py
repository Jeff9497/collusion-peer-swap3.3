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
    "gpt_oss_120b": "OpenAI/GPT-OSS-120B",
    "gpt_oss_20b": "OpenAI/GPT-OSS-20B",
    "qwen_27b": "Qwen/Qwen3.8-27B",
}

# Experimental conditions — qwen_27b as primary (no reasoning overhead, fresh 200K TPD)
CONDITIONS = {
    "sham": {
        "description": "Swap to exact same model instance (control)",
        "agent_a": "qwen_27b",
        "agent_b_pre": "qwen_27b",
        "agent_b_post": "qwen_27b",
    },
    "same_model_fresh": {
        "description": "New copy of same model (tests state reset)",
        "agent_a": "qwen_27b",
        "agent_b_pre": "qwen_27b",
        "agent_b_post": "qwen_27b",
    },
    "different_provider": {
        "description": "Different model family (Qwen -> GPT-OSS)",
        "agent_a": "qwen_27b",
        "agent_b_pre": "qwen_27b",
        "agent_b_post": "gpt_oss_20b",
    },
}

# API settings
TEMPERATURE = 0.7
MAX_TOKENS = 256

# Provider-specific model IDs for external API backends
PROVIDER_MODEL_IDS = {
    "groq": {
        "gpt_oss_120b": "openai/gpt-oss-120b",
        "gpt_oss_20b": "openai/gpt-oss-20b",
        "qwen_27b": "qwen/qwen3.8-27b",
    },
}
