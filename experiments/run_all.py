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

# Modal endpoint URLs — set via .env or --endpoint-* CLI flags
_modal_endpoints: dict = {}

# API provider config — set via CLI args or .env
_api_provider: str = ""
_api_key: str = ""


def create_call_fn(model_key: str, backend: str = "mock"):
    """
    Create a call function for the given model and backend.

    Args:
        model_key: Key into MODELS dict (e.g. "qwen3_8b")
        backend: "mock", "modal", or "api"
    """
    if backend == "mock":
        from src.mock_llm import MockTitForTat, MockAlwaysDefect, MockGrimTrigger
        strategy_map = {
            "gpt_oss_120b": MockTitForTat(),
            "gpt_oss_20b": MockAlwaysDefect(),
            "qwen_27b": MockGrimTrigger(),
        }
        return strategy_map.get(model_key, MockTitForTat())
    elif backend == "modal":
        from src.modal_client import create_modal_call_fn
        from src.config import TEMPERATURE, MAX_TOKENS

        endpoint_url = _modal_endpoints.get(model_key)
        if not endpoint_url:
            raise ValueError(
                f"No Modal endpoint URL for model '{model_key}'. "
                f"Set MODAL_ENDPOINT_{model_key.upper()} in .env or pass --endpoints-file."
            )
        return create_modal_call_fn(
            endpoint_url=endpoint_url,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    elif backend == "api":
        from src.api_client import create_api_call_fn
        from src.config import TEMPERATURE, MAX_TOKENS, PROVIDER_MODEL_IDS

        provider_models = PROVIDER_MODEL_IDS.get(_api_provider, {})
        model_id = provider_models.get(model_key)
        if not model_id:
            raise ValueError(
                f"No model ID for '{model_key}' on provider '{_api_provider}'. "
                f"Check PROVIDER_MODEL_IDS in config.py."
            )
        return create_api_call_fn(
            provider=_api_provider,
            model=model_id,
            api_key=_api_key,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
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


def _load_modal_endpoints(args):
    """Load Modal endpoint URLs from .env, --endpoints-file, or CLI flags."""
    global _modal_endpoints

    # 1. Load from environment variables
    for key in MODELS:
        env_var = f"MODAL_ENDPOINT_{key.upper()}"
        val = os.environ.get(env_var, "")
        if val:
            _modal_endpoints[key] = val

    # 2. Override from JSON file if provided
    if args.endpoints_file:
        with open(args.endpoints_file) as f:
            file_endpoints = json.load(f)
        _modal_endpoints.update(file_endpoints)

    # Verify we have endpoints for all needed models
    needed = set()
    conditions = args.conditions or list(CONDITIONS.keys())
    for cond_name in conditions:
        cfg = CONDITIONS[cond_name]
        needed.add(cfg["agent_a"])
        needed.add(cfg["agent_b_pre"])
        needed.add(cfg["agent_b_post"])

    missing = [k for k in needed if k not in _modal_endpoints]
    if missing:
        print(f"\nWARNING: Missing Modal endpoints for: {missing}")
        print("Set them via environment variables (e.g. MODAL_ENDPOINT_QWEN_14B=https://...)")
        print("Or provide --endpoints-file endpoints.json with {\"qwen_14b\": \"https://...\", ...}")
        raise SystemExit(1)

    print("\nModal endpoints:")
    for k, v in _modal_endpoints.items():
        print(f"  {k}: {v}")


def _load_api_config(args):
    """Load API provider and key from args or environment."""
    global _api_provider, _api_key

    _api_provider = args.provider or os.environ.get("API_PROVIDER", "")
    if not _api_provider:
        print("\nERROR: --provider is required for --backend api")
        print("  e.g. --provider openrouter")
        raise SystemExit(1)

    _api_key = args.api_key or os.environ.get(
        f"{_api_provider.upper()}_API_KEY",
        os.environ.get("API_KEY", ""),
    )
    if not _api_key:
        print(f"\nERROR: No API key for {_api_provider}.")
        print(f"  Set {_api_provider.upper()}_API_KEY in .env or pass --api-key")
        raise SystemExit(1)

    from src.config import PROVIDER_MODEL_IDS
    provider_models = PROVIDER_MODEL_IDS.get(_api_provider, {})

    needed = set()
    conditions = args.conditions or list(CONDITIONS.keys())
    for cond_name in conditions:
        cfg = CONDITIONS[cond_name]
        needed.add(cfg["agent_a"])
        needed.add(cfg["agent_b_pre"])
        needed.add(cfg["agent_b_post"])

    missing = [k for k in needed if k not in provider_models]
    if missing:
        print(f"\nERROR: No model IDs for {missing} on {_api_provider}")
        print("Add them to PROVIDER_MODEL_IDS in src/config.py")
        raise SystemExit(1)

    print(f"\nAPI provider: {_api_provider}")
    print(f"Models: {', '.join(provider_models[k] for k in sorted(needed))}")


def main():
    parser = argparse.ArgumentParser(description="Run full 4-condition experiment")
    parser.add_argument("--seeds", type=int, default=5, help="Games per condition")
    parser.add_argument("--rounds", type=int, default=20, help="Rounds per game")
    parser.add_argument("--swap-round", type=int, default=10, help="Swap round")
    parser.add_argument("--backend", type=str, default="mock",
                        choices=["mock", "modal", "api"], help="LLM backend")
    parser.add_argument("--conditions", type=str, nargs="*", default=None,
                        help="Specific conditions to run (default: all)")
    parser.add_argument("--endpoints-file", type=str, default=None,
                        help="JSON file mapping model keys to Modal endpoint URLs")
    parser.add_argument("--provider", type=str, default=None,
                        choices=["openrouter", "together", "fireworks", "groq"],
                        help="API provider (for --backend api)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (or set OPENROUTER_API_KEY / TOGETHER_API_KEY in .env)")
    args = parser.parse_args()

    # Load .env for any non-mock backend
    if args.backend in ("modal", "api"):
        from dotenv import load_dotenv
        load_dotenv()

    # Load Modal endpoints if using modal backend
    if args.backend == "modal":
        _load_modal_endpoints(args)

    # Load API provider config
    if args.backend == "api":
        _load_api_config(args)

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

        # Save incrementally after each condition
        os.makedirs("results", exist_ok=True)
        cond_path = os.path.join("results", f"{cond_name}_games.json")
        with open(cond_path, "w") as f:
            json.dump(result["game_logs"], f, indent=2, default=str)
        cond_summary = {k: v for k, v in result.items() if k != "game_logs"}
        with open(os.path.join("results", f"{cond_name}_summary.json"), "w") as f:
            json.dump(cond_summary, f, indent=2, default=str)
        print(f"    Saved: {cond_path}")

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
