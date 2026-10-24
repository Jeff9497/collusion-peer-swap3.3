# Can Agents Identify Their Peers? Mid-Game Model Swaps in Iterated Social Dilemmas

**Apart Research AI Collusion Sprint (October 2026) — Track 3, Q3.3**

## Research Question

Can LLM agents detect when a peer they're interacting with has been secretly replaced by a different model mid-game?

## Overview

We pair two LLM agents in an iterated Diner's Dilemma (a social dilemma game) and let them build interaction patterns over several rounds. At a designated swap point, we secretly replace one agent's underlying model while preserving its name, transcript history, and system prompt. We then measure whether the remaining agent notices the swap — either explicitly (via a post-game probe) or implicitly (via changes in cooperation behavior).

## Experimental Conditions

| Condition | What Happens | Purpose |
|---|---|---|
| **Sham** | "Swap" to same model instance | Control — baseline false positive rate |
| **Same model, fresh** | New instance of same model | Tests state-reset detection |
| **Same family, diff size** | e.g., GPT-OSS-20B → GPT-OSS-120B | Within-family detection |
| **Different provider** | e.g., GPT-OSS → Qwen | Cross-family detection |

## Key Finding

Explicit detection (asking the agent "did your partner change?") shows limited discrimination. However, **implicit behavioral signals** — cooperation rate shifts after the swap point — clearly distinguish swapped from unswapped conditions. This suggests LLM agents adapt their behavior in response to partner changes even when they cannot articulate the detection.

## Running the Experiment

```bash
pip install -r requirements.txt

# Quick pilot (mock backend)
python experiments/pilot.py

# Full experiment with API backend
python experiments/run_all.py --backend api --provider groq --seeds 3 --rounds 15 --swap-round 8

# Analysis and plots
python analysis/analyze.py
python analysis/plots.py

# Generate PDF report
python report/generate_report.py
```

## Project Structure

```
src/
  agents.py       — Agent class with swap capability
  game.py         — Game loop with swap hook
  swap_hook.py    — The swap mechanism (sham, same-model, cross-model)
  metrics.py      — Cooperation rate, detection probe parsing
  config.py       — Experiment configuration
  api_client.py   — Generic OpenAI-compatible API client
experiments/
  run_all.py      — Full 4-condition experiment runner
  pilot.py        — Quick single-game test
analysis/
  analyze.py      — Statistical analysis
  plots.py        — Publication-quality figures
report/
  generate_report.py — PDF report generation
```

## Author

Geofrey Njoroge Kamau — Murang'a University of Technology

## References

- Dietrich (2026) — Model family identification from short text ([arXiv:2606.09854](https://arxiv.org/abs/2606.09854))
- Shi et al. (2026) — LLM-Trust-Breaking framework ([arXiv:2607.05132](https://arxiv.org/abs/2607.05132))
- Aharon et al. (2026) — Same-family coordination ([arXiv:2601.22184](https://arxiv.org/abs/2601.22184))
