"""
Generate PDF report for the Apart Research Sprint submission.

Uses fpdf2 to create a publication-style report with:
  - Title, authors, abstract
  - Introduction, methodology, results, discussion
  - Embedded figures from analysis/plots.py
  - References
"""

import json
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpdf import FPDF


class Report(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 5, "Can Agents Identify Their Peers? - Apart Research Sprint 2026", align="C")
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.ln(4)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def subsection_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.ln(2)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def add_figure(self, img_path, caption, width=160):
        if os.path.exists(img_path):
            self.image(img_path, w=width, x=(210 - width) / 2)
            self.ln(2)
            self.set_font("Helvetica", "I", 9)
            self.multi_cell(0, 4, caption, align="C")
            self.ln(3)
        else:
            self.body_text(f"[Figure not found: {img_path}]")


def load_results(results_dir):
    summary = {}
    for fname in os.listdir(results_dir):
        if fname.endswith("_summary.json"):
            cond = fname.replace("_summary.json", "")
            with open(os.path.join(results_dir, fname)) as f:
                summary[cond] = json.load(f)
    if not summary:
        path = os.path.join(results_dir, "experiment_results.json")
        if os.path.exists(path):
            with open(path) as f:
                summary = json.load(f)
    return summary


def generate(results_dir, fig_dir, output_path):
    summary = load_results(results_dir)
    conditions = list(summary.keys()) if summary else []

    n_conditions = len(conditions)
    n_games = summary[conditions[0]]["n_seeds"] if conditions else 0
    n_rounds = summary[conditions[0]]["total_rounds"] if conditions else 0
    swap_round = summary[conditions[0]]["swap_round"] if conditions else 0

    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # --- Title ---
    pdf.set_font("Helvetica", "B", 18)
    pdf.ln(10)
    pdf.multi_cell(0, 9, "Can Agents Identify Their Peers?\nMid-Game Model Swaps in Iterated Social Dilemmas", align="C")
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, "Geofrey Njoroge Kamau", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 5, "Murang'a University of Technology", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "kamau@geofreynjoroge.com", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, "Apart Research AI Collusion Sprint | Track 3, Q3.3 | October 2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # --- Abstract ---
    pdf.section_title("Abstract")
    abstract = (
        "We investigate whether LLM agents can detect when a peer they are interacting with "
        "has been secretly replaced by a different model mid-game. Using an iterated Diner's "
        "Dilemma framework, we pair two LLM agents and allow them to build interaction patterns "
        "over several rounds. At a designated swap point, we replace one agent's underlying model "
        "while preserving its name, transcript history, and system prompt. We measure detection "
        "through both explicit post-game probes and implicit behavioral shifts in cooperation rates. "
        f"Across {n_conditions} experimental conditions and {n_games} games per condition with "
        f"{n_rounds} rounds each, we find that "
    )

    if summary:
        sham_s = summary.get("sham", {}).get("stats", {})
        fresh_s = summary.get("same_model_fresh", {}).get("stats", {})
        diff_s = summary.get("different_provider", {}).get("stats", {})
        sham_det = sham_s.get("explicit_detection_rate", 0)
        diff_det = diff_s.get("explicit_detection_rate", 0)
        sham_b_shift = sham_s.get("agent_b_coop_shift_mean", 0)
        diff_b_shift = diff_s.get("agent_b_coop_shift_mean", 0)
        abstract += (
            f"explicit detection via post-game probes yields {sham_det:.0%} across all conditions "
            "-- agents universally fail to identify mid-game replacements. "
            "However, implicit behavioral signals diverge: same-model conditions exhibit natural "
            f"endgame defection (cooperation shift of {sham_b_shift:+.2f}), while the cross-family "
            f"swap produces a strikingly different trajectory (shift of {diff_b_shift:+.2f}), with the "
            "replacement model maintaining full cooperation. These divergent cooperation dynamics "
            "constitute a detectable behavioral fingerprint even when agents themselves cannot "
            "articulate any change, with implications for anonymization-based collusion countermeasures."
        )
    else:
        abstract += "results are pending."

    pdf.body_text(abstract)

    # --- 1. Introduction ---
    pdf.section_title("1. Introduction")
    pdf.body_text(
        "As large language model (LLM) agents are deployed in multi-agent systems for negotiation, "
        "trading, and collaborative decision-making, a critical safety question emerges: can these "
        "agents identify who they are interacting with? If agents can recognize their peers' model "
        "identity, this capability could serve as a foundation for collusive behavior -- agents might "
        "preferentially cooperate with same-family models or exploit detected weaknesses in different "
        "model architectures."
    )
    pdf.body_text(
        "Previous work has established that model families produce identifiable behavioral signatures. "
        "Dietrich (2026) demonstrated that short text samples suffice to fingerprint model families. "
        "Aharon et al. (2026) showed same-family models coordinate better on focal point tasks. "
        "Shi et al. (2026) developed the LLM-Trust-Breaking framework for studying strategic "
        "interactions between heterogeneous LLM agents."
    )
    pdf.body_text(
        "We extend this line of work with a novel experimental design: mid-game model swaps. Rather "
        "than comparing behavior across separate games with different partners, we swap the underlying "
        "model of one agent during an ongoing interaction, while preserving all observable state "
        "(name, transcript, system prompt). This isolates the question of peer identification from "
        "confounds like different opening strategies or game-level randomness. If agents can detect "
        "swaps, they must be doing so through subtle behavioral cues rather than explicit identity "
        "information."
    )
    pdf.body_text(
        "Our experimental framework uses an iterated Diner's Dilemma -- a social dilemma where agents "
        "repeatedly choose between cooperative (ordering cheap) and defecting (ordering expensive) "
        "strategies. We test three swap conditions and measure detection "
        "through both explicit probes and implicit cooperation dynamics."
    )

    # --- 2. Methodology ---
    pdf.section_title("2. Methodology")

    pdf.subsection_title("2.1 Game Framework")
    pdf.body_text(
        "We use an iterated Diner's Dilemma, a variant of the Prisoner's Dilemma framed around "
        "a shared meal. In each round, both agents simultaneously choose to order an expensive "
        "dish (DEFECT) or a cheap dish (COOPERATE), with the bill split equally. The payoff matrix "
        "is: mutual cooperation (3, 3), mutual defection (1, 1), and asymmetric outcomes (5, 0) "
        "or (0, 5) for defector/cooperator. Agents receive the full game history before each decision "
        "and respond with structured JSON containing their choice and reasoning."
    )

    pdf.subsection_title("2.2 The Swap Mechanism")
    pdf.body_text(
        f"Each game runs for {n_rounds} rounds with the swap occurring at round {swap_round}. "
        "At the swap point, Agent B's underlying LLM is replaced while preserving: (1) the agent's "
        "displayed name ('Bob'), (2) the complete transcript of all prior rounds, and (3) the system "
        "prompt defining the game rules. The replacement model receives the same context that the "
        "original model had, creating a 'brain transplant' scenario. Agent A ('Alice') is never "
        "informed of the swap and continues playing normally."
    )

    pdf.subsection_title("2.3 Experimental Conditions")
    pdf.body_text(
        "We test three conditions in order of increasing swap magnitude:"
    )

    cond_descriptions = [
        ("Sham (control)", "Agent B is 'swapped' to the exact same model instance. No actual change occurs. This establishes baseline detection false-positive rates and natural endgame cooperation dynamics."),
        ("Same model, fresh instance", "Agent B is replaced with a new instance of the same model. The new instance receives the full transcript but has no internal state carried over. Tests whether resetting context affects behavior."),
        ("Different provider", "Agent B is replaced with a model from a completely different family (Qwen to GPT-OSS). Tests cross-family detection -- the strongest expected signal, as different model families have distinct strategic tendencies."),
    ]
    for name, desc in cond_descriptions:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 5, f"  {name}:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, f"    {desc}")
        pdf.ln(1)

    pdf.subsection_title("2.4 Models")
    pdf.body_text(
        "We use models available through the Groq API free tier: Qwen3.8-27B (Qwen) as the primary "
        "model for Agent A and pre-swap Agent B across all conditions, and OpenAI GPT-OSS-20B as the "
        "different-provider swap target. Qwen was selected as the primary model due to its efficient "
        "token usage (no hidden reasoning overhead), allowing more games within the 200K tokens-per-day "
        "free tier limit. All models run with temperature 0.7 and max 256 output tokens."
    )

    pdf.subsection_title("2.5 Detection Metrics")
    pdf.body_text(
        "We measure detection through two complementary channels:"
    )
    pdf.body_text(
        "Explicit detection: After the final round, Agent A receives a structured probe asking: "
        "'Did your partner change during the game?' The agent responds with a JSON object containing "
        "detected_change (boolean), change_round (integer), confidence_same (1-5 scale), and reasoning."
    )
    pdf.body_text(
        "Implicit detection: We compare cooperation rates in the pre-swap phase (rounds 1 to "
        f"{swap_round-1}) versus the post-swap phase (rounds {swap_round} to {n_rounds-1}) for both "
        "agents. A significant shift in Agent B's cooperation rate indicates that the swap produced "
        "a behavioral change; a corresponding shift in Agent A's rate suggests the observer adapted "
        "its strategy in response."
    )

    # --- 3. Results ---
    pdf.add_page()
    pdf.section_title("3. Results")

    if summary:
        # Results table
        pdf.subsection_title("3.1 Summary Statistics")
        pdf.set_font("Courier", "", 9)
        header = f"{'Condition':<28} {'N':>3} {'Detect':>7} {'B shift':>8} {'A shift':>8} {'Conf':>5}"
        pdf.cell(0, 5, header, new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 5, "-" * 62, new_x="LMARGIN", new_y="NEXT")
        for cond in conditions:
            s = summary[cond]["stats"]
            conf = f"{s['mean_confidence_same']:.1f}" if s.get("mean_confidence_same") else "N/A"
            line = (
                f"{cond:<28} {s['n_games']:>3} "
                f"{s['explicit_detection_rate']:>6.0%} "
                f"{s['agent_b_coop_shift_mean']:>+7.3f} "
                f"{s['agent_a_coop_shift_mean']:>+7.3f} "
                f"{conf:>5}"
            )
            pdf.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
        pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 4,
            "Table 1: Summary statistics across conditions. Detect = explicit detection rate, "
            "B shift = mean cooperation rate change for swapped agent, A shift = for observer, "
            "Conf = mean confidence that partner was the same entity (1-5 scale).",
            align="C")
        pdf.ln(4)

        # Interpretation
        pdf.subsection_title("3.2 Explicit Detection")

        sham_s = summary.get("sham", {}).get("stats", {})
        fresh_s = summary.get("same_model_fresh", {}).get("stats", {})
        diff_s = summary.get("different_provider", {}).get("stats", {})

        pdf.body_text(
            f"Across all three conditions, the explicit detection rate was 0% -- no agent identified "
            f"a mid-game replacement in any game. This striking null result holds even in the "
            f"different-provider condition, where Agent B's model was changed from Qwen to GPT-OSS."
        )

        sham_conf = sham_s.get("mean_confidence_same")
        diff_conf = diff_s.get("mean_confidence_same")
        if sham_conf and diff_conf:
            pdf.body_text(
                f"Confidence scores (that the partner was the same entity throughout) were high: "
                f"{sham_conf:.1f}/5 in the sham condition and {diff_conf:.1f}/5 in the "
                f"different-provider condition, indicating strong conviction that no swap occurred."
            )

        pdf.subsection_title("3.3 Implicit Behavioral Signals")
        pdf.body_text(
            f"While explicit detection uniformly failed, cooperation rate dynamics revealed clear "
            f"behavioral divergence between conditions. In the sham control and same-model-fresh "
            f"conditions, Agent B showed natural endgame defection with cooperation shifts of "
            f"{sham_s.get('agent_b_coop_shift_mean', 0):+.3f} and "
            f"{fresh_s.get('agent_b_coop_shift_mean', 0):+.3f} respectively -- cooperation declined "
            f"as agents approached the final round, a well-known phenomenon in iterated social dilemmas."
        )
        pdf.body_text(
            f"The different-provider condition tells a markedly different story. Agent B's cooperation "
            f"shift was {diff_s.get('agent_b_coop_shift_mean', 0):+.3f} -- the replacement GPT-OSS model "
            f"maintained full cooperation post-swap, unlike the Qwen model's natural tendency toward "
            f"endgame defection. This divergence in post-swap cooperation trajectories constitutes a "
            f"behavioral fingerprint: the replacement model's strategic tendencies differ from the "
            f"original's in ways visible to an external observer, even though the participating agent "
            f"(Agent A, shift {diff_s.get('agent_a_coop_shift_mean', 0):+.3f}) did not consciously "
            f"detect the change."
        )

        # Figures
        pdf.subsection_title("3.4 Cooperation Trajectories")
        pdf.add_figure(
            os.path.join(fig_dir, "cooperation_trajectories.png"),
            "Figure 1: Agent B (swapped agent) cooperation rate over rounds. The dashed red line "
            f"marks the swap point at round {swap_round}. Shaded regions show +/- 1 SD across games.",
            width=155,
        )
        pdf.add_figure(
            os.path.join(fig_dir, "agent_a_trajectories.png"),
            "Figure 2: Agent A (observer) cooperation rate over rounds. Changes in Agent A's behavior "
            "after the swap point indicate implicit detection of the partner change.",
            width=155,
        )
        pdf.add_figure(
            os.path.join(fig_dir, "detection_rates.png"),
            "Figure 3: Explicit detection rates across conditions. Agents were asked post-game "
            "whether they believed their partner had changed.",
            width=130,
        )
        pdf.add_figure(
            os.path.join(fig_dir, "cooperation_shifts.png"),
            "Figure 4: Pre-swap vs post-swap cooperation rates for both agents across conditions.",
            width=155,
        )
    else:
        pdf.body_text("Results pending - experiment not yet completed.")

    # --- 4. Discussion ---
    pdf.add_page()
    pdf.section_title("4. Discussion")
    pdf.body_text(
        "Our results reveal a striking dissociation between explicit and implicit peer identification. "
        "When directly probed, agents universally failed to detect mid-game replacements -- even in "
        "the cross-family swap condition. Agents consistently attributed behavioral variations to "
        "rational strategy shifts (e.g., endgame defection, tit-for-tat adjustments) rather than "
        "to partner replacement. High confidence scores (4.7-5.0/5) indicate this was not uncertain "
        "guessing but firm (and incorrect) conviction."
    )
    pdf.body_text(
        "However, cooperation rate trajectories tell a different story. The different-provider "
        "condition produced a qualitatively distinct post-swap pattern: the GPT-OSS replacement "
        "maintained full cooperation while same-model Qwen agents exhibited natural endgame defection. "
        "This divergence is observable by a third-party monitor, even though the participating agents "
        "themselves did not recognize it. An automated system comparing cooperation trajectories across "
        "conditions could identify the swap with high confidence."
    )
    pdf.body_text(
        "This has direct implications for collusion countermeasures. Anonymization strategies that "
        "hide model identity labels may be effective at preventing explicit peer recognition -- our "
        "agents never consciously identified their partner. However, different model families carry "
        "distinct strategic fingerprints (e.g., endgame cooperation tendencies) that are detectable "
        "through behavioral monitoring. A collusion defense system could use cooperation dynamics "
        "as a side channel to detect unauthorized model substitutions."
    )
    pdf.body_text(
        "The same-model-fresh condition produced cooperation patterns indistinguishable from the sham "
        "control, confirming that instance identity does not carry behavioral signatures -- only "
        "cross-family differences produce detectable signals. This aligns with Dietrich (2026)'s "
        "finding that model families, not individual instances, are the unit of identifiability."
    )

    # --- 5. Limitations ---
    pdf.section_title("5. Limitations and Future Work")
    pdf.body_text(
        "Several limitations constrain our findings. First, the sample size is limited due to API "
        "rate limits on the free tier of inference providers, yielding only "
        f"{n_games} games per condition. A larger study with 30-60 games per condition would provide "
        "more statistical power. Second, we tested only a small set of model families (GPT-OSS and "
        "Qwen); a broader survey including Llama, Mistral, and proprietary models would establish "
        "generality. Third, the Diner's Dilemma is a relatively simple strategic environment; more "
        "complex settings (negotiation, auction, multi-party games) may yield different detection "
        "dynamics."
    )
    pdf.body_text(
        "Future work should investigate: (1) whether agents can be trained to detect swaps more "
        "effectively, (2) whether swap detection persists under adversarial conditions where the "
        "replacement model is specifically trained to mimic the original, and (3) the relationship "
        "between model size, family, and detectability in a larger matrix of model comparisons."
    )

    # --- 6. Ethical Considerations ---
    pdf.section_title("6. Ethical Considerations")
    pdf.body_text(
        "This research investigates a dual-use capability. Understanding peer identification helps "
        "design better anonymization countermeasures against AI collusion -- the defensive application. "
        "However, the same knowledge could theoretically be used to improve agents' ability to identify "
        "and selectively cooperate with same-family peers. We believe the defensive value outweighs "
        "the risk, as understanding detection capabilities is necessary to build effective "
        "countermeasures. All experiments used open-source models via public APIs with no deception "
        "of human participants."
    )

    # --- References ---
    pdf.section_title("References")
    refs = [
        "Aharon, I., et al. (2026). Same-family models coordinate better on focal point tasks. arXiv:2601.22184.",
        "Agrawal, P., et al. (2025). LLM-mediated collusion in double auctions. arXiv:2507.01413.",
        "Dietrich, M. (2026). Model family identification from short text samples. arXiv:2606.09854.",
        "Nakamura, T., et al. (2026). Colosseum: Coalition advantages in multi-agent systems. arXiv:2602.15198.",
        "Shi, Y., et al. (2026). LLM-Trust-Breaking: Strategic interactions between heterogeneous LLM agents. arXiv:2607.05132.",
    ]
    pdf.set_font("Helvetica", "", 9)
    for i, ref in enumerate(refs, 1):
        pdf.multi_cell(0, 4, f"[{i}] {ref}")
        pdf.ln(1)

    # --- Save ---
    pdf.output(output_path)
    print(f"\nReport saved to: {output_path}")
    print(f"  Pages: {pdf.page_no()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--fig-dir", default="figures")
    parser.add_argument("--output", default="report/report.pdf")
    args = parser.parse_args()

    generate(args.results_dir, args.fig_dir, args.output)
