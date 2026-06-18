"""
Redrob Hackathon Track 1 — interactive sandbox.

Loads either the bundled 50-candidate sample or a user-uploaded small
JSON/JSONL file, runs the real ranking pipeline (the same code as
rank.py / the GitHub repo, not a reimplementation), and displays the
ranked output, honeypot screening, and timing in the browser.

This exists to satisfy the hackathon's required "sandbox" submission item:
a hosted environment where a reviewer can run the ranker on a small
sample without setting up a local Python environment.
"""

import json
import time
from pathlib import Path

import gradio as gr
import pandas as pd

from src.io_utils import load_jd_text
from src.honeypot import detect_honeypot
from src.text_features import candidate_text, build_relevance_scorer
from src.scorer import score_candidate
from src.reasoning import build_reasoning

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = BASE_DIR / "sample_candidates.jsonl"
JD_PATH = BASE_DIR / "data" / "job_description.txt"
GITHUB_REPO_URL = "https://github.com/mark392a-ux/redrob-ranker"

# This Space is a small-sample sandbox per the hackathon's spec (Section
# 10.5) -- it's meant to prove the code runs end-to-end, not to reproduce
# the full 100K-scale evaluation (that's what the timing/memory benchmark
# in the GitHub README documents). Capping uploads here avoids someone
# accidentally uploading the real candidates.jsonl and hanging the only
# worker this free-tier Space has.
MAX_CANDIDATES = 500


def _load_jsonl_or_json(path: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def run_ranking(uploaded_file, top_n):
    t0 = time.time()

    if uploaded_file is not None:
        try:
            candidates = _load_jsonl_or_json(uploaded_file.name)
        except Exception as e:
            return (
                pd.DataFrame(),
                f"Could not parse the uploaded file: {e}",
                "",
            )
        if len(candidates) > MAX_CANDIDATES:
            original_count = len(candidates)
            candidates = candidates[:MAX_CANDIDATES]
            source_note = (
                f"Uploaded file had {original_count} candidates -- this sandbox is "
                f"intentionally capped at {MAX_CANDIDATES} for a quick interactive demo "
                f"(per the hackathon's 'small sample' sandbox requirement). Truncated to "
                f"the first {MAX_CANDIDATES}. The full 100K-scale run, with timing and "
                f"memory benchmarks, is documented in the GitHub repo linked below."
            )
        else:
            source_note = f"Loaded {len(candidates)} candidates from your upload."
    else:
        candidates = _load_jsonl_or_json(str(SAMPLE_PATH))
        source_note = f"Using the bundled {len(candidates)}-candidate sample (no file uploaded)."

    top_n = min(int(top_n), len(candidates))
    jd_text = load_jd_text(str(JD_PATH))

    # --- Honeypot screening ---
    honeypot_results = {c["candidate_id"]: detect_honeypot(c) for c in candidates}
    survivors = [c for c in candidates if not honeypot_results[c["candidate_id"]]["is_honeypot"]]
    n_honeypots = len(candidates) - len(survivors)

    flag_type_counts = {}
    for c in candidates:
        result = honeypot_results[c["candidate_id"]]
        if result["is_honeypot"]:
            for flag in result["flags"]:
                flag_type_counts[flag["type"]] = flag_type_counts.get(flag["type"], 0) + 1

    # --- Score and rank ---
    texts = [candidate_text(c) for c in survivors]
    relevance_scores = build_relevance_scorer(jd_text, texts) if survivors else []

    scored = []
    for c, rel in zip(survivors, relevance_scores):
        result = score_candidate(c, float(rel))
        scored.append((c, result))
    scored.sort(key=lambda pair: (-round(pair[1]["final_score"], 4), pair[0]["candidate_id"]))

    top = scored[:top_n]
    rows = []
    for rank, (c, result) in enumerate(top, start=1):
        rows.append({
            "rank": rank,
            "candidate_id": c["candidate_id"],
            "score": round(result["final_score"], 4),
            "title": c["profile"].get("current_title", ""),
            "years_exp": c["profile"].get("years_of_experience", ""),
            "reasoning": build_reasoning(c, result),
        })
    df = pd.DataFrame(rows)

    elapsed = time.time() - t0
    summary_lines = [
        source_note,
        f"Honeypot screen: {n_honeypots} flagged out of {len(candidates)} total.",
    ]
    if flag_type_counts:
        breakdown = ", ".join(f"{v} {k}" for k, v in sorted(flag_type_counts.items(), key=lambda x: -x[1]))
        summary_lines.append(f"Flag breakdown: {breakdown}")
    summary_lines.append(f"Ranked {len(survivors)} surviving candidates in {elapsed:.2f}s (this small-sample run, not the 100K timing benchmark in the repo).")
    summary = "\n\n".join(summary_lines)

    architecture = (
        "**Pipeline:** honeypot screen → TF-IDF relevance vs the JD → rule-based "
        "features (title match, skills trust, production-vs-research signal, "
        "experience band, location, education) → behavioral multiplier from "
        "redrob_signals → template-built reasoning. Full source and the "
        "100K-scale benchmark are in the GitHub repo linked below."
    )

    return df, summary, architecture


with gr.Blocks(title="Redrob Ranker Sandbox") as demo:
    gr.Markdown(
        f"""
        # Redrob Hackathon Track 1 — Candidate Ranker Sandbox

        Run the actual ranking pipeline (same code as the GitHub repo) on a
        small candidate sample. Upload your own `.jsonl`/`.json` file
        matching the challenge's candidate schema, or just click **Run** to
        use the bundled 50-candidate sample.

        Full source, the 100K-scale compute benchmark, and the design
        rationale for every scoring component: [{GITHUB_REPO_URL}]({GITHUB_REPO_URL})
        """
    )

    with gr.Row():
        file_input = gr.File(label="Upload a small candidate sample (.jsonl or .json) — optional", file_types=[".json", ".jsonl"])
        top_n_input = gr.Slider(minimum=1, maximum=50, value=20, step=1, label="Top N to show")

    run_button = gr.Button("Run Ranking", variant="primary")

    summary_output = gr.Markdown(label="Run summary")
    output_table = gr.Dataframe(label="Ranked output", wrap=True)
    architecture_output = gr.Markdown()

    run_button.click(
        fn=run_ranking,
        inputs=[file_input, top_n_input],
        outputs=[output_table, summary_output, architecture_output],
    )

    demo.load(fn=run_ranking, inputs=[file_input, top_n_input], outputs=[output_table, summary_output, architecture_output])


if __name__ == "__main__":
    demo.launch()
