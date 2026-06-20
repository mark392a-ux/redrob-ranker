"""
Redrob Hackathon Track 1 — interactive sandbox.

Loads either the bundled 50-candidate sample or a user-uploaded small
JSON/JSONL file, runs the real ranking pipeline (the same code as
rank.py / the GitHub repo, not a reimplementation), and displays the
ranked output, honeypot screening, and timing in the browser.
"""

import json
import tempfile
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
MAX_CANDIDATES = 500
JD_ROLE = "Senior AI Engineer — Retrieval, Ranking & Embeddings (Founding Team)"


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
            return pd.DataFrame(), f"❌ Could not parse file: {e}", "", None

        if len(candidates) > MAX_CANDIDATES:
            original_count = len(candidates)
            candidates = candidates[:MAX_CANDIDATES]
            source_note = (
                f"📁 Uploaded {original_count} candidates — capped at {MAX_CANDIDATES} "
                f"for this sandbox demo. Full 100K benchmark: see GitHub repo."
            )
        else:
            source_note = f"📁 Loaded **{len(candidates)}** candidates from your upload."
    else:
        candidates = _load_jsonl_or_json(str(SAMPLE_PATH))
        source_note = f"📁 Using the bundled **{len(candidates)}-candidate** sample."

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
            "Rank": rank,
            "Candidate ID": c["candidate_id"],
            "Score": round(result["final_score"], 4),
            "Title": c["profile"].get("current_title", ""),
            "Yrs Exp": c["profile"].get("years_of_experience", ""),
            "Company": c["profile"].get("current_company", ""),
            "Reasoning": build_reasoning(c, result),
        })
    df = pd.DataFrame(rows)

    # CSV export -- written fresh each run, named so it's clear this is a
    # sandbox demo run, not the actual hackathon submission CSV (which has
    # a different required filename/format, validated by validate_submission.py)
    csv_path = Path(tempfile.gettempdir()) / "redrob_sandbox_ranked_output.csv"
    df.to_csv(csv_path, index=False)

    elapsed = time.time() - t0

    # Flag breakdown
    if flag_type_counts:
        flag_lines = " | ".join(
            f"**{k.replace('_',' ')}**: {v}"
            for k, v in sorted(flag_type_counts.items(), key=lambda x: -x[1])
        )
    else:
        flag_lines = "none"

    summary = f"""{source_note}

---
| Metric | Value |
|--------|-------|
| 🚨 Honeypots flagged | **{n_honeypots}** of {len(candidates)} |
| ✅ Candidates scored | **{len(survivors)}** |
| ⏱ Runtime | **{elapsed:.2f}s** |
| 🏆 Showing top | **{top_n}** |

**Flag types caught:** {flag_lines}"""

    pipeline_note = (
        f"**Role ranked against:** {JD_ROLE}\n\n"
        "**Pipeline:** Honeypot screen → TF-IDF relevance vs JD → "
        "Rule-based features (title match · skills trust · production signal · "
        "experience band · location · education) → Behavioral multiplier "
        "(redrob_signals) → Template reasoning.\n\n"
        f"📦 Full source + 100K benchmark: [{GITHUB_REPO_URL}]({GITHUB_REPO_URL})"
    )

    return df, summary, pipeline_note, str(csv_path)


CSS = """
.gr-dataframe table { font-size: 13px; }
.gr-dataframe td:last-child { min-width: 380px; white-space: normal !important; word-wrap: break-word; }
"""

with gr.Blocks(title="Redrob Ranker Sandbox") as demo:

    gr.Markdown(f"""
# 🎯 Redrob Hackathon Track 1 — Candidate Ranker Sandbox

> **Role:** {JD_ROLE}

Upload a `.json` or `.jsonl` candidate file (up to 500), or click **Run Ranking**
to score the bundled 50-candidate sample. The full ranking pipeline runs
live — same code as the GitHub repo, no shortcuts.
""")

    with gr.Row():
        with gr.Column(scale=2):
            file_input = gr.File(
                label="📂 Upload candidate file (.json or .jsonl) — optional",
                file_types=[".json", ".jsonl"]
            )
        with gr.Column(scale=1):
            top_n_input = gr.Slider(
                minimum=1, maximum=50, value=20, step=1,
                label="Top N results to display"
            )

    run_button = gr.Button("🚀 Run Ranking", variant="primary", size="lg")

    summary_output = gr.Markdown(label="Results summary")

    output_table = gr.Dataframe(
        label="📊 Ranked candidates",
        wrap=True,
        column_widths=["5%", "12%", "7%", "14%", "8%", "14%", "40%"]
    )

    download_button = gr.DownloadButton(
        label="⬇️ Download ranked results (CSV)",
        variant="secondary",
    )

    pipeline_output = gr.Markdown()

    run_button.click(
        fn=run_ranking,
        inputs=[file_input, top_n_input],
        outputs=[output_table, summary_output, pipeline_output, download_button],
    )

    demo.load(
        fn=run_ranking,
        inputs=[file_input, top_n_input],
        outputs=[output_table, summary_output, pipeline_output, download_button],
    )


if __name__ == "__main__":
    demo.launch(css=CSS)
