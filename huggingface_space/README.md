---
title: Redrob Ranker Sandbox
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
pinned: false
---

# Redrob Hackathon Track 1 — Ranker Sandbox

Interactive demo of the candidate ranking pipeline submitted for the
Redrob Intelligent Candidate Discovery & Ranking Challenge (Track 1).

Click **Run Ranking** to score the bundled 50-candidate sample against the
Senior AI Engineer job description, or upload your own small
`.jsonl`/`.json` file matching the challenge's candidate schema (capped
at 500 candidates for a fast interactive demo).

This runs the same scoring code as the GitHub repository — TF-IDF
relevance, rule-based title/skills/production-signal features, a
behavioral-signal multiplier, and structured honeypot detection — not a
simplified reimplementation for the demo.

## What you'll see

- **Honeypot screen** — structurally-impossible profiles (bad date math,
  overlapping roles, zero-usage "expert" skills, etc.) filtered out
  before scoring, with a breakdown by check type
- **Ranked table** — candidate ID, score, title, years of experience,
  company, and a full reasoning sentence per candidate
- **CSV download** — export the demo run's ranked output

On the real 100K-candidate pool (run locally, not in this sandbox): ~61
honeypots caught, ~55-70s runtime, ~3.5GB peak memory — see the GitHub
repo's README for the full methodology and the empirical investigation
behind the honeypot detection thresholds.

Full source, the 100K-scale compute benchmark, and the design rationale
for every scoring decision: see the GitHub repo linked in the app.
