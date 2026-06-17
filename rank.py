#!/usr/bin/env python3
"""
Usage:
    python rank.py --candidates ./candidates.jsonl --jd ./data/job_description.txt --out ./submission.csv

Single command that produces the submission CSV from the raw candidate
pool. Designed to stay well inside the compute constraints (CPU only, no
GPU, no network, <=16GB RAM, <=5 min wall clock) at 100K-candidate scale:
TF-IDF fit+transform over the whole pool is one sparse matrix operation,
and everything else is per-candidate dict access with no model inference.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.io_utils import load_candidates, load_jd_text
from src.honeypot import detect_honeypot
from src.text_features import candidate_text, build_relevance_scorer
from src.scorer import score_candidate
from src.reasoning import build_reasoning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--jd", default=str(Path(__file__).resolve().parent / "data" / "job_description.txt"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--top", type=int, default=100)
    args = parser.parse_args()

    t0 = time.time()

    candidates = load_candidates(args.candidates)
    jd_text = load_jd_text(args.jd)
    print(f"[1/5] Loaded {len(candidates)} candidates in {time.time() - t0:.1f}s", file=sys.stderr)

    # --- Honeypot screening, before any scoring work is spent on them ---
    t1 = time.time()
    honeypot_results = {c["candidate_id"]: detect_honeypot(c) for c in candidates}
    survivors = [c for c in candidates if not honeypot_results[c["candidate_id"]]["is_honeypot"]]
    n_honeypots = len(candidates) - len(survivors)
    print(
        f"[2/5] Honeypot screen: {n_honeypots} flagged / {len(candidates)} total "
        f"in {time.time() - t1:.1f}s",
        file=sys.stderr,
    )

    # --- Text relevance, fit once over the whole surviving pool ---
    t2 = time.time()
    texts = [candidate_text(c) for c in survivors]
    relevance_scores = build_relevance_scorer(jd_text, texts)
    print(f"[3/5] TF-IDF relevance scored in {time.time() - t2:.1f}s", file=sys.stderr)

    # --- Score every surviving candidate ---
    t3 = time.time()
    scored = []
    for c, rel in zip(survivors, relevance_scores):
        result = score_candidate(c, float(rel))
        scored.append((c, result))
    print(f"[4/5] Scored {len(scored)} candidates in {time.time() - t3:.1f}s", file=sys.stderr)

    # --- Rank, tie-break by candidate_id ascending per spec section 3 ---
    # Sort on the *rounded* score, not the raw float -- ties are defined by
    # what actually appears in the CSV (4 decimal places), not by precision
    # we then throw away. Sorting on the raw float let two candidates that
    # round to the same displayed score land in candidate_id-descending
    # order, which the validator correctly flags.
    scored.sort(key=lambda pair: (-round(pair[1]["final_score"], 4), pair[0]["candidate_id"]))
    top_n = scored[: args.top]

    if len(top_n) < args.top:
        print(
            f"WARNING: only {len(top_n)} candidates survived to rank, "
            f"need {args.top}. Pool is too small or honeypot filter too aggressive.",
            file=sys.stderr,
        )

    t4 = time.time()
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (c, result) in enumerate(top_n, start=1):
            reasoning = build_reasoning(c, result)
            writer.writerow([c["candidate_id"], rank, round(result["final_score"], 4), reasoning])
    print(f"[5/5] Wrote {args.out} in {time.time() - t4:.1f}s", file=sys.stderr)

    total = time.time() - t0
    print(f"\nTotal wall-clock time: {total:.1f}s (budget: 300s)", file=sys.stderr)
    if total > 300:
        print("WARNING: exceeded the 5-minute compute budget.", file=sys.stderr)


if __name__ == "__main__":
    main()
