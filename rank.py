#!/usr/bin/env python3
"""Rank candidates against a JD and write the submission CSV."""

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
    parser.add_argument("--debug-honeypots", default=None)
    args = parser.parse_args()

    t0 = time.time()

    candidates = load_candidates(args.candidates)
    jd_text = load_jd_text(args.jd)
    print(f"[1/5] loaded {len(candidates)} candidates ({time.time() - t0:.1f}s)", file=sys.stderr)

    t1 = time.time()
    honeypot_results = {c["candidate_id"]: detect_honeypot(c) for c in candidates}
    survivors = [c for c in candidates if not honeypot_results[c["candidate_id"]]["is_honeypot"]]
    n_honeypots = len(candidates) - len(survivors)
    print(
        f"[2/5] honeypots: {n_honeypots}/{len(candidates)} ({time.time() - t1:.1f}s)",
        file=sys.stderr,
    )

    flag_type_counts = {}
    for c in candidates:
        result = honeypot_results[c["candidate_id"]]
        if result["is_honeypot"]:
            for flag in result["flags"]:
                flag_type_counts[flag["type"]] = flag_type_counts.get(flag["type"], 0) + 1
    if flag_type_counts:
        print("      flag breakdown:", file=sys.stderr)
        for k, v in sorted(flag_type_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"        {v:5d}  {k}", file=sys.stderr)

    if args.debug_honeypots:
        with open(args.debug_honeypots, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["candidate_id", "flag_count", "flag_types", "details"])
            for c in candidates:
                result = honeypot_results[c["candidate_id"]]
                if result["is_honeypot"]:
                    types = ",".join(fl["type"] for fl in result["flags"])
                    details = " | ".join(fl["message"] for fl in result["flags"])
                    writer.writerow([c["candidate_id"], result["score"], types, details])
        print(f"      wrote {args.debug_honeypots}", file=sys.stderr)

    t2 = time.time()
    texts = [candidate_text(c) for c in survivors]
    relevance_scores = build_relevance_scorer(jd_text, texts)
    print(f"[3/5] tf-idf done ({time.time() - t2:.1f}s)", file=sys.stderr)

    t3 = time.time()
    scored = []
    for c, rel in zip(survivors, relevance_scores):
        result = score_candidate(c, float(rel))
        scored.append((c, result))
    print(f"[4/5] scored {len(scored)} ({time.time() - t3:.1f}s)", file=sys.stderr)

    # tie-break on rounded score + candidate_id (spec section 3)
    scored.sort(key=lambda pair: (-round(pair[1]["final_score"], 4), pair[0]["candidate_id"]))
    top_n = scored[: args.top]

    if len(top_n) < args.top:
        print(
            f"WARNING: only {len(top_n)} candidates to rank (need {args.top})",
            file=sys.stderr,
        )

    t4 = time.time()
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (c, result) in enumerate(top_n, start=1):
            reasoning = build_reasoning(c, result)
            writer.writerow([c["candidate_id"], rank, round(result["final_score"], 4), reasoning])
    print(f"[5/5] wrote {args.out} ({time.time() - t4:.1f}s)", file=sys.stderr)

    top_titles = {}
    for c, _ in top_n:
        title = c["profile"].get("current_title", "Unknown")
        top_titles[title] = top_titles.get(title, 0) + 1
    scores = [round(result["final_score"], 4) for _, result in top_n]
    print("\n      top titles:", file=sys.stderr)
    for title, count in sorted(top_titles.items(), key=lambda x: -x[1])[:12]:
        print(f"        {count:3d}  {title}", file=sys.stderr)
    if scores:
        print(
            f"      scores: max={max(scores):.4f} min={min(scores):.4f} "
            f"median={sorted(scores)[len(scores)//2]:.4f}",
            file=sys.stderr,
        )

    total = time.time() - t0
    print(f"\ntotal: {total:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
