#!/usr/bin/env python3
"""
Standalone diagnostic -- run this directly against candidates.jsonl.
Prints only aggregate counts, no candidate-identifying data, so it's safe
to paste the output back into chat.

Goal: figure out what threshold actually isolates the "expert proficiency
with near-zero usage" honeypot pattern in the real dataset, since the
shipped skill_zero_usage check (>=3 skills, duration<=1mo, endorsements==0)
is catching zero candidates -- almost certainly too strict, not because
the pattern doesn't exist.

Usage:
    python diagnose_skills.py candidates.jsonl
"""

import gzip
import json
import sys
from collections import Counter
from pathlib import Path


def load(path):
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    if len(sys.argv) != 2:
        print("Usage: python diagnose_skills.py candidates.jsonl")
        sys.exit(1)

    n = 0
    # For each candidate, how many skills are (expert or advanced) AND duration_months==0,
    # broken out by whether endorsements==0 too, or any endorsement count.
    per_candidate_strict = Counter()   # duration==0, endorsements==0
    per_candidate_loose = Counter()    # duration==0, any endorsements
    per_candidate_duration_le1 = Counter()  # duration<=1, any endorsements
    proficiency_dist = Counter()
    duration_for_high_prof = Counter()  # histogram of duration_months, only for expert/advanced skills

    for c in load(sys.argv[1]):
        n += 1
        strict_count = 0
        loose_count = 0
        le1_count = 0
        for s in c.get("skills", []):
            prof = s.get("proficiency")
            proficiency_dist[prof] += 1
            dur = s.get("duration_months", 0)
            end = s.get("endorsements", 0)
            if prof in ("expert", "advanced"):
                duration_for_high_prof[min(dur, 24)] += 1  # bucket 24+ together
                if dur == 0:
                    loose_count += 1
                    if end == 0:
                        strict_count += 1
                if dur <= 1:
                    le1_count += 1
        per_candidate_strict[strict_count] += 1
        per_candidate_loose[loose_count] += 1
        per_candidate_duration_le1[le1_count] += 1

    print(f"Total candidates: {n}\n")

    print("Proficiency label distribution (all skills, all candidates):")
    for k, v in proficiency_dist.most_common():
        print(f"  {k}: {v}")

    print("\nDuration (months) histogram for expert/advanced skills only (24 = 24+):")
    for k in sorted(duration_for_high_prof):
        print(f"  {k:3d} months: {duration_for_high_prof[k]}")

    print("\nCandidates by count of (expert/advanced, duration==0, endorsements==0) skills [STRICT, current rule]:")
    for k in sorted(per_candidate_strict):
        print(f"  {k} such skills: {per_candidate_strict[k]} candidates")

    print("\nCandidates by count of (expert/advanced, duration==0, ANY endorsements) skills [LOOSE]:")
    for k in sorted(per_candidate_loose):
        print(f"  {k} such skills: {per_candidate_loose[k]} candidates")

    print("\nCandidates by count of (expert/advanced, duration<=1, ANY endorsements) skills:")
    for k in sorted(per_candidate_duration_le1):
        print(f"  {k} such skills: {per_candidate_duration_le1[k]} candidates")


if __name__ == "__main__":
    main()
