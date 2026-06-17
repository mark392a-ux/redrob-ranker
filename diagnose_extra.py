#!/usr/bin/env python3
"""
Standalone diagnostic, round 2. Probes a few more cross-field impossibility
hypotheses that the schema wouldn't reject syntactically but that don't
make logical sense -- the same flavor of "subtly impossible" as the skill
pattern found in diagnose_skills.py. Prints only aggregate counts.

Usage:
    python diagnose_extra.py candidates.jsonl
"""

import gzip
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


def load(path):
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_date(d):
    return datetime.strptime(d, "%Y-%m-%d").date() if d else None


def main():
    if len(sys.argv) != 2:
        print("Usage: python diagnose_extra.py candidates.jsonl")
        sys.exit(1)

    n = 0
    salary_min_gt_max = 0
    active_before_signup = 0
    skill_duration_exceeds_career = Counter()  # bucketed by how many months over
    skill_duration_exceeds_yoe = 0
    multiple_current_roles = 0
    overlapping_roles_strict = 0  # any overlap at all, even 1 day
    n_career_entries_dist = Counter()

    for c in load(sys.argv[1]):
        n += 1
        sig = c.get("redrob_signals", {})
        profile = c["profile"]
        history = c.get("career_history", [])
        skills = c.get("skills", [])

        # 1. salary band inverted
        salary = sig.get("expected_salary_range_inr_lpa", {})
        if salary and salary.get("min", 0) > salary.get("max", 0):
            salary_min_gt_max += 1

        # 2. active before they even signed up
        signup = parse_date(sig.get("signup_date"))
        last_active = parse_date(sig.get("last_active_date"))
        if signup and last_active and last_active < signup:
            active_before_signup += 1

        # 3. a single skill's duration exceeds total career history
        total_career_months = sum(ch.get("duration_months", 0) for ch in history)
        max_skill_duration = max((s.get("duration_months", 0) for s in skills), default=0)
        if max_skill_duration > total_career_months + 6:
            over_by = max_skill_duration - total_career_months
            bucket = min((over_by // 12) * 12, 60)  # bucket by years-over, cap at 60
            skill_duration_exceeds_career[bucket] += 1

        # 4. a single skill's duration exceeds stated years_of_experience
        yoe_months = profile.get("years_of_experience", 0) * 12
        if max_skill_duration > yoe_months + 6:
            skill_duration_exceeds_yoe += 1

        # 5. more than one is_current (re-checking with stricter intent)
        if sum(1 for ch in history if ch.get("is_current")) > 1:
            multiple_current_roles += 1

        # 6. ANY overlap at all between roles, not just the >0 day version
        sorted_hist = sorted((ch for ch in history if ch.get("start_date")), key=lambda x: x["start_date"])
        for a, b in zip(sorted_hist, sorted_hist[1:]):
            a_end = a.get("end_date")
            if a_end and a_end > b["start_date"]:
                overlapping_roles_strict += 1
                break

        n_career_entries_dist[len(history)] += 1

    print(f"Total candidates: {n}\n")
    print(f"salary min > max: {salary_min_gt_max}")
    print(f"last_active_date < signup_date: {active_before_signup}")
    print(f"multiple is_current roles: {multiple_current_roles}")
    print(f"any overlapping roles (candidate-level, not entry-level): {overlapping_roles_strict}")

    print("\nSkill duration exceeds total career_history sum by N+ years (bucketed):")
    for k in sorted(skill_duration_exceeds_career):
        print(f"  {k}+ years over: {skill_duration_exceeds_career[k]} candidates")

    print(f"\nSkill duration exceeds stated years_of_experience: {skill_duration_exceeds_yoe} candidates")

    print("\nCareer_history entry count distribution:")
    for k in sorted(n_career_entries_dist):
        print(f"  {k} entries: {n_career_entries_dist[k]} candidates")


if __name__ == "__main__":
    main()
