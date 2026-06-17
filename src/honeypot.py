"""
Honeypot detection.

The submission spec gives two concrete examples of what a honeypot looks
like: "8 years of experience at a company founded 3 years ago" and "expert
proficiency in 10 skills with 0 years used". We can't check the first
pattern literally (company founding dates aren't in the schema, and this
dataset uses fictional company names), so we implement the *general form*
of both examples: internal inconsistency between a claimed fact and the
structured data that should support it.

Each check below is cheap, vectorizable, and explainable in one sentence --
that's a deliberate constraint, since Stage 4 reviewers and the Stage 5
interview will ask "how do you know this is a honeypot" and "templated
heuristic on field X vs field Y" is a much better answer than "the model
decided".
"""

from datetime import date, datetime


def _parse_date(d):
    if not d:
        return None
    return datetime.strptime(d, "%Y-%m-%d").date()


def detect_honeypot(candidate: dict) -> dict:
    """
    Returns {"flags": [str, ...], "score": int, "is_honeypot": bool}.
    score is the count of independent checks triggered. is_honeypot is True
    if score >= 2, or if any single check is a "hard" structural
    impossibility (date math, overlapping roles, multiple concurrent jobs).
    """
    flags = []
    hard_flags = []

    profile = candidate["profile"]
    history = candidate["career_history"]
    education = candidate.get("education", [])
    skills = candidate.get("skills", [])

    # --- Check 1: expert/advanced proficiency with ~zero time invested ---
    # ("expert proficiency in 10 skills with 0 years used")
    zero_duration_high_proficiency = [
        s for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and s.get("duration_months", 0) <= 1
        and s.get("endorsements", 0) == 0
    ]
    if len(zero_duration_high_proficiency) >= 3:
        flags.append(
            f"{len(zero_duration_high_proficiency)} skills listed as expert/advanced "
            f"with ~0 months of use and 0 endorsements"
        )

    # --- Check 2: years_of_experience inconsistent with career_history sum ---
    total_months = sum(ch.get("duration_months", 0) for ch in history)
    stated_years = profile.get("years_of_experience", 0)
    if abs(stated_years - total_months / 12) > 2.0:
        flags.append(
            f"stated years_of_experience ({stated_years}) doesn't match the sum of "
            f"career_history durations ({round(total_months / 12, 1)} years)"
        )

    # --- Check 3: date math doesn't support the stated duration_months ---
    # ("8 years of experience at a company founded 3 years ago" is, in
    # general form, a claim that contradicts dates the candidate themselves
    # provided)
    for ch in history:
        sd = _parse_date(ch.get("start_date"))
        ed = _parse_date(ch.get("end_date")) or date.today()
        if sd is None:
            continue
        computed_months = (ed.year - sd.year) * 12 + (ed.month - sd.month)
        if abs(computed_months - ch.get("duration_months", 0)) > 3:
            hard_flags.append(
                f"{ch.get('company')}: stated duration_months={ch.get('duration_months')} "
                f"doesn't match start/end dates (computed ~{computed_months} months)"
            )

    # --- Check 4: overlapping career history entries ---
    sorted_hist = sorted(
        (ch for ch in history if ch.get("start_date")),
        key=lambda c: c["start_date"],
    )
    for a, b in zip(sorted_hist, sorted_hist[1:]):
        a_end = a.get("end_date")
        if a_end and a_end > b["start_date"]:
            hard_flags.append(f"overlapping roles: {a['company']} and {b['company']}")

    # --- Check 5: more than one "is_current" role ---
    n_current = sum(1 for ch in history if ch.get("is_current"))
    if n_current > 1:
        hard_flags.append(f"{n_current} career_history entries marked is_current")

    # --- Check 6: education timeline impossible given experience claim ---
    if education:
        latest_end_year = max((e.get("end_year", 0) for e in education), default=0)
        years_since_grad = date.today().year - latest_end_year
        if years_since_grad >= 0 and stated_years > years_since_grad + 2:
            flags.append(
                f"claims {stated_years} years experience but graduated only "
                f"~{years_since_grad} years ago"
            )
        for e in education:
            if e.get("end_year", 0) < e.get("start_year", 0):
                hard_flags.append(f"education end_year before start_year: {e.get('institution')}")

    all_flags = flags + hard_flags
    score = len(all_flags)
    is_honeypot = bool(hard_flags) or score >= 2

    return {"flags": all_flags, "score": score, "is_honeypot": is_honeypot}
