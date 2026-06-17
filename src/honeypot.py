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

Flags are returned as structured {"type": ..., "message": ...} dicts, not
bare strings -- a free-text message like "Hooli: stated duration_months=..."
is for a human reading one candidate, but aggregating *across* candidates
(which check type is firing, and how often) needs a stable category key
that doesn't depend on which company name happened to be in the sentence.
"""

from datetime import date, datetime

# Stable category codes, used for cross-candidate aggregation in rank.py's
# diagnostics. Keep these short and check-specific.
SKILL_ZERO_USAGE = "skill_zero_usage"
YOE_MISMATCH = "yoe_mismatch"
DATE_MATH_MISMATCH = "date_math_mismatch"
OVERLAPPING_ROLES = "overlapping_roles"
MULTIPLE_CURRENT_ROLES = "multiple_current_roles"
EDUCATION_TIMELINE_IMPOSSIBLE = "education_timeline_impossible"
EDUCATION_DATE_ORDER = "education_date_order"

# Checks that represent a hard structural impossibility (the data
# contradicts itself, full stop) vs. a soft statistical anomaly (could in
# principle be a legitimate edge case). Any hard flag alone is enough to
# call a profile a honeypot; soft flags need to co-occur.
HARD_TYPES = {DATE_MATH_MISMATCH, OVERLAPPING_ROLES, MULTIPLE_CURRENT_ROLES, EDUCATION_DATE_ORDER}


def _parse_date(d):
    if not d:
        return None
    return datetime.strptime(d, "%Y-%m-%d").date()


def detect_honeypot(candidate: dict) -> dict:
    """
    Returns {"flags": [{"type": str, "message": str}, ...], "score": int,
    "is_honeypot": bool}. score is the count of independent checks
    triggered. is_honeypot is True if any HARD_TYPES flag fired, or if two
    or more flags fired in total.
    """
    flags = []

    profile = candidate["profile"]
    history = candidate["career_history"]
    education = candidate.get("education", [])
    skills = candidate.get("skills", [])

    # --- Check 1: expert/advanced proficiency with ~zero time invested ---
    zero_duration_high_proficiency = [
        s for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and s.get("duration_months", 0) <= 1
        and s.get("endorsements", 0) == 0
    ]
    if len(zero_duration_high_proficiency) >= 3:
        flags.append({
            "type": SKILL_ZERO_USAGE,
            "message": (
                f"{len(zero_duration_high_proficiency)} skills listed as expert/advanced "
                f"with ~0 months of use and 0 endorsements"
            ),
        })

    # --- Check 2: years_of_experience inconsistent with career_history sum ---
    total_months = sum(ch.get("duration_months", 0) for ch in history)
    stated_years = profile.get("years_of_experience", 0)
    if abs(stated_years - total_months / 12) > 2.0:
        flags.append({
            "type": YOE_MISMATCH,
            "message": (
                f"stated years_of_experience ({stated_years}) doesn't match the sum of "
                f"career_history durations ({round(total_months / 12, 1)} years)"
            ),
        })

    # --- Check 3: date math doesn't support the stated duration_months ---
    for ch in history:
        sd = _parse_date(ch.get("start_date"))
        ed = _parse_date(ch.get("end_date")) or date.today()
        if sd is None:
            continue
        computed_months = (ed.year - sd.year) * 12 + (ed.month - sd.month)
        if abs(computed_months - ch.get("duration_months", 0)) > 3:
            flags.append({
                "type": DATE_MATH_MISMATCH,
                "message": (
                    f"{ch.get('company')}: stated duration_months={ch.get('duration_months')} "
                    f"doesn't match start/end dates (computed ~{computed_months} months)"
                ),
            })

    # --- Check 4: overlapping career history entries ---
    sorted_hist = sorted(
        (ch for ch in history if ch.get("start_date")),
        key=lambda c: c["start_date"],
    )
    for a, b in zip(sorted_hist, sorted_hist[1:]):
        a_end = a.get("end_date")
        if a_end and a_end > b["start_date"]:
            flags.append({
                "type": OVERLAPPING_ROLES,
                "message": f"overlapping roles: {a['company']} and {b['company']}",
            })

    # --- Check 5: more than one "is_current" role ---
    n_current = sum(1 for ch in history if ch.get("is_current"))
    if n_current > 1:
        flags.append({
            "type": MULTIPLE_CURRENT_ROLES,
            "message": f"{n_current} career_history entries marked is_current",
        })

    # --- Check 6: education timeline impossible given experience claim ---
    if education:
        latest_end_year = max((e.get("end_year", 0) for e in education), default=0)
        years_since_grad = date.today().year - latest_end_year
        if years_since_grad >= 0 and stated_years > years_since_grad + 2:
            flags.append({
                "type": EDUCATION_TIMELINE_IMPOSSIBLE,
                "message": (
                    f"claims {stated_years} years experience but graduated only "
                    f"~{years_since_grad} years ago"
                ),
            })
        for e in education:
            if e.get("end_year", 0) < e.get("start_year", 0):
                flags.append({
                    "type": EDUCATION_DATE_ORDER,
                    "message": f"education end_year before start_year: {e.get('institution')}",
                })

    score = len(flags)
    is_honeypot = any(f["type"] in HARD_TYPES for f in flags) or score >= 2

    return {"flags": flags, "score": score, "is_honeypot": is_honeypot}
