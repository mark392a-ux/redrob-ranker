"""
Behavioral signal multiplier.

Per the JD: "a perfect-on-paper candidate who hasn't logged in for 6 months
and has a 5% recruiter response rate is, for hiring purposes, not actually
available." This is modeled as a multiplier (0.4 to 1.0) on top of the fit
score, not blended additively into it -- a candidate with a great fit score
and terrible engagement should drop sharply, not just lose a few points.
"""

from datetime import date, datetime


def _days_since(date_str: str) -> int:
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    return (date.today() - d).days


def behavioral_multiplier(candidate: dict) -> float:
    sig = candidate.get("redrob_signals", {})

    # Recency of activity: full credit inside 30 days, decaying to a floor
    # by 180+ days inactive.
    days_inactive = _days_since(sig["last_active_date"]) if sig.get("last_active_date") else 365
    if days_inactive <= 30:
        recency = 1.0
    elif days_inactive >= 180:
        recency = 0.3
    else:
        recency = 1.0 - 0.7 * (days_inactive - 30) / 150

    response_rate = sig.get("recruiter_response_rate", 0.0)
    interview_completion = sig.get("interview_completion_rate", 0.0)
    open_to_work = 1.0 if sig.get("open_to_work_flag") else 0.7

    # Notice period: JD explicitly says 30+ day candidates are "still in
    # scope but the bar gets higher" -- a soft discount, not a cutoff.
    notice_days = sig.get("notice_period_days", 30)
    notice_factor = 1.0 if notice_days <= 30 else max(0.6, 1.0 - (notice_days - 30) / 300)

    weighted = (
        0.35 * recency
        + 0.25 * response_rate
        + 0.15 * interview_completion
        + 0.15 * open_to_work
        + 0.10 * notice_factor
    )
    # Floor at 0.4 so a single weak signal can't zero out an otherwise
    # strong candidate -- this is a modifier, not a disqualifier.
    return max(0.4, min(1.0, weighted))
