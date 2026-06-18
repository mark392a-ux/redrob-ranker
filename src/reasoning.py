from . import config


def build_reasoning(candidate: dict, score_result: dict) -> str:
    profile = candidate["profile"]
    comps = score_result["components"]
    sig = candidate.get("redrob_signals", {})

    title = profile.get("current_title", "Unknown title")
    years = profile.get("years_of_experience", "?")
    company = profile.get("current_company", "")

    parts = [f"{title}, {years} yrs, currently at {company}."]

    if comps["title_match"] < 0.3:
        parts.append("Role/title history is off-target for this hire.")
    elif comps["production_signal"] >= 1.0:
        parts.append("Has shipped production systems, not just research or demos.")
    elif comps["production_signal"] == 0.0:
        parts.append("Reads research-only — no clear production deployment track record.")

    if comps["skills_trust"] >= 0.6:
        parts.append("Embeddings/retrieval/ranking skills look credible (tenure + endorsements back them up).")
    elif comps["skills_trust"] <= 0.15:
        parts.append("Weak evidence on the core retrieval/ranking stack.")

    if score_result["penalty_reasons"]:
        parts.append("Flags: " + "; ".join(score_result["penalty_reasons"]) + ".")

    response_rate = sig.get("recruiter_response_rate")
    if response_rate is not None:
        if response_rate < 0.2:
            parts.append(f"Low recruiter response rate ({response_rate:.0%}) — may be hard to reach.")
        elif response_rate >= 0.6:
            parts.append(f"Usually responds to recruiters ({response_rate:.0%}).")

    notice = sig.get("notice_period_days")
    if notice is not None and notice > 60:
        parts.append(f"{notice}-day notice period.")

    return " ".join(parts)
