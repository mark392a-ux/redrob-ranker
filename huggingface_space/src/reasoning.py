"""
Reasoning generation.

Deliberately NOT an LLM call. The submission spec's compute constraints
("no network during the ranking step") apply to whatever process produces
the final CSV, and the reasoning column is part of that CSV -- so this has
to be deterministic, local, and traceable to the same features the scorer
already computed. The upside: it can't hallucinate a skill the candidate
doesn't have, because every sentence is built from a field we already
extracted, and that's exactly what Stage 4 manual review checks for.
"""

from . import config


def build_reasoning(candidate: dict, score_result: dict) -> str:
    profile = candidate["profile"]
    comps = score_result["components"]
    sig = candidate.get("redrob_signals", {})

    title = profile.get("current_title", "Unknown title")
    years = profile.get("years_of_experience", "?")
    company = profile.get("current_company", "")

    clauses = [f"{title} with {years} yrs experience, currently at {company}."]

    if comps["title_match"] < 0.3:
        clauses.append("Title/role history doesn't match the AI-engineering target profile.")
    elif comps["production_signal"] >= 1.0:
        clauses.append("Career history shows evidence of shipping production systems to real users.")
    elif comps["production_signal"] == 0.0:
        clauses.append("Career history reads as research-only with no production deployment evidence.")

    if comps["skills_trust"] >= 0.6:
        clauses.append("Strong, well-substantiated match on core retrieval/ranking/embeddings skills.")
    elif comps["skills_trust"] <= 0.15:
        clauses.append("Little substantiated evidence of the core embeddings/retrieval/ranking skill set.")

    if score_result["penalty_reasons"]:
        clauses.append("Concern: " + "; ".join(score_result["penalty_reasons"]) + ".")

    response_rate = sig.get("recruiter_response_rate")
    if response_rate is not None:
        if response_rate < 0.2:
            clauses.append(f"Low recruiter response rate ({response_rate:.0%}) — may be hard to reach.")
        elif response_rate >= 0.6:
            clauses.append(f"Responsive to recruiters ({response_rate:.0%} response rate).")

    notice = sig.get("notice_period_days")
    if notice is not None and notice > 60:
        clauses.append(f"Long notice period ({notice} days).")

    return " ".join(clauses)
