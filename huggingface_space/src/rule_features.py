"""
Rule-based features extracted directly from the JD's own stated criteria
(config.py). These exist because TF-IDF similarity alone can't distinguish
"this person's title and career history say they did this job" from "this
person's skills section mentions the same words" -- and the JD explicitly
calls out that distinction as the trap.
"""

import math
from . import config


def _text_blob(candidate: dict) -> str:
    profile = candidate["profile"]
    parts = [profile.get("headline", ""), profile.get("summary", "")]
    for ch in candidate.get("career_history", []):
        parts.append(ch.get("description", ""))
    return " ".join(p for p in parts if p).lower()


def _any_term(text: str, terms: list[str]) -> bool:
    return any(t in text for t in terms)


def title_match_score(candidate: dict) -> float:
    """1.0 if current title (or recent past titles) clearly match the target
    role family; 0.0 if it clearly matches an off-target title; 0.5 if
    ambiguous/unseen. This is the decisive signal against keyword-stuffer
    traps -- an HR Manager with ten AI skills listed still scores low here."""
    titles = [candidate["profile"].get("current_title", "").lower()]
    titles += [ch.get("title", "").lower() for ch in candidate.get("career_history", [])[:2]]
    combined = " ".join(titles)

    on_target = _any_term(combined, config.TARGET_TITLE_TERMS)
    off_target = _any_term(combined, config.OFF_TARGET_TITLE_TERMS)

    if on_target and not off_target:
        return 1.0
    if off_target and not on_target:
        return 0.0
    if on_target and off_target:
        return 0.4  # mixed signal, e.g. a title change mid-career
    return 0.5  # neither list matched, stay neutral rather than penalize


def skills_trust_score(candidate: dict) -> float:
    """Weighted match against REQUIRED_SKILL_GROUPS + NICE_TO_HAVE_TERMS,
    where each matching skill is discounted if it shows little real usage.
    A skill listed as 'expert' with 0 months duration and 0 endorsements
    contributes almost nothing -- this directly fights lazy keyword
    stuffing in the skills list, independent of the honeypot filter."""
    skill_names = {s.get("name", "").lower(): s for s in candidate.get("skills", [])}
    proficiency_weight = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}

    def trust(skill: dict) -> float:
        prof = proficiency_weight.get(skill.get("proficiency"), 0.5)
        duration_factor = min(1.0, math.sqrt((skill.get("duration_months", 0) + 1) / 24))
        endorsement_factor = min(1.0, math.log1p(skill.get("endorsements", 0)) / math.log1p(20))
        return prof * (0.5 + 0.35 * duration_factor + 0.15 * endorsement_factor)

    group_scores = []
    for terms in config.REQUIRED_SKILL_GROUPS.values():
        matches = [skill_names[name] for term in terms for name in skill_names if term in name]
        group_scores.append(max((trust(s) for s in matches), default=0.0))
    required_score = sum(group_scores) / len(group_scores) if group_scores else 0.0

    nice_matches = [
        skill_names[name] for term in config.NICE_TO_HAVE_TERMS for name in skill_names if term in name
    ]
    nice_score = max((trust(s) for s in nice_matches), default=0.0)

    return min(1.0, 0.85 * required_score + 0.15 * nice_score)


def production_signal_score(candidate: dict) -> float:
    """Positive evidence of shipping to real users vs. research-only or
    wrapper-only signal, per the JD's explicit 'what we mean by 5-9 years'
    disqualifier section."""
    text = _text_blob(candidate)
    has_production = _any_term(text, config.PRODUCTION_EVIDENCE_TERMS)
    has_research_only = _any_term(text, config.RESEARCH_ONLY_TERMS) and not has_production
    if has_research_only:
        return 0.0
    return 1.0 if has_production else 0.5


def experience_band_score(candidate: dict) -> float:
    years = candidate["profile"].get("years_of_experience", 0)
    lo, hi = config.EXPERIENCE_BAND
    if lo <= years <= hi:
        return 1.0
    distance = (lo - years) if years < lo else (years - hi)
    return max(0.0, 1.0 - distance * config.EXPERIENCE_DECAY_PER_YEAR)


def location_score(candidate: dict) -> float:
    loc = candidate["profile"].get("location", "").lower()
    if any(city in loc for city in config.PREFERRED_LOCATIONS):
        return 1.0
    if candidate.get("redrob_signals", {}).get("willing_to_relocate"):
        return 0.7
    return 0.4


def education_tier_score(candidate: dict) -> float:
    education = candidate.get("education", [])
    if not education:
        return config.EDUCATION_TIER_SCORE["unknown"]
    best_tier = min(
        (e.get("tier", "unknown") for e in education),
        key=lambda t: ["tier_1", "tier_2", "tier_3", "tier_4", "unknown"].index(t)
        if t in ["tier_1", "tier_2", "tier_3", "tier_4"] else 4,
    )
    return config.EDUCATION_TIER_SCORE.get(best_tier, 0.4)


def disqualifier_penalty(candidate: dict) -> tuple[float, list[str]]:
    """Returns (total_penalty, reasons). Penalties are additive and floored
    when applied to the final score, not here."""
    text = _text_blob(candidate)
    companies = {ch.get("company", "").lower() for ch in candidate.get("career_history", [])}
    companies.add(candidate["profile"].get("current_company", "").lower())

    penalty = 0.0
    reasons = []

    if companies and companies.issubset(config.CONSULTING_FIRMS):
        penalty += config.PENALTIES["consulting_only"]
        reasons.append("entire career history is at IT-services/consulting firms")

    if _any_term(text, config.NON_NLP_DOMAINS) and not _any_term(text, config.NLP_IR_EVIDENCE):
        penalty += config.PENALTIES["non_nlp_domain"]
        reasons.append("primary domain is CV/speech/robotics with no NLP/IR evidence")

    recent_history = [ch for ch in candidate.get("career_history", []) if ch.get("is_current")]
    recent_months = recent_history[0].get("duration_months", 0) if recent_history else 0
    if (
        _any_term(text, config.WRAPPER_ONLY_TERMS)
        and recent_months < 12
        and not _any_term(text, config.PRE_LLM_ML_EVIDENCE)
    ):
        penalty += config.PENALTIES["wrapper_only_recent"]
        reasons.append("recent (<12mo) LLM-wrapper-only experience with no pre-LLM ML background")

    if _any_term(text, config.RESEARCH_ONLY_TERMS) and not _any_term(text, config.PRODUCTION_EVIDENCE_TERMS):
        penalty += config.PENALTIES["research_only"]
        reasons.append("research-only career history, no production deployment evidence")

    return penalty, reasons
