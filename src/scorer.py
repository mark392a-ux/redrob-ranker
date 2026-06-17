"""
Final scoring: combine rule-based fit components + TF-IDF text relevance
into a base "fit score", subtract disqualifier penalties, then apply the
behavioral multiplier on top.

    fit_score    = weighted_sum(title, text, skills, experience, production,
                                 location, education) - disqualifier_penalty
    final_score  = fit_score * behavioral_multiplier

Honeypots are filtered separately (see honeypot.py) before this even runs --
this module assumes the candidate has already passed that check.
"""

from . import config
from . import rule_features as rf
from .behavioral import behavioral_multiplier


def score_candidate(candidate: dict, text_relevance: float) -> dict:
    components = {
        "title_match": rf.title_match_score(candidate),
        "text_relevance": text_relevance,
        "skills_trust": rf.skills_trust_score(candidate),
        "experience_band": rf.experience_band_score(candidate),
        "production_signal": rf.production_signal_score(candidate),
        "location": rf.location_score(candidate),
        "education_tier": rf.education_tier_score(candidate),
    }

    weighted_sum = sum(components[k] * config.WEIGHTS[k] for k in config.WEIGHTS)
    penalty, penalty_reasons = rf.disqualifier_penalty(candidate)
    fit_score = max(0.0, weighted_sum - penalty)

    multiplier = behavioral_multiplier(candidate)
    final_score = fit_score * multiplier

    return {
        "final_score": final_score,
        "fit_score": fit_score,
        "behavioral_multiplier": multiplier,
        "components": components,
        "penalty": penalty,
        "penalty_reasons": penalty_reasons,
    }
