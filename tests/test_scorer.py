"""
Tests for src/scorer.py and src/rule_features.py.

These focus on the properties that matter most for the hackathon's
explicit grading criteria: that off-target titles score low regardless of
skills (the anti-keyword-stuffing guarantee), that disqualifier penalties
actually reduce the score, and that the final score stays within a sane,
bounded range no matter what's thrown at it.
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.scorer import score_candidate
from src.rule_features import title_match_score, disqualifier_penalty


def test_weights_sum_to_one():
    # If this drifts, every fit_score silently shifts scale -- catch it here
    # rather than discovering it from a confusing ranking later.
    assert abs(sum(config.WEIGHTS.values()) - 1.0) < 1e-9


def test_on_target_title_scores_higher_than_off_target(candidate):
    on_target_result = score_candidate(candidate, text_relevance=0.5)

    off_target = copy.deepcopy(candidate)
    off_target["profile"]["current_title"] = "HR Manager"
    off_target["career_history"][0]["title"] = "HR Manager"
    off_target_result = score_candidate(off_target, text_relevance=0.5)

    assert on_target_result["final_score"] > off_target_result["final_score"]


def test_keyword_stuffing_does_not_overcome_bad_title_match(candidate):
    # The core anti-stuffing guarantee: an HR Manager with a pile of
    # AI-sounding skills should still score below a genuine ML Engineer
    # with a clean, relevant skill set -- because title_match independently
    # gates the score, not just an average over text+skills.
    clean_result = score_candidate(candidate, text_relevance=0.5)

    stuffed = copy.deepcopy(candidate)
    stuffed["profile"]["current_title"] = "HR Manager"
    stuffed["career_history"][0]["title"] = "HR Manager"
    stuffed["career_history"][0]["description"] = "Managed HR operations and recruiting."
    stuffed["skills"] += [
        {"name": "Machine Learning", "proficiency": "expert", "endorsements": 1, "duration_months": 1},
        {"name": "RAG", "proficiency": "expert", "endorsements": 1, "duration_months": 1},
        {"name": "Vector Search", "proficiency": "expert", "endorsements": 1, "duration_months": 1},
    ]
    stuffed_result = score_candidate(stuffed, text_relevance=0.5)

    assert clean_result["final_score"] > stuffed_result["final_score"]


def test_consulting_only_career_is_penalized(candidate):
    candidate["profile"]["current_company"] = "TCS"
    candidate["career_history"][0]["company"] = "TCS"
    penalty, reasons = disqualifier_penalty(candidate)
    assert penalty > 0
    assert any("consulting" in r for r in reasons)


def test_no_penalty_for_clean_product_company_career(candidate):
    penalty, reasons = disqualifier_penalty(candidate)
    assert penalty == 0.0
    assert reasons == []


def test_title_match_score_is_bounded(candidate):
    score = title_match_score(candidate)
    assert 0.0 <= score <= 1.0


def test_final_score_is_non_negative(candidate):
    # Even a maximally-penalized candidate shouldn't produce a negative
    # score -- fit_score is explicitly floored at 0 in scorer.py.
    candidate["profile"]["current_company"] = "TCS"
    candidate["career_history"][0]["company"] = "TCS"
    candidate["profile"]["current_title"] = "HR Manager"
    candidate["career_history"][0]["title"] = "HR Manager"
    result = score_candidate(candidate, text_relevance=0.0)
    assert result["final_score"] >= 0.0
    assert result["fit_score"] >= 0.0


def test_behavioral_multiplier_is_within_documented_range(candidate):
    result = score_candidate(candidate, text_relevance=0.5)
    # behavioral.py documents a 0.4-1.0 floor/ceiling on the multiplier.
    assert 0.4 <= result["behavioral_multiplier"] <= 1.0
