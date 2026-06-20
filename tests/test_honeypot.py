"""
Tests for src/honeypot.py.

Each check gets two tests: one confirming it fires when the specific
impossibility is introduced, and (via test_clean_candidate_has_no_flags)
confirmation that a normal, internally-consistent profile doesn't trip
any check. That second part matters as much as the positive cases -- a
honeypot filter that's too trigger-happy disqualifies real candidates,
which is its own failure mode.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.honeypot import detect_honeypot, SKILL_ZERO_USAGE, YOE_MISMATCH, \
    DATE_MATH_MISMATCH, OVERLAPPING_ROLES, MULTIPLE_CURRENT_ROLES, \
    EDUCATION_TIMELINE_IMPOSSIBLE, EDUCATION_DATE_ORDER


def test_clean_candidate_has_no_flags(candidate):
    result = detect_honeypot(candidate)
    assert result["flags"] == []
    assert result["is_honeypot"] is False
    assert result["score"] == 0


def test_skill_zero_usage_fires_at_three_or_more(candidate):
    # 3 skills marked expert/advanced with 0 months -- the threshold found
    # empirically against the real 100K pool's bimodal distribution.
    candidate["skills"] += [
        {"name": "PyTorch", "proficiency": "expert", "endorsements": 5, "duration_months": 0},
        {"name": "Pinecone", "proficiency": "expert", "endorsements": 3, "duration_months": 0},
        {"name": "Qdrant", "proficiency": "advanced", "endorsements": 2, "duration_months": 0},
    ]
    result = detect_honeypot(candidate)
    flag_types = [f["type"] for f in result["flags"]]
    assert SKILL_ZERO_USAGE in flag_types
    assert result["is_honeypot"] is True


def test_skill_zero_usage_does_not_fire_below_threshold(candidate):
    # Only 2 such skills -- below the >=3 threshold, should NOT fire.
    candidate["skills"] += [
        {"name": "PyTorch", "proficiency": "expert", "endorsements": 5, "duration_months": 0},
        {"name": "Pinecone", "proficiency": "expert", "endorsements": 3, "duration_months": 0},
    ]
    result = detect_honeypot(candidate)
    flag_types = [f["type"] for f in result["flags"]]
    assert SKILL_ZERO_USAGE not in flag_types


def test_yoe_mismatch_fires_when_experience_overstated(candidate):
    candidate["profile"]["years_of_experience"] = 15.0  # career_history sums to 4 years
    result = detect_honeypot(candidate)
    flag_types = [f["type"] for f in result["flags"]]
    assert YOE_MISMATCH in flag_types


def test_yoe_mismatch_tolerates_small_gaps(candidate):
    # Within the 2-year tolerance -- shouldn't fire on rounding/gap noise.
    candidate["profile"]["years_of_experience"] = 5.5  # career_history sums to 4 years
    result = detect_honeypot(candidate)
    flag_types = [f["type"] for f in result["flags"]]
    assert YOE_MISMATCH not in flag_types


def test_date_math_mismatch_fires_on_self_contradicting_dates(candidate):
    # duration_months says 999, but start/end dates say ~48 months.
    candidate["career_history"][0]["duration_months"] = 999
    result = detect_honeypot(candidate)
    flag_types = [f["type"] for f in result["flags"]]
    assert DATE_MATH_MISMATCH in flag_types
    assert result["is_honeypot"] is True  # hard check, fires alone


def test_overlapping_roles_fires_on_concurrent_full_time_jobs(candidate):
    candidate["career_history"][0]["is_current"] = False
    candidate["career_history"][0]["end_date"] = "2023-06-01"
    candidate["career_history"].append({
        "company": "Swiggy", "title": "ML Engineer",
        "start_date": "2023-01-01", "end_date": None,  # overlaps with the first role
        "duration_months": 24, "is_current": True,
        "industry": "Technology", "company_size": "1001-5000",
        "description": "Worked on search ranking.",
    })
    result = detect_honeypot(candidate)
    flag_types = [f["type"] for f in result["flags"]]
    assert OVERLAPPING_ROLES in flag_types
    assert result["is_honeypot"] is True


def test_multiple_current_roles_fires(candidate):
    candidate["career_history"].append({
        "company": "Swiggy", "title": "ML Engineer",
        "start_date": "2022-01-01", "end_date": None,
        "duration_months": 30, "is_current": True,  # second concurrent "current" role
        "industry": "Technology", "company_size": "1001-5000",
        "description": "Worked on search ranking.",
    })
    result = detect_honeypot(candidate)
    flag_types = [f["type"] for f in result["flags"]]
    assert MULTIPLE_CURRENT_ROLES in flag_types
    assert result["is_honeypot"] is True


def test_education_timeline_impossible_fires(candidate):
    # Graduated 4 years ago but claims 15 years of experience. Also bump
    # the career_history duration so yoe_mismatch isn't the only thing
    # firing -- isolates education_timeline_impossible specifically.
    candidate["profile"]["years_of_experience"] = 15.0
    candidate["career_history"][0]["start_date"] = "2011-01-01"
    candidate["career_history"][0]["duration_months"] = 180
    result = detect_honeypot(candidate)
    flag_types = [f["type"] for f in result["flags"]]
    assert EDUCATION_TIMELINE_IMPOSSIBLE in flag_types


def test_education_date_order_fires(candidate):
    candidate["education"][0]["start_year"] = 2024
    candidate["education"][0]["end_year"] = 2018  # ends before it starts
    result = detect_honeypot(candidate)
    flag_types = [f["type"] for f in result["flags"]]
    assert EDUCATION_DATE_ORDER in flag_types
    assert result["is_honeypot"] is True


def test_single_soft_flag_alone_does_not_disqualify(candidate):
    # yoe_mismatch is a SOFT check -- one soft flag alone, with nothing
    # else wrong, should not be enough to call someone a honeypot.
    # Understating experience (1.0 vs the 4-year career_history) isolates
    # yoe_mismatch without also tripping education_timeline_impossible,
    # which only checks for *overstated* experience relative to graduation.
    candidate["profile"]["years_of_experience"] = 1.0
    result = detect_honeypot(candidate)
    flag_types = [f["type"] for f in result["flags"]]
    assert YOE_MISMATCH in flag_types
    assert len(flag_types) == 1  # confirm nothing else co-fired
    assert result["is_honeypot"] is False  # exactly one soft flag, no hard flags
