"""
Shared fixtures for the test suite.

`base_candidate()` returns a minimal-but-schema-valid, internally
*consistent* candidate dict -- it should NOT trigger any honeypot check
and should score reasonably as an on-target candidate. Individual tests
deep-copy this and mutate exactly the field(s) they're testing, so each
test failure points at one specific check rather than requiring you to
reverse-engineer a giant fixture.
"""

import copy
from datetime import date, timedelta

import pytest


def _iso(d: date) -> str:
    return d.isoformat()


def base_candidate() -> dict:
    today = date.today()
    job_start = today - timedelta(days=4 * 365)  # ~4 years ago

    return {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "Candidate A",
            "headline": "ML Engineer | Recommendation Systems | RAG",
            "summary": (
                "Machine learning engineer with production experience building "
                "recommendation and retrieval systems using embeddings and "
                "vector search, deployed to real users at scale."
            ),
            "location": "Bengaluru, Karnataka",
            "country": "India",
            "years_of_experience": 4.0,
            "current_title": "ML Engineer",
            "current_company": "Razorpay",
            "current_company_size": "501-1000",
            "current_industry": "Technology",
        },
        "career_history": [
            {
                "company": "Razorpay",
                "title": "ML Engineer",
                "start_date": _iso(job_start),
                "end_date": None,
                "duration_months": 48,
                "is_current": True,
                "industry": "Technology",
                "company_size": "501-1000",
                "description": (
                    "Built and deployed a recommendation system serving real "
                    "users, using embeddings and FAISS for retrieval. Improved "
                    "NDCG by 12% through A/B testing."
                ),
            }
        ],
        "education": [
            {
                "institution": "NIT Trichy",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": today.year - 4 - 4,
                "end_year": today.year - 4,
                "grade": "8.2 CGPA",
                "tier": "tier_2",
            }
        ],
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 30, "duration_months": 48},
            {"name": "Embeddings", "proficiency": "advanced", "endorsements": 15, "duration_months": 24},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 10, "duration_months": 18},
            {"name": "NDCG", "proficiency": "intermediate", "endorsements": 5, "duration_months": 12},
        ],
        "redrob_signals": {
            "profile_completeness_score": 90.0,
            "signup_date": _iso(today - timedelta(days=500)),
            "last_active_date": _iso(today - timedelta(days=5)),
            "open_to_work_flag": True,
            "profile_views_received_30d": 50,
            "applications_submitted_30d": 3,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 4.0,
            "skill_assessment_scores": {"Python": 90.0},
            "connection_count": 500,
            "endorsements_received": 60,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 20.0, "max": 28.0},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 70.0,
            "search_appearance_30d": 100,
            "saved_by_recruiters_30d": 5,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.5,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }


@pytest.fixture
def candidate():
    """Fresh deep copy for every test -- mutate freely, no cross-test bleed."""
    return copy.deepcopy(base_candidate())
