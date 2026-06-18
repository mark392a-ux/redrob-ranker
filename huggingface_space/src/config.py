"""
Configuration derived directly from job_description.docx and the hackathon
participant notes embedded in it. Keeping these as named, commented constants
(rather than burying them in scorer logic) is deliberate: at the Stage 5
interview you need to explain *why* a candidate scored the way they did, and
"because line 47 of config.py says so" is a much better answer than digging
through a 300-line scoring function.
"""

# ---------------------------------------------------------------------------
# Core technical requirement groups ("things you absolutely need" in the JD).
# Matching is done against a TF-IDF-vectorized concatenation of headline +
# summary + career_history descriptions + skill names, AND against the
# structured `skills` list with proficiency/duration weighting. The grouping
# matters more than any single term: a candidate doesn't need to say
# "Pinecone" by name, they need evidence from *any* term in the group.
# ---------------------------------------------------------------------------
REQUIRED_SKILL_GROUPS = {
    "embeddings_retrieval": [
        "embedding", "embeddings", "sentence-transformers", "sentence transformers",
        "openai embeddings", "bge", "e5", "dense retrieval", "semantic search",
        "retrieval augmented generation", "rag",
    ],
    "vector_search_infra": [
        "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
        "faiss", "vector database", "vector db", "hybrid search", "ann search",
        "approximate nearest neighbor",
    ],
    "ranking_eval": [
        "ndcg", "mrr", "map", "mean average precision", "a/b test", "ab test",
        "offline evaluation", "online evaluation", "ranking evaluation",
        "learning to rank", "ltr",
    ],
    "python_production": [
        "python", "production", "deployed", "shipped", "scale", "real users",
        "live system",
    ],
}

# "Things we'd like but won't reject for" — smaller positive weight, never a
# disqualifier on their own.
NICE_TO_HAVE_TERMS = [
    "lora", "qlora", "peft", "fine-tuning", "fine tuning", "xgboost",
    "learning-to-rank", "distributed systems", "inference optimization",
    "open source", "open-source", "hr-tech", "hr tech", "recruiting tech",
    "marketplace",
]

# Explicit JD disqualifier: consulting/services-only career with zero product
# company experience. Names taken verbatim from the JD's own exclusion list,
# plus the additional Indian IT-services firms that show up in this dataset.
CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "tech mahindra", "mindtree", "hcl", "hcltech", "lti", "ltimindtree",
}

# JD disqualifier: primary expertise in CV/speech/robotics without NLP/IR.
NON_NLP_DOMAINS = ["computer vision", "speech recognition", "robotics", "image processing"]
NLP_IR_EVIDENCE = ["nlp", "natural language", "retrieval", "search", "ranking", "text", "llm", "language model"]

# JD disqualifier: framework-tutorial-only, recent (<12mo) LangChain/API-wrapper
# experience with no pre-LLM-era production ML.
WRAPPER_ONLY_TERMS = ["langchain", "openai api", "gpt wrapper", "prompt engineering only"]
PRE_LLM_ML_EVIDENCE = ["recommendation", "search ranking", "click-through", "ctr prediction",
                        "fraud detection", "forecasting", "nlp pipeline", "information retrieval"]

# JD disqualifier: research-only career, no production deployment.
RESEARCH_ONLY_TERMS = ["research lab", "academic", "phd research", "published paper", "research scientist"]
PRODUCTION_EVIDENCE_TERMS = ["production", "deployed", "shipped", "scaled", "live", "real users", "on-call"]

# Title buckets, used as the single most decisive anti-keyword-stuffing
# signal: an "HR Manager" with ten AI skills listed should not outrank a
# "Senior Backend Engineer" with relevant career history.
TARGET_TITLE_TERMS = [
    "ai engineer", "ml engineer", "machine learning", "applied scientist",
    "research engineer", "data scientist", "nlp engineer", "search engineer",
    "recommendation systems", "ranking engineer", "ai researcher",
    "software engineer", "senior software engineer", "staff engineer",
    "backend engineer", "data engineer",
]
OFF_TARGET_TITLE_TERMS = [
    "hr manager", "marketing manager", "accountant", "recruiter", "sales",
    "operations manager", "business analyst", "graphic designer",
    "customer support", "civil engineer", "mechanical engineer",
    "project manager", ".net developer",
]

# JD location preference. Not a hard filter (JD explicitly welcomes these
# cities, and willing_to_relocate is its own signal) — just a soft boost.
PREFERRED_LOCATIONS = [
    "pune", "noida", "hyderabad", "mumbai", "delhi", "gurugram", "gurgaon",
    "bangalore", "bengaluru",
]

# Ideal-candidate experience band from the JD ("5-9 years... not a hard
# requirement"). Modeled as a soft band, not a cutoff: a fit score of 1.0
# inside the band, decaying gradually outside it.
EXPERIENCE_BAND = (5.0, 9.0)
EXPERIENCE_DECAY_PER_YEAR = 0.12  # score lost per year outside the band

# ---------------------------------------------------------------------------
# Top-level scoring weights. These sum to 1.0 across the "fit" components;
# the behavioral multiplier and honeypot filter are applied on top, not
# blended in, because they answer a different question ("is this person
# real and reachable?" vs "do they fit the role?").
# ---------------------------------------------------------------------------
WEIGHTS = {
    "title_match": 0.20,
    "text_relevance": 0.25,
    "skills_trust": 0.20,
    "experience_band": 0.10,
    "production_signal": 0.15,
    "location": 0.05,
    "education_tier": 0.05,
}

# Disqualifier penalties, subtracted from the weighted sum above (floored at 0).
PENALTIES = {
    "consulting_only": 0.35,
    "non_nlp_domain": 0.25,
    "wrapper_only_recent": 0.30,
    "research_only": 0.30,
}

EDUCATION_TIER_SCORE = {"tier_1": 1.0, "tier_2": 0.7, "tier_3": 0.45, "tier_4": 0.3, "unknown": 0.4}
