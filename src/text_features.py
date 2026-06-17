"""
Text relevance scoring.

Default path: TF-IDF + cosine similarity over (JD text) vs (candidate's
headline + summary + career_history descriptions + skill names). This is
deliberately the *default*, not a fallback, for a few concrete reasons:

1. It has zero network dependency and zero model-download step, so the
   "no network during ranking" constraint is trivially satisfied with no
   precomputation choreography required.
2. It's CPU-fast at 100K-candidate scale: fit + transform for the whole
   pool runs in low single-digit seconds.
3. IDF weighting already does a lot of the "don't just count keywords"
   work you'd otherwise hand-wave at an embedding model: a rare, JD-specific
   term like "qdrant" or "ndcg" carries far more weight than common words
   like "experience" or "team", without needing a 400MB model file.

If you want to upgrade to dense embeddings (sentence-transformers, a local
GGUF model, etc.), this module is the only one that needs to change --
`score_text_relevance` is the single integration point. Swap the TF-IDF
vectorizer for a precomputed embedding matrix + cosine similarity and
nothing else in the pipeline needs to know. Just make sure the embedding
model is downloaded and cached *before* the timed ranking step, since
downloading it counts as network access during ranking otherwise.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def candidate_text(candidate: dict) -> str:
    profile = candidate["profile"]
    parts = [profile.get("headline", ""), profile.get("summary", "")]
    for ch in candidate.get("career_history", []):
        parts.append(ch.get("title", ""))
        parts.append(ch.get("description", ""))
    for s in candidate.get("skills", []):
        parts.append(s.get("name", ""))
    return " ".join(p for p in parts if p)


def build_relevance_scorer(jd_text: str, candidate_texts: list[str]):
    """
    Fits one TF-IDF vectorizer over the JD + all candidate texts, returns a
    function that scores a single candidate index against the JD.
    Fitting once over the whole corpus (rather than per-candidate) is what
    keeps this fast at 100K scale -- it's a single sparse matrix multiply,
    not 100K independent vectorizations.
    """
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=1,
    )
    corpus = [jd_text] + candidate_texts
    matrix = vectorizer.fit_transform(corpus)
    jd_vec = matrix[0:1]
    candidate_matrix = matrix[1:]
    sims = cosine_similarity(jd_vec, candidate_matrix)[0]  # shape (n_candidates,)
    return sims  # array aligned with candidate_texts order
