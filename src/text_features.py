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
    return cosine_similarity(jd_vec, candidate_matrix)[0]
