# Redrob Intelligent Candidate Ranking — Track 1 Submission

## What this is

A hybrid ranker for the Redrob "Intelligent Candidate Discovery & Ranking
Challenge" (Track 1, Data & AI Challenge). It scores all 100K candidates
against the Senior AI Engineer JD and outputs the top 100, ranked, with a
1-2 sentence reasoning per candidate.

## Quick start

```bash
pip install -r requirements.txt
python rank.py --candidates /path/to/candidates.jsonl --out submission.csv
python validate_submission.py submission.csv   # validator from the hackathon bundle
```

Also accepts the gzipped form directly: `--candidates candidates.jsonl.gz`.

A small 50-candidate fixture is included at `test_data/sample_candidates.jsonl`
for a fast smoke test:

```bash
python rank.py --candidates test_data/sample_candidates.jsonl --out test_out.csv --top 20
```

## Architecture

```
candidates.jsonl
       │
       ▼
 honeypot.py ──── flags & drops structurally-impossible profiles
       │           (date-math errors, overlapping roles, expert-skill-with-
       │            zero-usage, yoe/career-history mismatch, etc.)
       ▼
 text_features.py ── TF-IDF + cosine similarity, JD vs candidate text,
       │              fit once over the whole surviving pool
       ▼
 rule_features.py ── title match, skills trust (proficiency × duration ×
       │              endorsements), production-vs-research signal,
       │              experience-band fit, location, education tier,
       │              JD-specific disqualifier penalties
       ▼
 behavioral.py ───── multiplier (0.4-1.0) from redrob_signals: recency,
       │              recruiter response rate, interview completion,
       │              notice period, open-to-work
       ▼
 scorer.py ───────── fit_score = weighted_sum(rule features) - penalties
       │              final_score = fit_score × behavioral_multiplier
       ▼
 reasoning.py ─────── template-built explanation from the same components
       │              the scorer used (no LLM call — see below)
       ▼
 rank.py ──────────── sorts, takes top 100, ties broken by candidate_id
                       ascending, writes submission.csv
```

## Why these design choices (for the methodology summary / interview)

**TF-IDF instead of a dense embedding model.** Zero network dependency,
zero model-download step, and it satisfies the "no network during
ranking" constraint with no precomputation choreography required. IDF
weighting means a rare JD-specific term (e.g. "qdrant", "ndcg") counts for
much more than a common word, which does a lot of the "don't just count
keywords" work without a 400MB model file. `text_features.py` is the only
module that would need to change to swap in dense embeddings later — fit a
local sentence-transformers model offline, cache it, and replace the
`build_relevance_scorer` internals with a precomputed embedding matrix +
cosine similarity. (If you do this, make sure the model download happens
*before* the timed run — downloading at ranking time is "network during
ranking" and gets you disqualified at Stage 3.)

**Why title match is weighted independently from skills/text relevance.**
This is the single decisive signal against the dataset's keyword-stuffer
traps: a "Marketing Manager" or "HR Manager" with ten AI-sounding skills
listed will still score low here, because their actual role history
doesn't match, while a "Senior Backend Engineer" with relevant career
descriptions but a sparse skills list scores well.

**Why skills are weighted by duration_months and endorsements, not just
presence.** A skill listed as "expert" with 0 months of use and 0
endorsements should count for almost nothing — this is a direct,
explainable defense against lazy keyword stuffing in the skills array,
independent of the honeypot filter.

**Why honeypot detection is a hard pre-filter, not a scoring penalty.**
The spec disqualifies submissions with >10% honeypot rate in the top 100
at Stage 3 — that's a binary cliff, not a gradient, so a candidate flagged
as structurally impossible is removed before scoring rather than merely
down-weighted.

**Why the reasoning column is template-generated, not LLM-generated.**
The reasoning text is part of the CSV produced by the timed, no-network
ranking step — calling a hosted LLM per candidate there would both blow
the 5-minute budget at 100K scale and violate the network constraint.
Building it from the same extracted features the scorer used also means
it structurally can't hallucinate a skill the candidate doesn't have,
which is exactly what Stage 4's "no hallucination" check looks for.

## Performance at 100K scale

Measured on a synthetic 100K-row stress test (see commit history): ~70
seconds wall-clock, ~3.5GB peak memory. Comfortably inside the 5-minute /
16GB budget, with headroom to spare if you add a heavier feature later.

## Known limitations / next steps

- TF-IDF catches lexical and near-lexical overlap; it will miss a
  candidate who describes "built a system that learns user preferences
  from clicks" without ever saying "recommendation" or "ranking". A dense
  embedding model would close this gap — see the swap-in note above.
- The disqualifier keyword lists in `config.py` are a starting point
  extracted from the JD's explicit language; they should be tuned against
  real examples once you have visibility into the actual 100K pool's
  vocabulary (company names, title variants, skill naming conventions).
- Honeypot heuristics are validated structurally (date math, internal
  consistency) but haven't been validated against real honeypot examples,
  since the 50-candidate sample didn't contain any (expected: ~80/100K is
  too sparse to show up in a random 50-sample). Worth spot-checking
  against the real pool once available.
