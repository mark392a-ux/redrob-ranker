# Redrob Intelligent Candidate Ranking — Track 1 Submission

**Repo:**  
[GitHub Repository](https://github.com/mark392a-ux/redrob-ranker)

**Live Sandbox:**  
[Hugging Face Space](https://huggingface.co/spaces/Pantheon00/redrob-ranker)

**Team:**  
Bharat_Vani (solo)

**Submitted by:**  
Ankit Mishra

**Live sandbox:**
https://pantheon00-redrob-ranker.hf.space/

**Team:**
Bharat_Vani (solo)

Submitted by Team Bharat Vani (Ankit Mishra).

## TL;DR

Given 100,000 resumes and one job description, this picks the best 100
matches — not by counting keywords, but by checking whether someone's
actual career history matches the role, catching fake/impossible profiles
along the way, and explaining *why* each person made the list in plain
English. Runs in about a minute, no internet connection needed, no GPU.

## What this is

A hybrid ranker for the Redrob "Intelligent Candidate Discovery & Ranking
Challenge" (Track 1, Data & AI Challenge). It scores all 100K candidates
against the Senior AI Engineer JD and outputs the top 100, ranked, with a
1-2 sentence reasoning per candidate.

## Results at a glance

| Metric | Value |
|---|---|
| Pool size | 100,000 candidates |
| Runtime (CPU, no GPU/network) | ~55-70s |
| Peak memory | ~3.5 GB |
| Honeypots flagged | 61 (target: ~80) |
| Top-100 composition | Cleanly dominated by on-target AI/ML titles (NLP Engineer, ML Engineer, Search Engineer, Recommendation Systems Engineer, etc.) |
| Validator | Passes `validate_submission.py` cleanly |

## Quick start

```bash
pip install -r requirements.txt
python rank.py --candidates /path/to/candidates.jsonl --out Bharat_Vani.csv
python validate_submission.py Bharat_Vani.csv   # validator from the hackathon bundle
```

Also accepts the gzipped form directly: `--candidates candidates.jsonl.gz`.

A small 50-candidate fixture is included at `test_data/sample_candidates.jsonl`
for a fast smoke test:

```bash
python rank.py --candidates test_data/sample_candidates.jsonl --out test_out.csv --top 20
```

> **Note on `candidates.jsonl`:** the 100K-candidate dataset itself is not
> included in this repo (it's organizer-provided, ~465MB uncompressed —
> well past what should live in a git repo, and not original work to
> redistribute). Supply your own copy from the hackathon bundle to run
> `rank.py` against the full pool. `.gitignore` excludes it explicitly so
> it's never accidentally committed.

## Honeypot flag types — what each check catches

| Flag | What it checks | Why it's a honeypot signal |
|---|---|---|
| `skill_zero_usage` | 3+ skills marked expert/advanced with `duration_months == 0` | Nobody is an "expert" at something used for zero months. Threshold (≥3) was tuned empirically: the real pool's distribution is cleanly bimodal — candidates have either 0 such skills or exactly 3-5, nothing in between |
| `yoe_mismatch` | Stated `years_of_experience` differs from the sum of `career_history` durations by 2+ years | A profile claiming 12 years whose listed jobs add up to 4 doesn't add up |
| `date_math_mismatch` | A career entry's start/end dates don't support its own stated `duration_months` (off by 3+ months) | The record contradicts itself using only its own fields |
| `education_timeline_impossible` | Claims more years of experience than years since graduation (+2yr buffer) | Can't have 15 years of work experience having graduated 3 years ago |
| `overlapping_roles` | Two career entries have overlapping date ranges | Can't hold two full-time roles simultaneously |
| `multiple_current_roles` | More than one `is_current: true` entry | Same issue as above, different shape |
| `education_date_order` | An education entry's `end_year` is before its `start_year` | Self-contradictory on its own |

Any one of the *hard* checks (date math, overlapping roles, multiple
current roles, education date order, skill zero usage) is enough to flag
a profile alone. The two *soft* checks (yoe mismatch, education timeline)
need to co-occur with something else, since either alone could in
principle describe an unusual but real career. See `src/honeypot.py` for
the exact logic and `diagnose_skills.py` / `diagnose_extra.py` for the
empirical investigation that shaped these thresholds.

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

## Testing

```bash
pytest tests/ -v
```

19 tests covering `honeypot.py` (every check type, both the positive
"does it fire on an impossible profile" case and the negative "does it
stay silent on a clean profile" case — a honeypot filter that's too
trigger-happy disqualifies real candidates, which matters as much as
catching fakes) and `scorer.py` (weight-sum sanity, the core anti-keyword-
stuffing guarantee, disqualifier penalties, and score bounds).

## Live sandbox

https://pantheon00-redrob-ranker.hf.space/ — runs the exact
same `src/` modules as this repo (synced copies, not a reimplementation).
Upload a `.json`/`.jsonl` candidate file (capped at 500 for a fast
interactive demo per the hackathon's "small sample" sandbox requirement)
or use the bundled 50-candidate sample. Shows honeypot screening
results, a flag-type breakdown, the ranked table with full reasoning
text, and a CSV download of the demo run's output.

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
  consistency) and confirmed against the real 100K pool: 61 candidates
  flagged (40 yoe_mismatch, 35 date_math_mismatch, 21 skill_zero_usage, 18
  education_timeline_impossible — these overlap, since one candidate can
  trigger multiple checks). Four additional hypotheses were tested via
  `diagnose_extra.py` and deliberately *not* turned into filters, because
  their base rates on the real pool were far too high to be the ~80-in-100K
  honeypot signal: salary min>max (18,865 candidates, 18.9%), skill
  duration exceeding total career history (13,581 candidates across
  buckets), skill duration exceeding stated years_of_experience (13,449
  candidates), and last_active_date before signup_date (7,496 candidates,
  7.5%). These are evidently just noise in how the synthetic profiles were
  generated, not a deliberate trap — filtering on them would have
  disqualified a large fraction of legitimate candidates instead of
  catching honeypots. `multiple_current_roles` and `overlapping_roles`
  stayed at exactly 0 across the full pool, suggesting those particular
  honeypot sub-types (if they exist at all) aren't represented in this
  dataset's construction.
- The remaining gap (61 of ~80 caught) is accepted as a practical
  stopping point rather than chased further: the actual disqualification
  risk is the honeypot rate *within the top 100*, not total detection
  rate across the pool, and the top 100 here is already cleanly dominated
  by on-target titles with coherent career narratives — recommend a
  manual spot-check of the final top 100 against known honeypot patterns
  as a last safety net before submitting.
