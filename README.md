# INDIA.RUNS — Intelligent Candidate Discovery
**Team:** ByteMasters | Redrob × Hack2skill — Data & AI Challenge

An explainable, multi-signal AI pipeline that ranks candidates against a Job
Description (JD), going beyond keyword matching by combining semantic similarity,
structured skill coverage, experience fit, education fit, and career trajectory
into one transparent score.

## Problem
Traditional candidate matching relies on keyword/Boolean search, which misses
semantically equivalent skills (e.g. "React" vs. "frontend JS framework"),
provides no justification for rankings, and offers no defense against
inconsistent or low-quality resume data.

## Solution
A pipeline that:
1. Parses the JD into structured requirements (title, min. experience, must-have /
   nice-to-have skills, education, seniority, location).
2. Scores every candidate on 5 independent signals:
   - **Skill coverage** — rule-based match against must-have skills
   - **Semantic similarity** — TF-IDF cosine similarity between JD text and
     candidate skills/resume summary (swappable for embedding-based search)
   - **Experience match** — relative to JD minimum, with diminishing returns for
     large excess and a penalty for under-qualification
   - **Education match** — degree-requirement check
   - **Trajectory** — seniority signal from years of experience + role titles
3. Combines signals into a configurable weighted final score (0–100).
4. Flags low-confidence or inconsistent profiles instead of silently
   dropping or silently scoring them.
5. Produces a grounded, per-candidate explanation referencing only data
   actually extracted from the resume (no hallucinated justifications).
6. Exports a ranked CSV/XLSX output.

## Architecture
```
JD Input → JD Parser → JD Profile (structured)
                              ↓
Candidate DB → Multi-Signal Scoring Engine
               (Skill Coverage | Semantic Similarity | Experience | Education | Trajectory)
                              ↓
               Confidence / Data-Validation Layer
                              ↓
               Explainability Generator (grounded)
                              ↓
               Ranked Output (CSV/XLSX + rationale)
```

## Project Structure
```
bytemasters-candidate-discovery/
├── data/
│   ├── sample_jd.txt              # Sample job description
│   └── sample_candidates.csv      # Sample candidate dataset
├── src/
│   ├── jd_parser.py               # JD → structured requirements
│   ├── scorer.py                  # Multi-signal scoring engine
│   └── main.py                    # End-to-end pipeline entry point
├── output/
│   └── ranked_candidates.csv/xlsx # Generated ranked output
├── requirements.txt
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
python src/main.py --jd data/sample_jd.txt --candidates data/sample_candidates.csv --out output/ranked_candidates.csv
```
Optional: limit to top-N candidates with `--top_k 5`. Export to Excel by
pointing `--out` at a `.xlsx` path.

## Methodology Notes
- **Why TF-IDF instead of an embedding API in this build:** keeps the repo
  runnable fully offline with zero external API keys for judging/reproducibility.
  The scorer is structured so `_build_semantic_corpus()` can be swapped for a
  sentence-transformer or LLM-embedding call with no changes elsewhere in the
  pipeline (see comments in `scorer.py`).
- **Explainability:** every explanation string is built only from computed
  fields (matched/missing skills, years vs. requirement, confidence flag) —
  never freeform generative text — to prevent hallucinated justifications.
- **Data validation:** `_confidence_flag()` checks for missing required fields
  and implausible values (e.g. negative or >50 years experience) and discounts
  (rather than discards) flagged profiles, surfacing them for human review.
- **Configurable weights:** `DEFAULT_WEIGHTS` in `scorer.py` can be tuned per
  role family (e.g., weight skills higher for IC roles, trajectory higher for
  leadership roles).

## Tech Stack
| Component | Technology | Why |
|---|---|---|
| Data processing | pandas | Fast, simple tabular manipulation |
| Semantic similarity | scikit-learn TF-IDF + cosine similarity | Lightweight, dependency-free semantic-ish matching; swappable for embeddings |
| Output | openpyxl (via pandas) | CSV/XLSX export |
| JD parsing | Regex + curated skill vocabulary (LLM-swappable) | Deterministic, reproducible for demo; designed for LLM upgrade |

## Future Improvements
- Swap TF-IDF for true embedding-based retrieval (sentence-transformers / API embeddings + vector DB)
- LLM-based JD parsing for handling arbitrary JD phrasing/structure
- Learned re-ranker trained on recruiter feedback instead of fixed weights
- Duplicate/near-duplicate resume detection
- Per-role configurable weight profiles surfaced via a config file
