"""
scorer.py
Multi-signal candidate scoring engine.

Signals combined into a final weighted score:
  1. Semantic skill/text similarity (TF-IDF cosine similarity as an offline stand-in
     for embedding-based semantic search; swap with sentence-transformer / API
     embeddings in production for stronger semantic matching).
  2. Explicit must-have skill coverage (rule-based, deterministic).
  3. Experience match relative to JD minimum (with diminishing returns for large excess).
  4. Education match.
  5. Trajectory / seniority signal derived from years of experience and role titles.

Each candidate's final explanation is generated ONLY from fields that were actually
extracted/computed (grounded), so no unsupported claims are produced.
"""
from dataclasses import dataclass
from typing import List, Dict
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from jd_parser import JDProfile

# Configurable weights per role category. Defaults tuned for technical IC roles.
DEFAULT_WEIGHTS = {
    "skill_coverage": 0.35,
    "semantic_similarity": 0.25,
    "experience_match": 0.20,
    "education_match": 0.10,
    "trajectory": 0.10,
}


@dataclass
class CandidateScore:
    candidate_id: str
    name: str
    final_score: float
    skill_coverage_score: float
    semantic_score: float
    experience_score: float
    education_score: float
    trajectory_score: float
    matched_must_have: List[str]
    missing_must_have: List[str]
    confidence_flag: str
    explanation: str


def _normalize_skill(s: str) -> str:
    return re.sub(r"[^a-z0-9+]", "", s.lower())


def _skill_coverage(candidate_skills: str, must_have: List[str]) -> tuple:
    cand_set = {_normalize_skill(s) for s in candidate_skills.split(",")}
    matched, missing = [], []
    for skill in must_have:
        norm = _normalize_skill(skill)
        if not norm:
            continue
        hit = any(norm in c or c in norm for c in cand_set if c)
        (matched if hit else missing).append(skill)
    coverage = len(matched) / max(len(must_have), 1)
    return coverage, matched, missing


def _experience_score(candidate_years: float, min_years: float) -> float:
    if min_years <= 0:
        return 1.0
    if candidate_years < min_years:
        return max(0.0, candidate_years / min_years) * 0.7  # penalize under-qualified
    excess = candidate_years - min_years
    # diminishing returns after 2x the requirement; too much excess is mildly capped
    score = 1.0 + min(excess, min_years) / (2 * min_years)
    return min(score, 1.15) / 1.15  # normalize back to <=1.0


def _education_score(candidate_education: str, jd_requirement: str) -> float:
    if not jd_requirement:
        return 1.0
    jd_keywords = {"bachelor", "b.tech", "btech", "b.e", "be", "master", "m.tech", "mtech", "mca"}
    cand_lower = candidate_education.lower()
    cand_has_degree = any(k in cand_lower.replace(".", "") for k in jd_keywords)
    return 1.0 if cand_has_degree else 0.5


def _trajectory_score(years: float, current_role: str) -> float:
    seniority_terms = ["lead", "principal", "senior", "staff", "manager", "architect"]
    role_lower = current_role.lower()
    base = min(years / 10.0, 1.0)
    bump = 0.15 if any(t in role_lower for t in seniority_terms) else 0.0
    return min(base + bump, 1.0)


def _confidence_flag(row: pd.Series) -> str:
    required_fields = ["name", "email", "total_experience_years", "skills", "education"]
    missing = [f for f in required_fields if pd.isna(row.get(f)) or str(row.get(f)).strip() == ""]
    if missing:
        return f"LOW_CONFIDENCE: missing fields {missing}"
    if row["total_experience_years"] < 0 or row["total_experience_years"] > 50:
        return "FLAGGED: implausible experience value"
    return "OK"


def _build_semantic_corpus(jd: JDProfile, candidates: pd.DataFrame) -> List[float]:
    jd_doc = jd.raw_text
    candidate_docs = (candidates["skills"].fillna("") + " " + candidates["resume_summary"].fillna(""))
    corpus = [jd_doc] + candidate_docs.tolist()
    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    tfidf = vectorizer.fit_transform(corpus)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    return sims.tolist()


def score_candidates(jd: JDProfile, candidates: pd.DataFrame, weights: Dict[str, float] = None) -> List[CandidateScore]:
    weights = weights or DEFAULT_WEIGHTS
    semantic_scores = _build_semantic_corpus(jd, candidates)

    results = []
    for idx, row in candidates.reset_index(drop=True).iterrows():
        coverage, matched, missing = _skill_coverage(str(row["skills"]), jd.must_have_skills)
        exp_score = _experience_score(float(row["total_experience_years"]), jd.min_experience_years)
        edu_score = _education_score(str(row["education"]), jd.education_requirement)
        traj_score = _trajectory_score(float(row["total_experience_years"]), str(row["current_role"]))
        sem_score = semantic_scores[idx]
        confidence = _confidence_flag(row)

        final = (
            weights["skill_coverage"] * coverage
            + weights["semantic_similarity"] * sem_score
            + weights["experience_match"] * exp_score
            + weights["education_match"] * edu_score
            + weights["trajectory"] * traj_score
        ) * 100

        if confidence != "OK":
            final *= 0.85  # discount, but do not silently drop

        explanation_parts = [
            f"Matched {len(matched)}/{len(jd.must_have_skills)} must-have skills" if jd.must_have_skills else "No must-have skills specified in JD",
            f"{row['total_experience_years']} yrs experience vs {jd.min_experience_years} yr requirement",
        ]
        if missing:
            explanation_parts.append(f"Missing: {', '.join(missing)}")
        if confidence != "OK":
            explanation_parts.append(confidence)
        explanation = "; ".join(explanation_parts)

        results.append(CandidateScore(
            candidate_id=str(row["candidate_id"]),
            name=str(row["name"]),
            final_score=round(final, 2),
            skill_coverage_score=round(coverage * 100, 2),
            semantic_score=round(sem_score * 100, 2),
            experience_score=round(exp_score * 100, 2),
            education_score=round(edu_score * 100, 2),
            trajectory_score=round(traj_score * 100, 2),
            matched_must_have=matched,
            missing_must_have=missing,
            confidence_flag=confidence,
            explanation=explanation,
        ))

    results.sort(key=lambda r: r.final_score, reverse=True)
    return results
