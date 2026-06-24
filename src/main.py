"""
main.py
End-to-end pipeline: JD input -> parsing -> retrieval/scoring -> ranked, explainable output.

Usage:
    python src/main.py --jd data/sample_jd.txt --candidates data/sample_candidates.csv --out output/ranked_candidates.csv

Outputs a ranked CSV (and optionally XLSX) with one row per candidate, sorted by
final_score descending, including the grounded explanation and confidence flag
required for explainability and data-validation requirements.
"""
import argparse
import pandas as pd

from jd_parser import parse_jd
from scorer import score_candidates


def run_pipeline(jd_path: str, candidates_path: str, out_path: str, top_k: int = None):
    with open(jd_path) as f:
        jd_text = f.read()

    jd_profile = parse_jd(jd_text)
    candidates_df = pd.read_csv(candidates_path)

    ranked = score_candidates(jd_profile, candidates_df)
    if top_k:
        ranked = ranked[:top_k]

    rows = []
    for rank, r in enumerate(ranked, start=1):
        rows.append({
            "rank": rank,
            "candidate_id": r.candidate_id,
            "name": r.name,
            "final_score": r.final_score,
            "skill_coverage_score": r.skill_coverage_score,
            "semantic_similarity_score": r.semantic_score,
            "experience_score": r.experience_score,
            "education_score": r.education_score,
            "trajectory_score": r.trajectory_score,
            "matched_must_have_skills": ", ".join(r.matched_must_have),
            "missing_must_have_skills": ", ".join(r.missing_must_have),
            "confidence_flag": r.confidence_flag,
            "explanation": r.explanation,
        })

    out_df = pd.DataFrame(rows)

    if out_path.endswith(".xlsx"):
        out_df.to_excel(out_path, index=False)
    else:
        out_df.to_csv(out_path, index=False)

    print(f"JD parsed: title='{jd_profile.title}', min_exp={jd_profile.min_experience_years}, "
          f"must_have_skills={jd_profile.must_have_skills}")
    print(f"Ranked {len(out_df)} candidates -> {out_path}")
    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intelligent Candidate Discovery pipeline")
    parser.add_argument("--jd", default="data/sample_jd.txt")
    parser.add_argument("--candidates", default="data/sample_candidates.csv")
    parser.add_argument("--out", default="output/ranked_candidates.csv")
    parser.add_argument("--top_k", type=int, default=None)
    args = parser.parse_args()

    run_pipeline(args.jd, args.candidates, args.out, args.top_k)
