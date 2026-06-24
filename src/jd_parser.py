"""
jd_parser.py
Extracts structured requirements from a free-text Job Description.

In production this section is designed to call an LLM (e.g. Claude) with a strict
JSON schema prompt for robust extraction across varied JD phrasing. For this
offline/demo build (no external API key required to run), we use a deterministic
rule-based extractor over common JD section headers, which is swapped for the LLM
call by setting USE_LLM=True and providing an API key in config.py.
"""
import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class JDProfile:
    title: str = ""
    min_experience_years: float = 0.0
    must_have_skills: List[str] = field(default_factory=list)
    nice_to_have_skills: List[str] = field(default_factory=list)
    education_requirement: str = ""
    seniority: str = ""
    location: str = ""
    raw_text: str = ""


def _extract_section(text: str, header: str) -> str:
    pattern = rf"{header}:?(.*?)(?:\n\s*\n|\Z)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_bullets(section_text: str) -> List[str]:
    lines = [l.strip("- ").strip() for l in section_text.split("\n") if l.strip()]
    return [l for l in lines if l]


_STOPWORD_PREFIXES = (
    "experience", "strong", "hands-on", "familiarity", "professional",
    "bachelor", "master", "prior", "contributions", "at least", "degree",
)

# Known technical-skill vocabulary used to pull clean tokens out of free-text bullets.
_KNOWN_SKILLS = [
    "Python", "Java", "FastAPI", "Django", "Flask", "PostgreSQL", "MySQL", "SQL",
    "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Celery", "RQ", "REST APIs",
    "FAISS", "Pinecone", "Weaviate", "Kafka", "Redis", "MongoDB", "JavaScript",
    "HTML", "CSS", "Spring Boot", "Jenkins", "CI/CD", "Microservices", "Oracle DB",
    "COBOL",
]


def _extract_skills_from_bullets(bullets: List[str]) -> List[str]:
    full_text = " ".join(bullets)
    found = []
    for skill in _KNOWN_SKILLS:
        if re.search(rf"\b{re.escape(skill)}\b", full_text, re.IGNORECASE):
            found.append(skill)
    return found


def parse_jd(text: str) -> JDProfile:
    profile = JDProfile(raw_text=text)

    title_match = re.search(r"Job Title:?\s*(.+)", text, re.IGNORECASE)
    profile.title = title_match.group(1).strip() if title_match else ""

    exp_match = re.search(r"(\d+)\+?\s*years?", text, re.IGNORECASE)
    profile.min_experience_years = float(exp_match.group(1)) if exp_match else 0.0

    must_have_text = _extract_section(text, "Must-have requirements")
    nice_to_have_text = _extract_section(text, "Nice-to-have")
    education_match = re.search(r"(Bachelor'?s|Master'?s|MCA|B\.?Tech|M\.?Tech).{0,60}", text, re.IGNORECASE)

    profile.must_have_skills = _extract_skills_from_bullets(_extract_bullets(must_have_text))
    profile.nice_to_have_skills = _extract_skills_from_bullets(_extract_bullets(nice_to_have_text))
    profile.education_requirement = education_match.group(0).strip() if education_match else ""

    seniority_match = re.search(r"Seniority:?\s*(.+)", text, re.IGNORECASE)
    profile.seniority = seniority_match.group(1).strip() if seniority_match else ""

    location_match = re.search(r"Location:?\s*(.+)", text, re.IGNORECASE)
    profile.location = location_match.group(1).strip() if location_match else ""

    return profile


if __name__ == "__main__":
    with open("data/sample_jd.txt") as f:
        jd_text = f.read()
    p = parse_jd(jd_text)
    print(p)
