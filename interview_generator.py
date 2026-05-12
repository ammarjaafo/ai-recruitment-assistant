import json
import re
import time
from typing import Dict, List, Optional

from openai import OpenAI


MODEL_NAME = "gpt-4.1-mini"
client = OpenAI(timeout=30.0, max_retries=2)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


JOB_FAMILIES = {
    "data_ai": ["data scientist", "data analyst", "machine learning", "ai", "analytics", "data engineer"],
    "software": ["software engineer", "developer", "backend", "frontend", "qa", "devops"],
    "finance": ["accountant", "financial", "banking", "auditor", "investment"],
    "marketing": ["marketing", "digital marketing", "seo", "content", "social media", "brand", "campaign"],
    "hr": ["human resources", "hr", "recruiter", "talent acquisition"],
    "sales": ["sales", "business development", "account manager", "sales representative"],
    "customer_service": ["customer service", "customer support", "call center", "customer success"],
    "operations": ["operations", "logistics", "supply chain", "procurement", "administrative assistant", "data entry"],
    "education": ["teacher", "instructor", "trainer", "education"],
    "healthcare": ["nurse", "medical", "healthcare", "clinical"],
    "design": ["designer", "graphic designer", "ui ux", "creative director"],
}


def parse_json_text(text: str):
    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return json.loads(text)


def call_openai_with_retry(prompt: str, max_retries: int = 3, base_delay: int = 2) -> str:
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=MODEL_NAME,
                input=prompt,
            )
            return response.output_text.strip()

        except Exception as exc:
            last_error = exc
            if attempt == max_retries - 1:
                raise last_error

            time.sleep(base_delay * (2 ** attempt))

    raise last_error


def detect_job_family(job_title: str, job_description: str = "") -> str:
    text = clean_text(job_title + " " + job_description)
    scores = {
        family: sum(1 for keyword in keywords if keyword in text)
        for family, keywords in JOB_FAMILIES.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def get_context_skills(job_family: str) -> List[str]:
    context_skills = {
        "data_ai": ["python", "sql", "machine learning", "data analysis", "model evaluation"],
        "software": ["programming", "apis", "databases", "testing", "debugging"],
        "finance": ["financial analysis", "excel", "reporting", "budgeting", "forecasting"],
        "marketing": ["content creation", "social media", "campaign analysis", "paid advertising", "community engagement"],
        "hr": ["recruiting", "interviewing", "employee relations", "onboarding"],
        "sales": ["lead generation", "negotiation", "crm", "account management"],
        "customer_service": ["customer support", "issue resolution", "service quality", "crm"],
        "operations": ["scheduling", "reporting", "coordination", "process improvement"],
        "education": ["lesson planning", "classroom management", "student assessment"],
        "healthcare": ["patient care", "medical documentation", "clinical procedures"],
        "design": ["visual design", "ui design", "ux research", "branding"],
        "general": ["communication", "reporting", "problem solving"],
    }
    return context_skills.get(job_family, context_skills["general"])


def filter_relevant_skills(skills: List[str], job_title: str, job_description: str, max_skills: int = 6) -> List[str]:
    text = clean_text(job_title + " " + job_description)
    filtered = []

    for skill in skills:
        skill_clean = clean_text(skill)

        if not skill_clean:
            continue
        if len(skill_clean.split()) > 5:
            continue
        if skill_clean in text or len(skill_clean.split()) <= 3:
            filtered.append(skill_clean)

    return list(dict.fromkeys(filtered))[:max_skills]


def generate_questions_openai(
    job_title: str,
    job_description: str,
    skills: List[str],
    difficulty: str = "mid",
    num_questions: int = 7,
) -> List[Dict]:
    skills_str = ", ".join(skills)

    prompt = f"""
You are an expert interviewer.

Generate {num_questions} realistic interview questions for this role.

Job title: {job_title}
Difficulty: {difficulty}
Relevant skills: {skills_str}

Job description:
{job_description}

Requirements:
- Mix technical, behavioral, situational, and role-specific questions
- Keep them realistic and professional
- Avoid repetition
- If a question is technical or situational, assign the most relevant skill
- If a question is role-specific and not tied to one exact skill, set skill to null
- Keep the wording natural and interview-like

Return ONLY valid JSON in this exact format:

[
  {{
    "type": "technical",
    "skill": "skill name or null",
    "question": "question text"
  }}
]
""".strip()

    text = call_openai_with_retry(prompt)
    return parse_json_text(text)


def generate_interview_for_job(
    job_title: str,
    job_description: str,
    matching_skills: Optional[List[str]] = None,
    difficulty: str = "mid",
    num_questions: int = 7,
) -> Dict:
    job_family = detect_job_family(job_title, job_description)

    skills = filter_relevant_skills(
        matching_skills or [],
        job_title=job_title,
        job_description=job_description,
        max_skills=6,
    )

    if not skills:
        skills = get_context_skills(job_family)

    questions = generate_questions_openai(
        job_title=job_title,
        job_description=job_description,
        skills=skills,
        difficulty=difficulty,
        num_questions=num_questions,
    )

    return {
        "job_title": job_title,
        "job_family": job_family,
        "difficulty": difficulty,
        "job_description": job_description,
        "skills_used": skills,
        "questions": questions,
    }
