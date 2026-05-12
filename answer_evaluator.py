from typing import Dict, Optional

from interview_generator import call_openai_with_retry, parse_json_text


def evaluate_answer(
    question: str,
    expected_skill: Optional[str],
    answer: str,
    job_title: str,
    job_description: str,
    difficulty: str = "mid",
) -> Dict:
    expected = expected_skill if expected_skill else "general role understanding"

    prompt = f"""
You are an expert interview evaluator.

Evaluate the candidate's answer for the following role.

Job title: {job_title}
Difficulty: {difficulty}
Expected skill: {expected}

Job description:
{job_description}

Interview question:
{question}

Candidate answer:
{answer}

Evaluate the answer based on:
1. Relevance to the question
2. Demonstration of the expected skill
3. Clarity and reasoning
4. Practical depth
5. Use of examples or evidence

Rules:
- Do not reward the candidate merely for mentioning the skill word.
- If the candidate says they do not use the tool/skill, treat that as weak evidence.
- Keep strengths and weaknesses concise and specific.
- Feedback should be actionable.

Return ONLY valid JSON in this exact format:

{{
  "score": 0,
  "level": "Weak",
  "strengths": ["..."],
  "weaknesses": ["..."],
  "feedback": ["..."]
}}
""".strip()

    text = call_openai_with_retry(prompt)
    result = parse_json_text(text)

    score = result.get("score", 0)
    if score <= 10:
        score *= 10

    result["score"] = score
    return result


def improve_answer(
    question: str,
    answer: str,
    job_title: str,
    job_description: str,
) -> str:
    prompt = f"""
You are an expert interview coach.

Improve the following answer to make it:
- clear
- professional
- concise
- with a concrete example
- strong for interviews

Question:
{question}

Candidate Answer:
{answer}

Job Title:
{job_title}

Job Description:
{job_description}

Return only the improved answer.
""".strip()

    return call_openai_with_retry(prompt).strip()
