from typing import Dict
from uuid import uuid4
import os
import shutil
import uuid
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


load_dotenv()

from answer_evaluator import evaluate_answer, improve_answer
from cv_matching import CVMatcher
from interview_generator import generate_interview_for_job
from resume_recommendation import ResumeRecommender
from behavioral_interview_analyzer import run_behavioral_analysis


app = FastAPI(title="AI Recruitment Assistant API")

cv_matcher = CVMatcher()
resume_recommender = ResumeRecommender()

# In-memory interview sessions
interview_sessions: Dict[str, dict] = {}


class CVMatchRequest(BaseModel):
    cv: str
    top_n: int = Field(default=10, ge=1, le=20)


class InterviewFromCVRequest(BaseModel):
    cv: str
    difficulty: str = "mid"
    num_questions: int = Field(default=7, ge=1, le=15)


class EvaluateAnswerRequest(BaseModel):
    session_id: str
    question_id: int
    answer: str


class ImproveAnswerRequest(BaseModel):
    session_id: str
    question_id: int
    answer: str


class ResumeRecommendationRequest(BaseModel):
    cv: str
    top_n: int = Field(default=3, ge=1, le=10)


@app.get("/")
def home():
    return {
        "message": "AI Recruitment Assistant API is running",
        "endpoints": [
            "/cv-match",
            "/interview/from-cv",
            "/answer/evaluate",
            "/answer/improve",
            "/resume/recommend-category",
            "/interview/behavioral-analysis",
        ],
    }


@app.post("/cv-match")
def cv_match(request: CVMatchRequest):
    return cv_matcher.match_cv_to_jobs(request.cv, request.top_n)


@app.post("/interview/from-cv")
def interview_from_cv(request: InterviewFromCVRequest):
    matching_result = cv_matcher.match_cv_to_jobs(request.cv, top_n=1)
    best_job = matching_result.get("best_job")

    if not best_job:
        raise HTTPException(status_code=404, detail="No matching job found")

    interview = generate_interview_for_job(
        job_title=best_job["job_title"],
        job_description=best_job["job_description"],
        matching_skills=best_job.get("matching_skills", []),
        difficulty=request.difficulty,
        num_questions=request.num_questions,
    )

    session_id = str(uuid4())

    questions_with_ids = []
    for idx, question_data in enumerate(interview.get("questions", []), start=1):
        questions_with_ids.append({
            "question_id": idx,
            **question_data,
        })

    interview["questions"] = questions_with_ids

    interview_sessions[session_id] = {
        "best_job": best_job,
        "difficulty": request.difficulty,
        "questions": questions_with_ids,
    }

    return {
        "session_id": session_id,
        "best_job": best_job,
        "interview": interview,
    }


def get_question_from_session(session_id: str, question_id: int):
    session = interview_sessions.get(session_id)

    if not session:
        raise HTTPException(
            status_code=404,
            detail="Invalid or expired session_id",
        )

    question_data = next(
        (q for q in session["questions"] if q["question_id"] == question_id),
        None,
    )

    if not question_data:
        raise HTTPException(
            status_code=404,
            detail="Invalid question_id for this session",
        )

    return session, question_data


@app.post("/answer/evaluate")
def evaluate(request: EvaluateAnswerRequest):
    session, question_data = get_question_from_session(
        request.session_id,
        request.question_id,
    )

    best_job = session["best_job"]

    result = evaluate_answer(
        question=question_data["question"],
        expected_skill=question_data.get("skill"),
        answer=request.answer,
        job_title=best_job["job_title"],
        job_description=best_job["job_description"],
        difficulty=session["difficulty"],
    )

    return {
        "session_id": request.session_id,
        "question_id": request.question_id,
        "question": question_data["question"],
        "expected_skill": question_data.get("skill"),
        "evaluation": result,
    }


@app.post("/answer/improve")
def improve(request: ImproveAnswerRequest):
    session, question_data = get_question_from_session(
        request.session_id,
        request.question_id,
    )

    best_job = session["best_job"]

    improved = improve_answer(
        question=question_data["question"],
        answer=request.answer,
        job_title=best_job["job_title"],
        job_description=best_job["job_description"],
    )

    return {
        "session_id": request.session_id,
        "question_id": request.question_id,
        "question": question_data["question"],
        "improved_answer": improved,
    }


@app.post("/resume/recommend-category")
def recommend_resume_category(request: ResumeRecommendationRequest):
    return resume_recommender.predict_top_n(request.cv, request.top_n)


@app.post("/interview/behavioral-analysis")
async def behavioral_interview_analysis(
    file: UploadFile = File(...),
    every_n_seconds: int = Form(2)
):
    """
    Upload an interview video and receive behavioral analysis results.

    Returns:
    - video_behavior_summary
    - temporal_behavior_summary
    - behavioral_trend
    - speech_behavior_summary

    Note:
    FFmpeg must be installed and available in PATH for speech analysis.
    """
    request_id = str(uuid.uuid4())
    temp_dir = os.path.join("uploads", request_id)

    os.makedirs(temp_dir, exist_ok=True)

    video_path = os.path.join(temp_dir, file.filename)

    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = run_behavioral_analysis(
            video_path=video_path,
            every_n_seconds=every_n_seconds,
            work_dir=temp_dir
        )

        return JSONResponse(content=result)

    except Exception as exc:
        error_traceback = traceback.format_exc()
        print(error_traceback)

        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "error": str(exc),
                "traceback": error_traceback
            }
        )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
