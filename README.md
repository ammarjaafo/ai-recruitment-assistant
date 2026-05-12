# AI Recruitment Assistant System

## Overview
End-to-end AI recruitment assistant with:
- CV-to-jobs matching
- Job-based interview question generation
- AI answer evaluation
- AI answer improvement
- Resume category recommendation

## Main Cores

### 1. CV Matching
Input: CV text  
Output: top matching jobs with scores and matching skills.

### 2. AI Interviewer
Uses the best matched job from CV Matching and generates interview questions.

### 3. AI Analyzer
Evaluates candidate answers and returns a structured score, strengths, weaknesses, and actionable feedback.

### 4. AI Answer Improvement
Generates a stronger interview-ready response based on the candidate's original answer.

### 5. Resume Recommendation
Predicts the most relevant job categories for a CV using SBERT embeddings + TF-IDF + LinearSVC.

## Required Data Files
Place these files in the project folder:

```bash
job_descriptions.csv
Resume.csv
```

Optional:
```bash
job_embeddings.npy
```

If `job_embeddings.npy` is missing, the system will generate it on first run.

## Run
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn app:app --reload
```

Open:
```bash
http://127.0.0.1:8000/docs
```

## Environment
Create a `.env` file:

```bash
OPENAI_API_KEY=your_api_key_here
```

## Author
Ammar Jaafo
