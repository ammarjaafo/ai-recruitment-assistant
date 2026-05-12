import re
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Any

import numpy as np
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = Path("job_descriptions.csv")
EMBEDDINGS_PATH = Path("job_embeddings.npy")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

ALPHA_EMBEDDING = 0.50
BETA_SKILLS = 0.30
GAMMA_TITLE = 0.05
DELTA_OVERLAP = 0.15


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z+#./ ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class CVMatcher:
    def __init__(self, data_path: Path = DATA_PATH, embeddings_path: Path = EMBEDDINGS_PATH):
        if not data_path.exists():
            raise FileNotFoundError(f"Missing dataset: {data_path}")

        self.df = pd.read_csv(data_path)
        self.df.columns = self.df.columns.str.strip()
        self.df = self.df.fillna("")

        required = ["Job Title", "skills", "Responsibilities", "Job Description"]
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns in job dataset: {missing}")

        self.df["full_job_text"] = (
            self.df["Job Title"].astype(str) + " " +
            self.df["skills"].astype(str) + " " +
            self.df["Responsibilities"].astype(str) + " " +
            self.df["Job Description"].astype(str)
        )
        self.df["cleaned_job"] = self.df["full_job_text"].apply(clean_text)

        self.model = SentenceTransformer(MODEL_NAME)

        if embeddings_path.exists():
            self.job_embeddings = np.load(embeddings_path)
        else:
            self.job_embeddings = self.model.encode(
                self.df["cleaned_job"].tolist(),
                convert_to_numpy=True,
                show_progress_bar=True,
            )
            np.save(embeddings_path, self.job_embeddings)

        self.skill_aliases = {
            "ml": "machine learning",
            "machine learning algorithms": "machine learning",
            "dl": "deep learning",
            "nlp": "natural language processing",
            "ai": "artificial intelligence",
            "data analytics": "data analysis",
            "data visualization": "visualization",
            "visualisation": "visualization",
            "python programming": "python",
            "js": "javascript",
            "ms excel": "excel",
            "microsoft excel": "excel",
            "ms word": "word",
            "microsoft word": "word",
            "ms office": "microsoft office",
            "office suite": "microsoft office",
            "hr": "human resources",
            "recruitment": "recruiting",
        }

        self.skills_list = self._build_skills_list()
        self.nlp = spacy.blank("en")
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        patterns = [self.nlp.make_doc(skill) for skill in self.skills_list if skill.strip()]
        self.matcher.add("SKILLS", patterns)

        self.df["skills_extracted"] = self.df["cleaned_job"].apply(self.extract_skills)
        self.df["skills_from_column"] = self.df["skills"].apply(self.parse_skills_column)
        self.df["job_skills_final"] = [
            extracted.union(from_col)
            for extracted, from_col in zip(self.df["skills_extracted"], self.df["skills_from_column"])
        ]

        self.skill_weights = self._build_skill_weights()

    def normalize_skill(self, skill: str) -> str:
        skill = str(skill).strip().lower()
        skill = re.sub(r"\s+", " ", skill)
        return self.skill_aliases.get(skill, skill)

    def _build_skills_list(self) -> List[str]:
        all_skills = set()
        for skills_text in self.df["skills"]:
            for skill in str(skills_text).split(","):
                skill = skill.strip().lower()
                if skill:
                    all_skills.add(skill)

        base_skills = {
            "python", "machine learning", "deep learning", "natural language processing",
            "data analysis", "data analytics", "visualization", "pandas", "numpy", "sql",
            "excel", "communication", "teamwork", "leadership", "reporting",
            "customer service", "project management", "sales", "marketing", "recruiting",
            "human resources", "finance", "accounting", "data entry", "java", "javascript",
            "c++", "tableau", "power bi", "api development", "database management"
        }

        return sorted({
            self.normalize_skill(skill)
            for skill in (all_skills | base_skills)
            if str(skill).strip()
        })

    def extract_skills(self, text: str) -> Set[str]:
        doc = self.nlp.make_doc(str(text))
        matches = self.matcher(doc)

        found = set()
        for _, start, end in matches:
            found.add(self.normalize_skill(doc[start:end].text))

        return found

    def parse_skills_column(self, skills_text: str) -> Set[str]:
        text = clean_text(skills_text)
        parsed = self.extract_skills(text)

        for part in str(skills_text).split(","):
            part = self.normalize_skill(part)
            if part:
                parsed.add(part)

        return parsed

    def _build_skill_weights(self) -> Dict[str, float]:
        skill_freq = Counter()
        for skills_set in self.df["job_skills_final"]:
            for skill in skills_set:
                skill_freq[skill] += 1

        n_jobs = len(self.df)
        return {
            skill: math.log((n_jobs + 1) / (freq + 1)) + 1.0
            for skill, freq in skill_freq.items()
        }

    def weighted_skill_score(self, job_skills: Set[str], cv_skills: Set[str]) -> float:
        if not job_skills:
            return 0.0

        total_weight = sum(self.skill_weights.get(skill, 1.0) for skill in job_skills)
        matched_weight = sum(
            self.skill_weights.get(skill, 1.0)
            for skill in job_skills
            if skill in cv_skills
        )

        return matched_weight / total_weight if total_weight else 0.0

    @staticmethod
    def title_score(job_title: str, cv_text: str) -> float:
        stop_words = {"and", "or", "the", "a", "an", "of", "for", "to", "in"}
        job_title = clean_text(job_title)
        cv_text = clean_text(cv_text)

        title_tokens = set(job_title.split())
        cv_tokens = set(cv_text.split())

        important_tokens = {
            token for token in title_tokens
            if len(token) > 2 and token not in stop_words
        }

        if not important_tokens:
            return 0.0

        return len(important_tokens.intersection(cv_tokens)) / len(important_tokens)

    @staticmethod
    def overlap_score(job_skills: Set[str], cv_skills: Set[str]) -> float:
        overlap_count = len(job_skills.intersection(cv_skills))

        if overlap_count >= 3:
            return 1.0
        if overlap_count == 2:
            return 0.6
        if overlap_count == 1:
            return 0.2
        return 0.0

    def match_cv_to_jobs(self, cv_text: str, top_n: int = 10) -> Dict[str, Any]:
        cv_clean = clean_text(cv_text)
        cv_skills = self.extract_skills(cv_clean)

        cv_embedding = self.model.encode([cv_clean], convert_to_numpy=True)
        emb_scores = cosine_similarity(cv_embedding, self.job_embeddings)[0]

        candidate_count = min(5000, len(self.df))
        candidate_idx = np.argsort(emb_scores)[-candidate_count:]
        candidate_idx = candidate_idx[np.argsort(emb_scores[candidate_idx])[::-1]]

        df_top = self.df.iloc[candidate_idx].copy().reset_index(drop=True)
        emb_top_scores = emb_scores[candidate_idx]

        skill_scores = np.array([
            self.weighted_skill_score(job_skills, cv_skills)
            for job_skills in df_top["job_skills_final"]
        ])

        title_scores = np.array([
            self.title_score(title, cv_clean)
            for title in df_top["Job Title"]
        ])

        overlap_scores = np.array([
            self.overlap_score(job_skills, cv_skills)
            for job_skills in df_top["job_skills_final"]
        ])

        final_scores = (
            ALPHA_EMBEDDING * emb_top_scores +
            BETA_SKILLS * skill_scores +
            GAMMA_TITLE * title_scores +
            DELTA_OVERLAP * overlap_scores
        )

        df_top["embedding_score"] = emb_top_scores
        df_top["skill_score"] = skill_scores
        df_top["title_score"] = title_scores
        df_top["overlap_score"] = overlap_scores
        df_top["final_score"] = final_scores

        df_grouped = df_top.loc[df_top.groupby("Job Title")["final_score"].idxmax()]
        df_grouped = df_grouped.sort_values("final_score", ascending=False).reset_index(drop=True)
        top_results = df_grouped.head(top_n)

        results = []
        for i, row in top_results.iterrows():
            job_skills = row["job_skills_final"]
            common_skills = sorted(list(job_skills.intersection(cv_skills)))

            results.append({
                "rank": i + 1,
                "job_title": row.get("Job Title", ""),
                "final_score": round(float(row["final_score"]), 4),
                "embedding_score": round(float(row["embedding_score"]), 4),
                "skill_score": round(float(row["skill_score"]), 4),
                "title_score": round(float(row["title_score"]), 4),
                "overlap_score": round(float(row["overlap_score"]), 4),
                "matching_skills": common_skills,
                "job_description": str(row.get("Job Description", ""))[:700],
            })

        return {
            "extracted_cv_skills": sorted(list(cv_skills)),
            "best_job": results[0] if results else None,
            "results": results,
        }
