import re
from pathlib import Path
from typing import Dict, List, Any

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC


DATA_PATH = Path("Resume.csv")
ARTIFACT_PATH = Path("resume_recommendation_artifacts.joblib")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class ResumeRecommender:
    def __init__(self, data_path: Path = DATA_PATH, artifact_path: Path = ARTIFACT_PATH):
        self.data_path = data_path
        self.artifact_path = artifact_path
        self.encoder_model = SentenceTransformer(MODEL_NAME)

        if artifact_path.exists():
            artifacts = joblib.load(artifact_path)
            self.tfidf = artifacts["tfidf"]
            self.label_encoder = artifacts["label_encoder"]
            self.svm_model = artifacts["svm_model"]
            self.accuracy = artifacts.get("accuracy")
        else:
            self.train_and_save()

    def train_and_save(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Missing resume dataset: {self.data_path}")

        df = pd.read_csv(self.data_path)
        required = ["Resume_str", "Category"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in resume dataset: {missing}")

        df["cleaned_resume"] = df["Resume_str"].apply(clean_text)
        df = df[df["cleaned_resume"].str.split().str.len() > 30].copy()

        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df["Category"])

        embeddings = self.encoder_model.encode(
            df["cleaned_resume"].tolist(),
            batch_size=32,
            show_progress_bar=True,
        )

        self.tfidf = TfidfVectorizer(max_features=5000)
        x_tfidf = self.tfidf.fit_transform(df["cleaned_resume"]).toarray()

        x_combined = np.hstack((embeddings, x_tfidf))

        x_train, x_test, y_train, y_test = train_test_split(
            x_combined,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        self.svm_model = LinearSVC(class_weight="balanced", random_state=42)
        self.svm_model.fit(x_train, y_train)

        y_pred = self.svm_model.predict(x_test)
        self.accuracy = float(accuracy_score(y_test, y_pred))

        joblib.dump(
            {
                "tfidf": self.tfidf,
                "label_encoder": self.label_encoder,
                "svm_model": self.svm_model,
                "accuracy": self.accuracy,
            },
            self.artifact_path,
        )

    def predict_top_n(self, cv_text: str, top_n: int = 3) -> Dict[str, Any]:
        text_clean = clean_text(cv_text)

        text_embedding = self.encoder_model.encode([text_clean])
        text_tfidf = self.tfidf.transform([text_clean]).toarray()
        text_combined = np.hstack((text_embedding, text_tfidf))

        scores = self.svm_model.decision_function(text_combined)[0]
        top_indices = np.argsort(scores)[::-1][:top_n]

        categories = self.label_encoder.inverse_transform(top_indices)

        results = []
        for rank, (category, score) in enumerate(zip(categories, scores[top_indices]), start=1):
            results.append({
                "rank": rank,
                "category": category,
                "score": round(float(score), 4),
            })

        return {
            "accuracy": self.accuracy,
            "results": results,
        }
