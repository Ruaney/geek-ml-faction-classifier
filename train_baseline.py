"""Train a lightweight, reproducible baseline for the Hugging Face Hub.

The model uses TF-IDF features and logistic regression. It is intentionally
small so the published artifact can be inspected and reused easily.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).parent
DATASET_PATH = ROOT / "data" / "geek_genres.csv"
OUTPUT_DIR = ROOT / "artifacts" / "hf-baseline"


def load_rows() -> tuple[list[str], list[str]]:
    with DATASET_PATH.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    return [row["text"] for row in rows], [row["label"] for row in rows]


def main() -> None:
    texts, labels = load_rows()
    train_x, test_x, train_y, test_y = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=1_000, random_state=42)),
        ]
    )
    model.fit(train_x, train_y)
    predicted = model.predict(test_x)
    metrics = {
        "accuracy": accuracy_score(test_y, predicted),
        "test_examples": len(test_y),
        "train_examples": len(train_y),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUTPUT_DIR / "model.joblib")
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:\n")
    print(classification_report(test_y, predicted, zero_division=0))


if __name__ == "__main__":
    main()
