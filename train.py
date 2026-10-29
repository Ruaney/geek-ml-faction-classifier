"""Train and evaluate a small sci-fi faction classifier."""

from __future__ import annotations

import csv
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).parent
DATASET_PATH = ROOT / "data" / "characters.csv"
MODEL_PATH = ROOT / "artifacts" / "faction_classifier.joblib"
FEATURES = ["combat", "diplomacy", "technology", "cybernetics"]


def load_dataset() -> tuple[list[list[float]], list[str]]:
    """Load feature vectors and labels from the bundled CSV file."""
    with DATASET_PATH.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    inputs = [[float(row[feature]) for feature in FEATURES] for row in rows]
    labels = [row["faction"] for row in rows]
    return inputs, labels


def main() -> None:
    inputs, labels = load_dataset()
    train_x, test_x, train_y, test_y = train_test_split(
        inputs,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1_000, random_state=42)),
        ]
    )
    model.fit(train_x, train_y)

    predictions = model.predict(test_x)
    accuracy = accuracy_score(test_y, predictions)
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification report:\n")
    print(classification_report(test_y, predictions, zero_division=0))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
