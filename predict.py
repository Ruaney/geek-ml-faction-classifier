"""Predict a fictional sci-fi faction with the trained ML model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib


MODEL_PATH = Path(__file__).parent / "artifacts" / "faction_classifier.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a fictional character faction.")
    for feature in ("combat", "diplomacy", "technology", "cybernetics"):
        parser.add_argument(f"--{feature}", type=float, required=True, choices=range(0, 11))
    return parser.parse_args()


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run 'python train.py' first.")

    args = parse_args()
    bundle = joblib.load(MODEL_PATH)
    features = bundle["features"]
    model = bundle["model"]
    vector = [[getattr(args, feature) for feature in features]]

    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    confidence = max(probabilities)

    print(f"Predicted faction: {prediction}")
    print(f"Confidence: {confidence:.1%}")


if __name__ == "__main__":
    main()
