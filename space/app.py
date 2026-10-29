"""Public Gradio demo for the uploaded Geek Genre Classifier."""

from __future__ import annotations

import os

import gradio as gr
from transformers import pipeline


MODEL_ID = os.getenv("MODEL_ID", "Ruaney/geek-genre-text-classifier")
classifier = pipeline("text-classification", model=MODEL_ID, top_k=None)


def classify(text: str) -> dict[str, float]:
    if not text or not text.strip():
        raise gr.Error("Write a short geek-fiction description first.")
    results = classifier(text)
    return {item["label"]: round(float(item["score"]), 4) for item in results}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(
        lines=5,
        label="Description",
        placeholder="A rebel pilot races through an asteroid field to protect a distant colony...",
    ),
    outputs=gr.Label(num_top_classes=4, label="Predicted genre"),
    title="Geek Genre Classifier",
    description="Educational DistilBERT demo: cyberpunk, high fantasy, space opera, or superhero.",
    examples=[
        ["A neon hacker discovers that the city government is controlled by an artificial intelligence."],
        ["A dragon guardian asks a young mage to recover a crown from an enchanted forest."],
        ["A starship captain leads a small fleet through a nebula to stop an empire."],
        ["A masked inventor uses a rescue suit to protect civilians during an attack."],
    ],
)


if __name__ == "__main__":
    demo.launch()
