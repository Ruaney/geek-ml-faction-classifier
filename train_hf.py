"""Fine-tune DistilBERT on the original Geek Genre seed dataset.

This is a portfolio demo. It intentionally reports metrics honestly and does
not claim that its small synthetic dataset represents genre classification in
the real world.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


ROOT = Path(__file__).parent
DATASET_PATH = ROOT / "data" / "geek_genres.csv"
DEFAULT_OUTPUT = ROOT / "artifacts" / "hf" / "model"
BASE_MODEL = "distilbert-base-uncased"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a geek genre classifier.")
    parser.add_argument("--epochs", type=float, default=8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = Dataset.from_csv(str(DATASET_PATH))
    labels = sorted(set(dataset["label"]))
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    indexes = list(range(len(dataset)))
    train_indexes, test_indexes = train_test_split(
        indexes,
        test_size=0.25,
        random_state=42,
        stratify=dataset["label"],
    )
    train_dataset = dataset.select(train_indexes)
    test_dataset = dataset.select(test_indexes)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[int]]:
        return tokenizer(batch["text"], truncation=True, max_length=128)

    def encode_label(example: dict[str, str]) -> dict[str, int]:
        return {"labels": label_to_id[example["label"]]}

    train_dataset = train_dataset.map(tokenize, batched=True).map(encode_label)
    test_dataset = test_dataset.map(tokenize, batched=True).map(encode_label)
    train_dataset = train_dataset.remove_columns(["text", "label"])
    test_dataset = test_dataset.remove_columns(["text", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    accuracy = evaluate.load("accuracy")

    def compute_metrics(prediction: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        logits, expected = prediction
        predicted = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=predicted, references=expected)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to="none",
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
