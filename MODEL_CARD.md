---
language:
- en
license: mit
library_name: transformers
pipeline_tag: text-classification
tags:
- educational
- portfolio
- geek-culture
- genre-classification
---

# Geek Genre Classifier

## Model description

This is an educational DistilBERT fine-tuning project that classifies short, original geek-fiction descriptions as `cyberpunk`, `high_fantasy`, `space_opera`, or `superhero`.

## Intended use

Use it as a portfolio demonstration of a reproducible text-classification workflow: dataset loading, stratified splitting, tokenization, fine-tuning, evaluation, and Hub-ready artifacts.

## Limitations

The training data is a small, original seed dataset. It is not representative of literary genres, fandom discourse, or all forms of geek culture. Metrics obtained from this dataset are only internal experiment metrics and should not be used as a quality claim for real-world classification.

## Training

- Base model: `distilbert-base-uncased`
- Dataset: `data/geek_genres.csv` from the companion GitHub repository
- Split: stratified 75% train / 25% evaluation, random seed 42
- Training script: `train_hf.py`

## Evaluation

After training, `metrics.json` records the evaluation metrics. Copy the measured value into this card before publishing a production-facing version.

## License

MIT. The seed examples are original text created for this project.
