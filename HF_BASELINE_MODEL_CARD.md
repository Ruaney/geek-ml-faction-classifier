---
language:
- en
license: mit
library_name: scikit-learn
pipeline_tag: text-classification
tags:
- educational
- portfolio
- geek-culture
- genre-classification
- tfidf
---

# Geek Genre Text Classifier (Baseline)

## Model description

This lightweight baseline classifies short, original geek-fiction descriptions as `cyberpunk`, `high_fantasy`, `space_opera`, or `superhero`.

It uses a TF-IDF text vectorizer followed by logistic regression, saved as a `joblib` artifact.

## Training data

The model was trained on a small original seed dataset from the companion GitHub project. The texts were written specifically for this educational portfolio project.

## Metrics and limitations

The first reproducible run obtained **31.25% accuracy on 16 evaluation examples** after training on 48 examples. Metrics are stored in `metrics.json` and come from a small stratified holdout split. They measure performance only on this limited synthetic seed dataset, not on real fiction, fandom discourse, or production traffic.

## Usage

```python
import joblib

model = joblib.load("model.joblib")
model.predict(["A rebel pilot races through an asteroid field to protect a colony."])
```

## License

MIT
