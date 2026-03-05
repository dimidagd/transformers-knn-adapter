<h1 align="center">transformers-knn-adapter</h1>

<p align="center">
  Hugging Face image embeddings with a scikit-learn KNN head.
</p>

<p align="center">
  <a href="https://github.com/dimidagd/transformers-knn-adapter/actions/workflows/tests.yml">
    <img src="https://github.com/dimidagd/transformers-knn-adapter/actions/workflows/tests.yml/badge.svg?branch=main" alt="tests">
  </a>
  <a href="https://github.com/dimidagd/transformers-knn-adapter/actions/workflows/release-please.yml">
    <img src="https://github.com/dimidagd/transformers-knn-adapter/actions/workflows/release-please.yml/badge.svg?branch=main" alt="release-please">
  </a>
  <a href="https://github.com/dimidagd/transformers-knn-adapter/actions/workflows/publish.yml">
    <img src="https://github.com/dimidagd/transformers-knn-adapter/actions/workflows/publish.yml/badge.svg?branch=main" alt="publish">
  </a>
</p>

`transformers_knn_adapter` extends Hugging Face image models by attaching a scikit-learn KNN classifier on top of transformer embeddings.

## Requirements

- Python 3.11+
- `uv`

## Setup

```bash
uv sync --dev
```

## Run Tests

```bash
uv run pytest
```

## Train

```bash
uv run python -m transformers_knn_adapter.knn_image_pipeline train \
  --model microsoft/resnet-50 \
  --knn-model-path /tmp/knn/dinov2_small_mini_imagenet_full.joblib \
  --dataset timm/mini-imagenet \
  --split train \
  --max-samples 1000 \
  --shuffle \
  --grid-search \
  --grid-search-splits 3 \
  --grid-search-repeats 2 \
  --grid-search-scoring f1_macro
```

## Evaluate

```bash
uv run python -m transformers_knn_adapter.knn_image_pipeline eval \
  --model microsoft/resnet-50 \
  --knn-model-path /tmp/knn/dinov2_small_mini_imagenet_full.joblib \
  --dataset timm/mini-imagenet \
  --split test \
  --stratified \
  --max-samples 100 \
  --shuffle \
  --batch-size 100
```

## Inference

```bash
uv run python -m transformers_knn_adapter.knn_image_pipeline infer \
  --model microsoft/resnet-50 \
  --knn-model-path /tmp/knn/dinov2_small_mini_imagenet_full.joblib \
  --image https://picsum.photos/200 \
  --inference-batch-size 5
```

## Python API

```python
from transformers_knn_adapter.knn_image_pipeline import pipeline

clf = pipeline(
    "image-classification",
    model_path="microsoft/resnet-50",
    knn_model_path="/tmp/knn/model.joblib",
)
```

### Train from Python

```python
clf.train(
    dataset="timm/mini-imagenet",
    split="train",
    max_samples=1000,
    shuffle=True,
    grid_search=True,
    grid_search_splits=3,
    grid_search_repeats=2,
    grid_search_scoring="f1_macro",
)
```

### Evaluate from Python

```python
metrics = clf.evaluate(
    dataset="timm/mini-imagenet",
    split="test",
    max_samples=100,
    shuffle=True,
    batch_size=100,
)
print(metrics["top1_accuracy"])
```

### Inference from Python

```python
single = clf("https://picsum.photos/200")
batch = clf(["https://picsum.photos/200"] * 5)
print(single)
print(batch)
```

## Notes

Real train/eval runs can download model and dataset artifacts from Hugging Face.
