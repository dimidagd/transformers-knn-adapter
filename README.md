# transformers-knn-adapter

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

## Python API

```python
from transformers_knn_adapter.knn_image_pipeline import pipeline

clf = pipeline(
    "image-classification",
    model_path="microsoft/resnet-50",
    knn_model_path="/tmp/knn/model.joblib",
)
```

## Notes

Real train/eval runs can download model and dataset artifacts from Hugging Face.
