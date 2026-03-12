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

## CLI Arguments

### `train`

| Argument | Required | Default | Description |
|---|---:|---|---|
| `--model` | Yes | - | HF model id/path for feature extraction |
| `--knn-model-path` | Yes | - | Path to save/load KNN model (`.joblib`) |
| `--dataset` | Yes | - | HF dataset name or local `imagefolder` path |
| `--split` | No | `train` | Dataset split |
| `--image-column` | No | `image` | Dataset image column |
| `--label-column` | No | `label` | Dataset label column |
| `--batch-size` | No | `16` | Embedding batch size |
| `--stream` | No | `false` | Enable streaming mode |
| `--stratified` | No | `false` | Stratified sampling (`--max-samples` subset size, non-streaming only) |
| `--shuffle` | No | `false` | Shuffle dataset before sampling/training |
| `--shuffle-seed` | No | `42` | Shuffle seed |
| `--shuffle-buffer-size` | No | `1000` | Streaming shuffle buffer size |
| `--max-samples` | No | `None` | Optional training sample cap |
| `--n-neighbors` | No | `None` | KNN neighbors (mutually exclusive with `--grid-search`) |
| `--grid-search` | No | `false` | Run `GridSearchCV` over neighbors/metrics |
| `--grid-search-splits` | No | `None` | Stratified splits per repeat for grid search |
| `--grid-search-repeats` | No | `None` | Repeat count for stratified folds in grid search |
| `--grid-search-scoring` | No | `None` | Grid-search scoring metric (`f1_macro`, `precision_macro`, `recall_macro`) |
| `--top-k` | No | `2` | Top-k at inference time |
| `--device` | No | `-1` | Transformers device index (`-1` for CPU) |

### `eval`

| Argument | Required | Default | Description |
|---|---:|---|---|
| `--model` | Yes | - | HF model id/path for feature extraction |
| `--knn-model-path` | Yes | - | Path to trained KNN model (`.joblib`) |
| `--dataset` | Yes | - | HF dataset name or local `imagefolder` path |
| `--split` | No | `validation` | Dataset split |
| `--image-column` | No | `image` | Dataset image column |
| `--label-column` | No | `label` | Dataset label column |
| `--batch-size` | No | `16` | Embedding batch size |
| `--stream` | No | `false` | Enable streaming mode |
| `--stratified` | No | `false` | Stratified sampling (`--max-samples` subset size, non-streaming only) |
| `--shuffle` | No | `false` | Shuffle dataset before evaluation |
| `--shuffle-seed` | No | `42` | Shuffle seed |
| `--shuffle-buffer-size` | No | `1000` | Streaming shuffle buffer size |
| `--max-samples` | No | `None` | Optional evaluation sample cap |
| `--min-class-instances` | No | `None` | Drop classes with fewer than this number of eval instances |
| `--negative-classes` | No | `other` | Comma-separated classes treated as negative |
| `--positive-classes-population-ratio` | No | `None` | Target positive/(total) ratio after subsampling |
| `--top-k` | No | `1` | Top-k predictions (evaluation uses top-1) |
| `--device` | No | `-1` | Transformers device index (`-1` for CPU) |

### `infer`

| Argument | Required | Default | Description |
|---|---:|---|---|
| `--model` | Yes | - | HF model id/path for feature extraction |
| `--knn-model-path` | Yes | - | Path to save/load KNN model (`.joblib`) |
| `--top-k` | No | `3` | Top-k predictions |
| `--device` | No | `-1` | Transformers device index (`-1` for CPU) |
| `--image` | No | `https://picsum.photos/200` | Image input (file path or URL) |
| `--inference-batch-size` | No | `5` | Number of images for batched inference |

### `predict`

| Argument | Required | Default | Description |
|---|---:|---|---|
| `--model` | Yes | - | HF model id/path for feature extraction |
| `--knn-model-path` | Yes | - | Path to trained KNN model (`.joblib`) |
| `--image` | Yes | - | Image path/URL accepted by Transformers image pipeline |
| `--top-k` | No | `3` | Top-k predictions |
| `--device` | No | `-1` | Transformers device index (`-1` for CPU) |

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
