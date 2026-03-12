---
name: knn-cli-training
description: Run CLI KNN train and eval jobs in this repository using either a Hugging Face dataset id or a local imagefolder path.
---

# knn-cli-training

Use this skill when asked to run a CLI KNN training or validation/evaluation
job in this repository.

## Goal

Train and evaluate a `transformers_knn_adapter` KNN model using:

- a Hugging Face dataset id (for example `timm/mini-imagenet`), or
- a local `imagefolder` dataset path (for example `data/.../images_2`).

## Prerequisites

- Run from repository root.
- Dependencies installed with `uv`.
- Writeable output path for `--knn-model-path`.

## Commands

### Train from a Hugging Face dataset

```bash
uv run python -m transformers_knn_adapter.knn_image_pipeline train \
  --model microsoft/resnet-50 \
  --knn-model-path knn_artifacts/hf_train.joblib \
  --dataset timm/mini-imagenet \
  --split train \
  --max-samples 1000 \
  --shuffle \
  --grid-search \
  --grid-search-splits 3 \
  --grid-search-repeats 2 \
  --grid-search-scoring f1_macro
```

### Train from a local imagefolder

```bash
uv run python -m transformers_knn_adapter.knn_image_pipeline train \
  --model microsoft/resnet-50 \
  --knn-model-path knn_artifacts/local_train.joblib \
  --dataset /absolute/path/to/imagefolder_root \
  --split train \
  --shuffle
```

Notes:

- The local folder must follow `imagefolder` structure:
  `class_name/image1.jpg`, `class_name/image2.jpg`, ...
- `--dataset` can be either HF dataset id or local path.

## Evaluate

### Evaluate on a local imagefolder

```bash
uv run python -m transformers_knn_adapter.knn_image_pipeline eval \
  --model microsoft/resnet-50 \
  --knn-model-path knn_artifacts/local_train.joblib \
  --dataset /absolute/path/to/imagefolder_root \
  --split train
```

### Evaluate on a Hugging Face dataset

```bash
uv run python -m transformers_knn_adapter.knn_image_pipeline eval \
  --model microsoft/resnet-50 \
  --knn-model-path knn_artifacts/hf_train.joblib \
  --dataset timm/mini-imagenet \
  --split test \
  --stratified \
  --max-samples 100 \
  --shuffle
```

### Quick inference

```bash
uv run python -m transformers_knn_adapter.knn_image_pipeline infer \
  --model microsoft/resnet-50 \
  --knn-model-path knn_artifacts/local_train.joblib \
  --image https://picsum.photos/200
```

## Troubleshooting

- If labels or images are in non-default columns, pass `--image-column` and
  `--label-column`.
- Use `--stratified` only in non-streaming mode.
- If memory is constrained, reduce `--batch-size` and/or set `--max-samples`.
