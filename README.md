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
from transformers_knn_adapter import pipeline

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

## Trainer Integration

The package also provides:

- `Dinov2ForImageClassificationWithArcFaceLoss`
- `KNNCallback`
- `FreezeScheduleCallback`

These are useful when you already have a working Hugging Face `Trainer` script for `Dinov2ForImageClassification` and want to:

- replace the classifier loss with ArcFace
- log KNN retrieval metrics during evaluation
- freeze and unfreeze modules by epoch

### ArcFace Dinov2 Model

`Dinov2ForImageClassificationWithArcFaceLoss` is intended as a Trainer-compatible replacement for `Dinov2ForImageClassification`.

Example:

```python
from transformers import AutoConfig, AutoImageProcessor, Trainer, TrainingArguments
from transformers_knn_adapter import Dinov2ForImageClassificationWithArcFaceLoss

config = AutoConfig.from_pretrained("facebook/dinov2-small")
config.num_labels = num_labels
config.label2id = label2id
config.id2label = id2label
config.arcface_margin = 28.6
config.arcface_scale = 64.0

image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
model = Dinov2ForImageClassificationWithArcFaceLoss.from_pretrained(
    "facebook/dinov2-small",
    config=config,
    ignore_mismatched_sizes=True,
)

training_args = TrainingArguments(
    output_dir="/tmp/arcface-run",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    dataloader_num_workers=4,
    learning_rate=1e-3,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    processing_class=image_processor,
    compute_metrics=compute_metrics,
)
trainer.train()
```

Notes:

- training loss uses ArcFace logits
- `outputs.logits` returned to Trainer evaluation/prediction are inference logits, not margin-modified training logits
- the ArcFace head weight matrix is initialized automatically when loading from a plain Dinov2 checkpoint

### KNN Callback

`KNNCallback` evaluates embeddings with a scikit-learn KNN classifier during `Trainer.evaluate()`.

Current embedding sources:

- `cls`
- `cls_mean`

Example:

```python
from transformers_knn_adapter import KNNCallback

trainer.add_callback(
    KNNCallback(
        trainer=trainer,
        label_column="labels",
        ks=(1, 5),
        embedding_source="cls_mean",
    )
)
```

Requirements:

- train and eval datasets must yield `pixel_values`
- train and eval datasets must expose the label column you pass to `label_column`
- for ViT-like models, hidden states must be available so the callback can extract token embeddings

Current callback behavior:

- uses `trainer.args.per_device_eval_batch_size` by default
- uses `trainer.args.dataloader_num_workers` by default
- fits KNN once at `max(ks)` and derives smaller-`k` metrics from the same neighbor query

### Combined Trainer Example

```python
from transformers import AutoConfig, AutoImageProcessor, Trainer, TrainingArguments
from transformers_knn_adapter import (
    Dinov2ForImageClassificationWithArcFaceLoss,
    FreezeScheduleCallback,
    KNNCallback,
)

config = AutoConfig.from_pretrained("facebook/dinov2-small")
config.num_labels = num_labels
config.label2id = label2id
config.id2label = id2label
config.arcface_margin = 28.6
config.arcface_scale = 64.0

model = Dinov2ForImageClassificationWithArcFaceLoss.from_pretrained(
    "facebook/dinov2-small",
    config=config,
    ignore_mismatched_sizes=True,
)
image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")

training_args = TrainingArguments(
    output_dir="/tmp/arcface-run",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    dataloader_num_workers=4,
    learning_rate=1e-3,
    num_train_epochs=10,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    processing_class=image_processor,
    compute_metrics=compute_metrics,
)

trainer.add_callback(
    FreezeScheduleCallback(
        trainer=trainer,
        freeze_schedule=[
            {"epoch": 0.0, "freeze_modules": ["dinov2"], "unfreeze_modules": []},
            {
                "epoch": 3.0,
                "freeze_modules": [],
                "unfreeze_modules": ["dinov2.encoder.layer.11"],
            },
        ],
    )
)
trainer.add_callback(
    KNNCallback(
        trainer=trainer,
        label_column="labels",
        ks=(1, 5),
        embedding_source="cls_mean",
    )
)

trainer.train()
metrics = trainer.evaluate()
```

### Freeze Schedule Callback

`FreezeScheduleCallback` applies module freeze/unfreeze rules based on epoch number.

You can provide the schedule inline without modifying `model.config`:

```python
from transformers_knn_adapter import FreezeScheduleCallback

freeze_schedule = [
    {"epoch": 0.0, "freeze_modules": ["dinov2"], "unfreeze_modules": []},
    {"epoch": 3.0, "freeze_modules": [], "unfreeze_modules": ["dinov2.encoder.layer.11"]},
]

trainer.add_callback(
    FreezeScheduleCallback(
        trainer=trainer,
        freeze_schedule=freeze_schedule,
    )
)
```

Or, if you prefer, store the schedule on `model.config.freeze_schedule` and instantiate the callback without the `freeze_schedule=` argument.

Schedule format:

```python
[
    {
        "epoch": 0.0,
        "freeze_modules": ["dinov2"],
        "unfreeze_modules": [],
    },
    {
        "epoch": 3.0,
        "freeze_modules": [],
        "unfreeze_modules": ["dinov2.encoder.layer.11"],
    },
]
```

The callback:

- applies the schedule at train begin and each epoch begin
- logs trainable parameter count as:
  - `train/trainable_parameters`

Module names must match `model.named_modules()`, for example:

- `dinov2`
- `dinov2.embeddings`
- `dinov2.encoder.layer.11`
- `.` for the whole model

### Smoke Script

The repository includes a practical end-to-end example in:

- [`scripts/dogfaces_smoke.py`](scripts/dogfaces_smoke.py)

It shows how to combine:

- `Dinov2ForImageClassificationWithArcFaceLoss`
- `KNNCallback`
- `FreezeScheduleCallback`
- Hugging Face `Trainer`
- optional W&B logging

Example:

```bash
uv run --with accelerate --with pytorch-metric-learning python scripts/dogfaces_smoke.py \
  --dataset dimidagd/DogFaceNet_224resize \
  --model facebook/dinov2-small \
  --freeze-schedule-config configs/freeze_schedule.backbone_then_unfreeze.json \
  --num-train-epochs 10 \
  --learning-rate 0.001 \
  --warmup-steps 10 \
  --logging-strategy steps \
  --logging-steps 20
```
