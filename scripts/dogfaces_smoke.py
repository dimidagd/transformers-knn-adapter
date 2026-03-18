"""Small finetuning smoke script for DogFaceNet with ArcFace Dinov2 and Trainer.

Example:
    uv run --with accelerate --with pytorch-metric-learning \
      python scripts/dogfaces_smoke.py \
      --dataset dimidagd/DogFaceNet_224resize \
      --model facebook/dinov2-small \
      --num-train-epochs 2
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers import (
    pipeline as hf_pipeline,
)

from transformers_knn_adapter import (
    Dinov2ForImageClassificationWithArcFaceLoss,
    FreezeScheduleCallback,
    KNNCallback,
)
from transformers_knn_adapter.knn_image_pipeline import pipeline as knn_pipeline

DEFAULT_DATASET = "dimidagd/DogFaceNet_224resize"
DEFAULT_MODEL = "facebook/dinov2-small"
DEFAULT_OUTPUT_DIR = Path("/tmp/dogfaces-arcface-smoke")


class ProcessedImageDataset(TorchDataset):
    def __init__(self, dataset: HFDataset, image_processor: Any) -> None:
        self.dataset = dataset
        self.image_processor = image_processor

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.dataset[index]
        image = example["image"]
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL image, got {type(image)!r}")
        if image.mode != "RGB":
            image = image.convert("RGB")
        encoded = self.image_processor(images=image, return_tensors="pt")
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": torch.tensor(example["label"], dtype=torch.long),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset repo id to load.")
    parser.add_argument("--base-split", default="train", help="Base split to load before deterministic shuffling/splitting.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Dinov2 checkpoint to finetune.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Training output directory.")
    parser.add_argument("--train-fraction", type=float, default=0.7, help="Fraction of the base split used for training.")
    parser.add_argument("--train-samples", type=int, default=128, help="Optional cap for train samples after splitting.")
    parser.add_argument("--eval-samples", type=int, default=64, help="Optional cap for eval samples after splitting.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=8, help="Per-device train batch size.")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8, help="Per-device eval batch size.")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Optimizer learning rate.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--num-train-epochs", type=float, default=1.0, help="Number of training epochs.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Override training duration with a step cap. Use a negative value to train for `--num-train-epochs`.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay.")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Warmup steps for the LR scheduler.")
    parser.add_argument(
        "--report-to",
        nargs="+",
        default=("none",),
        help="Trainer reporting integrations, e.g. `none` or `wandb`.",
    )
    parser.add_argument("--run-name", default=None, help="Optional Trainer run name.")
    parser.add_argument("--wandb-project", default=None, help="Optional W&B project name.")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity/team name.")
    parser.add_argument("--wandb-name", default=None, help="Optional W&B run name override.")
    parser.add_argument("--wandb-tags", nargs="*", default=None, help="Optional W&B tags.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--shuffle-seed", type=int, default=42, help="Dataset shuffle seed used before splitting.")
    parser.add_argument("--arcface-margin", type=float, default=28.6, help="ArcFace angular margin.")
    parser.add_argument("--arcface-scale", type=float, default=64.0, help="ArcFace logit scale.")
    parser.add_argument(
        "--use-focal-loss",
        action="store_true",
        help="Use torchvision sigmoid focal loss over ArcFace-adjusted logits instead of cross-entropy.",
    )
    parser.add_argument(
        "--focal-loss-alpha",
        type=float,
        default=0.25,
        help="Alpha parameter for torchvision sigmoid focal loss.",
    )
    parser.add_argument(
        "--focal-loss-gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for torchvision sigmoid focal loss.",
    )
    parser.add_argument(
        "--freeze-schedule-config",
        type=Path,
        default=None,
        help="Optional JSON config file containing a `freeze_schedule` list.",
    )
    parser.add_argument(
        "--knn-embedding-source",
        choices=("cls", "cls_mean"),
        default="cls_mean",
        help="Embedding reduction used by the KNN evaluation callback.",
    )
    parser.add_argument(
        "--disable-knn-callback",
        action="store_true",
        help="Skip attaching the KNN evaluation callback.",
    )
    parser.add_argument("--dataloader-num-workers", type=int, default=0, help="Trainer dataloader worker count.")
    parser.add_argument(
        "--logging-strategy",
        default="epoch",
        choices=("no", "steps", "epoch"),
        help="Trainer logging strategy. Use `epoch` to log only at epoch boundaries.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=1,
        help="Trainer logging interval in steps when `--logging-strategy steps` is used.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=5,
        help="Evaluation interval in steps when `--eval-strategy steps` is used.",
    )
    parser.add_argument("--save-strategy", default="no", choices=("no", "steps", "epoch"), help="Checkpoint save strategy.")
    parser.add_argument(
        "--eval-strategy",
        default="epoch",
        choices=("no", "steps", "epoch"),
        help="Trainer evaluation strategy. Use `epoch` to run the KNN callback after each epoch.",
    )
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable Trainer progress bars.")
    parser.add_argument("--use-cpu", action="store_true", help="Force CPU training even when CUDA is available.")
    return parser


def _subset_dataset(dataset: HFDataset, limit: int | None) -> HFDataset:
    if limit is None or limit >= len(dataset):
        return dataset
    return dataset.select(range(limit))


def load_train_eval_datasets(
    *,
    dataset_name: str,
    base_split: str,
    train_fraction: float,
    shuffle_seed: int,
    train_samples: int | None,
    eval_samples: int | None,
) -> tuple[HFDataset, HFDataset]:
    dataset_path = Path(os.fspath(dataset_name)).expanduser()
    if dataset_path.is_dir():
        dataset = load_dataset("imagefolder", data_dir=str(dataset_path), split=base_split)
    else:
        dataset = load_dataset(dataset_name, split=base_split)

    total = len(dataset)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be between 0 and 1, got {train_fraction}.")
    train_size = int(total * train_fraction)
    if train_size <= 0 or train_size >= total:
        raise ValueError(f"train_fraction must leave non-empty train/eval partitions, got total={total}.")

    labels = np.asarray(dataset["label"])
    indices = np.arange(total)
    train_indices, eval_indices = train_test_split(
        indices,
        train_size=train_fraction,
        random_state=shuffle_seed,
        shuffle=True,
        stratify=labels,
    )
    train_dataset = dataset.select(sorted(train_indices.tolist())).shuffle(seed=shuffle_seed)
    eval_dataset = dataset.select(sorted(eval_indices.tolist()))
    return _subset_dataset(train_dataset, train_samples), _subset_dataset(eval_dataset, eval_samples)


def collate_fn(examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.stack([example["pixel_values"] for example in examples]),
        "labels": torch.stack([example["labels"] for example in examples]),
    }


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.asarray(logits).argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def load_freeze_schedule_config(config_path: Path | None) -> list[dict[str, Any]]:
    if config_path is None:
        return []

    payload = json.loads(config_path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        freeze_schedule = payload.get("freeze_schedule", [])
        if not isinstance(freeze_schedule, list):
            raise ValueError("freeze_schedule config file must contain a list under `freeze_schedule`.")
        return freeze_schedule
    raise ValueError("freeze_schedule config file must be either a list or an object with `freeze_schedule`.")


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    if args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_name is not None:
        os.environ["WANDB_NAME"] = args.wandb_name
    if args.wandb_tags:
        os.environ["WANDB_TAGS"] = ",".join(args.wandb_tags)

    train_dataset, eval_dataset = load_train_eval_datasets(
        dataset_name=args.dataset,
        base_split=args.base_split,
        train_fraction=args.train_fraction,
        shuffle_seed=args.shuffle_seed,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
    )

    label_feature = train_dataset.features["label"]
    if not hasattr(label_feature, "names"):
        raise ValueError("Expected a ClassLabel 'label' feature on the dataset.")

    label_names = list(label_feature.names)
    image_processor = AutoImageProcessor.from_pretrained(args.model)

    config = AutoConfig.from_pretrained(args.model)
    config.num_labels = len(label_names)
    config.label2id = {label: idx for idx, label in enumerate(label_names)}
    config.id2label = {idx: label for idx, label in enumerate(label_names)}
    config.arcface_margin = args.arcface_margin
    config.arcface_scale = args.arcface_scale
    config.use_focal_loss = args.use_focal_loss
    config.focal_loss_alpha = args.focal_loss_alpha
    config.focal_loss_gamma = args.focal_loss_gamma
    config.freeze_schedule = load_freeze_schedule_config(args.freeze_schedule_config)
    config.embedding_source = args.knn_embedding_source

    model = Dinov2ForImageClassificationWithArcFaceLoss.from_pretrained(
        args.model,
        config=config,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        do_train=True,
        do_eval=True,
        remove_unused_columns=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        report_to=list(args.report_to),
        run_name=args.run_name,
        seed=args.seed,
        data_seed=args.shuffle_seed,
        dataloader_num_workers=args.dataloader_num_workers,
        use_cpu=args.use_cpu,
        save_only_model=True,
        disable_tqdm=args.disable_tqdm,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ProcessedImageDataset(train_dataset, image_processor),
        eval_dataset=ProcessedImageDataset(eval_dataset, image_processor),
        data_collator=collate_fn,
        processing_class=image_processor,
        compute_metrics=compute_metrics,
    )
    if not args.disable_knn_callback:
        trainer.add_callback(
            KNNCallback(
                trainer=trainer,
                label_column="labels",
                ks=(1,),
                embedding_source=args.knn_embedding_source,
            )
        )
    if config.freeze_schedule:
        trainer.add_callback(FreezeScheduleCallback(trainer=trainer))

    pretrain_eval_metrics = trainer.evaluate()
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    trainer.save_model(str(args.output_dir))
    image_processor.save_pretrained(args.output_dir)
    reloaded_backbone = AutoModel.from_pretrained(str(args.output_dir))
    backbone_export_dir = args.output_dir / "auto_model_export"
    backbone_export_dir.mkdir(parents=True, exist_ok=True)
    reloaded_backbone.save_pretrained(str(backbone_export_dir))
    image_processor.save_pretrained(backbone_export_dir)
    pipeline_probe: dict[str, Any]
    try:
        image_classification_pipeline = hf_pipeline(
            "image-classification",
            model=str(args.output_dir),
            image_processor=str(args.output_dir),
            device=-1 if args.use_cpu else 0,
        )
        pipeline_result = image_classification_pipeline(eval_dataset[0]["image"], num_workers=0)
        pipeline_probe = {
            "task": "image-classification",
            "status": "ok",
            "pipeline_class": type(image_classification_pipeline).__name__,
            "result_preview": pipeline_result[:3] if isinstance(pipeline_result, list) else pipeline_result,
        }
    except Exception as exc:
        pipeline_probe = {
            "task": "image-classification",
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    backbone_pipeline_probe: dict[str, Any]
    try:
        backbone_image_classification_pipeline = hf_pipeline(
            "image-classification",
            model=str(backbone_export_dir),
            image_processor=str(backbone_export_dir),
            device=-1 if args.use_cpu else 0,
        )
        backbone_pipeline_result = backbone_image_classification_pipeline(eval_dataset[0]["image"], num_workers=0)
        backbone_pipeline_probe = {
            "task": "image-classification",
            "status": "ok",
            "pipeline_class": type(backbone_image_classification_pipeline).__name__,
            "result_preview": (
                backbone_pipeline_result[:3] if isinstance(backbone_pipeline_result, list) else backbone_pipeline_result
            ),
        }
    except Exception as exc:
        backbone_pipeline_probe = {
            "task": "image-classification",
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    feature_extraction_probe: dict[str, Any]
    try:
        feature_extraction_pipeline = hf_pipeline(
            "image-feature-extraction",
            model=str(backbone_export_dir),
            image_processor=str(backbone_export_dir),
            device=-1 if args.use_cpu else 0,
        )
        feature_result = feature_extraction_pipeline(eval_dataset[0]["image"], num_workers=0)
        feature_shape: list[int] | None = None
        if isinstance(feature_result, list):
            array = np.asarray(feature_result)
            feature_shape = list(array.shape)
        feature_extraction_probe = {
            "task": "image-feature-extraction",
            "status": "ok",
            "pipeline_class": type(feature_extraction_pipeline).__name__,
            "output_shape": feature_shape,
        }
    except Exception as exc:
        feature_extraction_probe = {
            "task": "image-feature-extraction",
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    custom_pipeline_probe: dict[str, Any]
    try:
        knn_model_path = args.output_dir / "custom_pipeline_knn.joblib"
        custom_clf = knn_pipeline(
            "image-classification",
            model_path=str(backbone_export_dir),
            knn_model_path=str(knn_model_path),
            device=-1 if args.use_cpu else 0,
            top_k=3,
        )
        custom_clf.train(
            train_dataset,
            batch_size=args.per_device_train_batch_size,
            num_workers=0,
            max_samples=min(len(train_dataset), 16),
            n_neighbors=1,
            save_knn_model_path=knn_model_path,
        )
        custom_pipeline_result = custom_clf(eval_dataset[0]["image"], num_workers=0)
        custom_pipeline_probe = {
            "task": "image-classification",
            "status": "ok",
            "pipeline_class": type(custom_clf).__name__,
            "knn_model_path": str(knn_model_path),
            "result_preview": (
                custom_pipeline_result[:3] if isinstance(custom_pipeline_result, list) else custom_pipeline_result
            ),
        }
    except Exception as exc:
        custom_pipeline_probe = {
            "task": "image-classification",
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    saved_model_custom_pipeline_probe: dict[str, Any]
    try:
        saved_model_knn_path = args.output_dir / "saved_model_custom_pipeline_knn.joblib"
        saved_model_custom_clf = knn_pipeline(
            "image-classification",
            model_path=str(args.output_dir),
            knn_model_path=str(saved_model_knn_path),
            device=-1 if args.use_cpu else 0,
            top_k=3,
        )
        saved_model_custom_clf.train(
            train_dataset,
            batch_size=args.per_device_train_batch_size,
            num_workers=0,
            max_samples=min(len(train_dataset), 16),
            n_neighbors=1,
            save_knn_model_path=saved_model_knn_path,
        )
        saved_model_custom_pipeline_result = saved_model_custom_clf(eval_dataset[0]["image"], num_workers=0)
        saved_model_custom_pipeline_probe = {
            "task": "image-classification",
            "status": "ok",
            "pipeline_class": type(saved_model_custom_clf).__name__,
            "knn_model_path": str(saved_model_knn_path),
            "result_preview": (
                saved_model_custom_pipeline_result[:3]
                if isinstance(saved_model_custom_pipeline_result, list)
                else saved_model_custom_pipeline_result
            ),
        }
    except Exception as exc:
        saved_model_custom_pipeline_probe = {
            "task": "image-classification",
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    return {
        "dataset": args.dataset,
        "base_split": args.base_split,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "model": args.model,
        "knn_callback_enabled": not args.disable_knn_callback,
        "report_to": list(args.report_to),
        "run_name": args.run_name,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_name": args.wandb_name,
        "wandb_tags": args.wandb_tags or [],
        "pretrain_eval_metrics": pretrain_eval_metrics,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "freeze_schedule": config.freeze_schedule,
        "use_focal_loss": config.use_focal_loss,
        "focal_loss_alpha": config.focal_loss_alpha,
        "focal_loss_gamma": config.focal_loss_gamma,
        "model_embedding_source": config.embedding_source,
        "saved_model_class": type(model).__name__,
        "auto_model_reload_class": type(reloaded_backbone).__name__,
        "auto_model_export_dir": str(backbone_export_dir),
        "pipeline_probe": pipeline_probe,
        "backbone_pipeline_probe": backbone_pipeline_probe,
        "feature_extraction_probe": feature_extraction_probe,
        "custom_pipeline_probe": custom_pipeline_probe,
        "saved_model_custom_pipeline_probe": saved_model_custom_pipeline_probe,
        "output_dir": str(args.output_dir),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
