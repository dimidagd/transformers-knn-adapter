"""Hugging Face image classification pipeline with a scikit-learn KNN head.

This module exposes a custom pipeline class and a factory with a familiar
signature:

    pipeline("image-classification", model_path, knn_model_path="...")
"""

from __future__ import annotations

import argparse
import io
import logging
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import IterableDataset
from PIL import Image
from sklearn.base import ClassifierMixin
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel
from transformers.pipelines import ImageClassificationPipeline

logger = logging.getLogger(__name__)
GRID_SEARCH_ALLOWED_SCORING = {"f1_macro", "precision_macro", "recall_macro"}
DEFAULT_N_NEIGHBORS = 5


class KNNImageClassificationPipeline(ImageClassificationPipeline):
    """Image classification pipeline using transformer embeddings + sklearn KNN."""

    def __init__(
        self,
        *args: Any,
        knn_model_path: str | Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.knn_model_path = str(knn_model_path)
        self.knn_model: ClassifierMixin | None = None
        if Path(self.knn_model_path).exists():
            logger.info("Loading KNN model from %s", self.knn_model_path)
            self.knn_model = joblib.load(self.knn_model_path)
            self._validate_knn_model(self.knn_model)
            logger.info("Loaded KNN model with %d classes", len(self.knn_model.classes_))

    @staticmethod
    def _validate_knn_model(model: ClassifierMixin) -> None:
        if not hasattr(model, "predict_proba"):
            raise ValueError("Loaded KNN model must implement predict_proba().")
        if not hasattr(model, "classes_"):
            raise ValueError("Loaded KNN model must expose classes_.")

    @staticmethod
    def _coerce_image(image_value: Any) -> Image.Image:
        if isinstance(image_value, Image.Image):
            return image_value.convert("RGB")
        if isinstance(image_value, np.ndarray):
            return Image.fromarray(image_value).convert("RGB")
        if isinstance(image_value, dict):
            image_bytes = image_value.get("bytes")
            image_path = image_value.get("path")
            if image_bytes is not None:
                return Image.open(io.BytesIO(image_bytes)).convert("RGB")
            if image_path:
                return Image.open(image_path).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image_value)!r}")

    def _resolve_dataset(self, dataset: Any, split: str, streaming: bool = False) -> HFDataset | IterableDataset:
        if isinstance(dataset, str):
            try:
                from datasets import load_dataset
            except Exception as exc:  # pragma: no cover - dependency/runtime detail
                raise ImportError("`datasets` package is required when dataset is a string name.") from exc
            dataset_obj = load_dataset(dataset, split=split, streaming=streaming)
            if not isinstance(dataset_obj, (HFDataset, IterableDataset)):
                raise TypeError("Only Hugging Face Dataset/IterableDataset is supported.")
            return dataset_obj

        dataset_obj = dataset[split] if hasattr(dataset, "keys") and split in dataset else dataset
        if not isinstance(dataset_obj, (HFDataset, IterableDataset)):
            raise TypeError("Only Hugging Face Dataset/IterableDataset is supported.")
        return dataset_obj

    @staticmethod
    def _normalize_label(raw_label: Any, label_names: list[str] | None) -> Any:
        if label_names is not None and isinstance(raw_label, (int, np.integer)) and 0 <= int(raw_label) < len(label_names):
            return label_names[int(raw_label)]
        return raw_label

    def _materialize_training_features_and_labels(
        self,
        *,
        dataset_obj: IterableDataset,
        image_column: str,
        label_column: str,
        batch_size: int,
        max_samples: int | None,
        label_names: list[str] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.memmap | None, str | None, int]:
        """Extract embeddings and labels, optionally using memmap storage for known sample counts."""
        self.model.eval()
        embedding_batches: list[np.ndarray] = []
        embeddings_mm: np.memmap | None = None
        memmap_path: str | None = None
        write_idx = 0
        labels: list[Any] = []
        loaded = 0
        dataset_for_batches = dataset_obj.take(max_samples) if max_samples is not None else dataset_obj
        batched_iter = dataset_for_batches.batch(batch_size=batch_size)
        total_samples = max_samples if max_samples is not None else (len(dataset_obj) if hasattr(dataset_obj, "__len__") else None)
        total_batches = math.ceil(total_samples / batch_size) if total_samples is not None else None

        for batch in tqdm(
            batched_iter,
            total=total_batches,
            desc="Extracting train embeddings",
        ):
            image_values = batch[image_column]
            label_values = batch[label_column]
            batch_images = [self._coerce_image(image_value) for image_value in image_values]
            labels.extend(self._normalize_label(raw_label, label_names) for raw_label in label_values)
            loaded += len(label_values)
            if self.image_processor is None:
                raise ValueError("image_processor is not configured.")
            model_inputs = self.image_processor(images=batch_images, return_tensors="pt")
            model_inputs_on_device: dict[str, Any] = {
                k: v.to(self.device) if hasattr(v, "to") else v for k, v in model_inputs.items()
            }

            if self.model is None:
                raise ValueError("model is not configured.")
            with torch.inference_mode():
                model_outputs = self.model(**model_inputs_on_device)
            embeddings = self._extract_embeddings(model_outputs).detach().cpu().numpy()
            if total_samples is not None:
                if embeddings_mm is None:
                    with tempfile.NamedTemporaryFile(prefix="knn_emb_", suffix=".mmap", delete=False) as f:
                        memmap_path = f.name
                    embeddings_mm = np.memmap(
                        memmap_path,
                        dtype=embeddings.dtype,
                        mode="w+",
                        shape=(int(total_samples), embeddings.shape[1]),
                    )
                    logger.info("Using memory-mapped embedding storage at %s", memmap_path)
                next_idx = write_idx + embeddings.shape[0]
                embeddings_mm[write_idx:next_idx] = embeddings
                write_idx = next_idx
            else:
                # When total sample count is unknown, fallback to in-memory chunks.
                embedding_batches.append(embeddings)
            logger.debug("Processed embedding batch size=%d", len(batch_images))

        if loaded == 0:
            raise ValueError("Dataset is empty; cannot train KNN.")
        logger.info("Loaded %d items for training", loaded)

        if embeddings_mm is not None:
            embeddings_mm.flush()
            features = np.asarray(embeddings_mm[:loaded])
        else:
            features = np.concatenate(embedding_batches, axis=0)
        y = np.asarray(labels, dtype=object)
        return features, y, embeddings_mm, memmap_path, loaded

    def _fit_knn_from_features(
        self,
        *,
        features: np.ndarray,
        y: np.ndarray,
        loaded: int,
        n_neighbors: int,
        weights: str,
        metric: str,
        grid_search: bool,
        grid_search_splits: int,
        grid_search_repeats: int,
        grid_search_scoring: str,
        shuffle_seed: int,
    ) -> ClassifierMixin:
        """Fit KNN directly or via grid search from materialized features/labels."""
        if grid_search:
            metric_candidates = ["euclidean", "manhattan", "chebyshev", "minkowski", "cosine"]
            label_counts = Counter(y.tolist())
            min_class_count = min(label_counts.values())
            effective_cv = min(grid_search_splits, int(min_class_count))
            if effective_cv <= 1:
                logger.warning(
                    "Skipping grid search because smallest class count (%d) is too small for CV; "
                    "falling back to n_neighbors=%d metric=%s",
                    min_class_count,
                    n_neighbors,
                    metric,
                )
                knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
                knn.fit(features, y)
                return knn

            min_train_samples_per_fold = loaded - math.ceil(loaded / effective_cv)
            neighbor_candidates = [k for k in [1, 2, 4, 8, 16, 32] if k <= min_train_samples_per_fold]
            if not neighbor_candidates:
                raise ValueError("Not enough training samples per CV fold to run KNN grid search.")
            logger.info(
                "Running KNN grid search: neighbors=%s metrics=%s cv=%d repeats=%d min_train_samples_per_fold=%d (repeated stratified folds)",
                neighbor_candidates,
                metric_candidates,
                effective_cv,
                grid_search_repeats,
                min_train_samples_per_fold,
            )
            cv_splitter = RepeatedStratifiedKFold(
                n_splits=effective_cv,
                n_repeats=grid_search_repeats,
                random_state=shuffle_seed,
            )
            search = GridSearchCV(
                estimator=KNeighborsClassifier(weights=weights, algorithm="brute"),
                param_grid={
                    "n_neighbors": neighbor_candidates,
                    "metric": metric_candidates,
                },
                cv=cv_splitter,
                scoring=grid_search_scoring,
                n_jobs=-1,
                refit=True,
            )
            search.fit(features, y)
            best_knn = search.best_estimator_
            assert best_knn is not None
            logger.info(
                "Grid search complete: best_params=%s best_score=%.4f",
                search.best_params_,
                float(search.best_score_),
            )
            return best_knn

        knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
        knn.fit(features, y)
        return knn

    def train(
        self,
        dataset: Any,
        *,
        split: str = "train",
        image_column: str = "image",
        label_column: str = "label",
        batch_size: int = 16,
        streaming: bool = False,
        stratified: bool = False,
        shuffle: bool = False,
        shuffle_seed: int = 42,
        shuffle_buffer_size: int = 1000,
        max_samples: int | None = None,
        n_neighbors: int = DEFAULT_N_NEIGHBORS,
        weights: str = "distance",
        metric: str = "minkowski",
        grid_search: bool = False,
        grid_search_splits: int = 3,
        grid_search_repeats: int = 2,
        grid_search_scoring: str | None = None,
        save_knn_model_path: str | Path | None = None,
    ) -> ClassifierMixin:
        """Train and attach a KNN head from extracted embeddings.

        `dataset` can be:
        - a Hugging Face dataset name (e.g. ``"timm/mini-imagenet"``), or
        - a loaded `datasets.Dataset` / `datasets.DatasetDict`, or
        - a loaded `datasets.IterableDataset` / `datasets.IterableDatasetDict`.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if shuffle_buffer_size <= 0:
            raise ValueError("shuffle_buffer_size must be > 0")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be > 0")
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be > 0")
        if grid_search:
            if grid_search_splits <= 1:
                raise ValueError("grid_search_splits must be > 1 when grid_search=True.")
            if grid_search_repeats <= 0:
                raise ValueError("grid_search_repeats must be > 0 when grid_search=True.")
        if grid_search and grid_search_scoring is None:
            raise ValueError("grid_search_scoring is required when grid_search=True.")
        if grid_search and grid_search_scoring not in GRID_SEARCH_ALLOWED_SCORING:
            raise ValueError(
                f"grid_search_scoring must be one of {sorted(GRID_SEARCH_ALLOWED_SCORING)} when grid_search=True."
            )
        if not grid_search and grid_search_scoring is not None:
            raise ValueError("grid_search_scoring can only be set when grid_search=True.")
        if streaming and stratified:
            raise ValueError("stratified mode is only supported when streaming=False.")

        logger.info(
            "Starting KNN training: split=%s image_column=%s label_column=%s batch_size=%d n_neighbors=%d streaming=%s stratified=%s shuffle=%s shuffle_seed=%d max_samples=%s",
            split,
            image_column,
            label_column,
            batch_size,
            n_neighbors,
            streaming,
            stratified,
            shuffle,
            shuffle_seed,
            max_samples,
        )
        dataset_obj = self._resolve_dataset(dataset=dataset, split=split, streaming=streaming)

        if stratified:
            if not isinstance(dataset_obj, HFDataset):
                raise ValueError("stratified mode requires a non-streaming Hugging Face Dataset.")
            if max_samples is not None:
                dataset_obj = dataset_obj.train_test_split(
                    train_size=max_samples,
                    stratify_by_column=label_column,
                    seed=shuffle_seed,
                )["train"]
            logger.info("Applied stratified sampling for training subset")

        if not isinstance(dataset_obj, IterableDataset):
            dataset_obj = dataset_obj.to_iterable_dataset()
            
        if shuffle:
            if isinstance(dataset_obj, IterableDataset):
                dataset_obj = dataset_obj.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer_size)
            else:
                dataset_obj = dataset_obj.shuffle(seed=shuffle_seed)

        label_names = None
        if hasattr(dataset_obj, "features") and label_column in dataset_obj.features:
            feature = dataset_obj.features[label_column]
            label_names = getattr(feature, "names", None)

        features, y, embeddings_mm, memmap_path, loaded = self._materialize_training_features_and_labels(
            dataset_obj=dataset_obj,
            image_column=image_column,
            label_column=label_column,
            batch_size=batch_size,
            max_samples=max_samples,
            label_names=label_names,
        )

        try:
            knn = self._fit_knn_from_features(
                features=features,
                y=y,
                loaded=loaded,
                n_neighbors=n_neighbors,
                weights=weights,
                metric=metric,
                grid_search=grid_search,
                grid_search_splits=grid_search_splits,
                grid_search_repeats=grid_search_repeats,
                grid_search_scoring=grid_search_scoring or "f1_macro",
                shuffle_seed=shuffle_seed,
            )
        finally:
            if embeddings_mm is not None:
                del embeddings_mm
            if memmap_path is not None:
                try:
                    os.remove(memmap_path)
                except OSError:
                    logger.warning("Failed to remove temporary memmap file: %s", memmap_path)
        self._validate_knn_model(knn)
        self.knn_model = knn
        logger.info("KNN training complete: features_shape=%s classes=%d", features.shape, len(knn.classes_))

        output_path = Path(save_knn_model_path) if save_knn_model_path is not None else Path(self.knn_model_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(knn, output_path)
        self.knn_model_path = str(output_path)
        logger.info("Saved trained KNN model to %s", self.knn_model_path)
        return knn

    def evaluate(
        self,
        dataset: Any,
        *,
        split: str = "validation",
        image_column: str = "image",
        label_column: str = "label",
        batch_size: int = 16,
        streaming: bool = False,
        stratified: bool = False,
        shuffle: bool = False,
        shuffle_seed: int = 42,
        shuffle_buffer_size: int = 1000,
        max_samples: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate top-1 accuracy on a dataset or dataset split."""
        if self.knn_model is None:
            raise ValueError("KNN head is not loaded. Provide knn_model_path or call train(...) first.")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if shuffle_buffer_size <= 0:
            raise ValueError("shuffle_buffer_size must be > 0")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be > 0")
        if streaming and stratified:
            raise ValueError("stratified mode is only supported when streaming=False.")

        dataset_obj = self._resolve_dataset(dataset=dataset, split=split, streaming=streaming)
        if stratified:
            if not isinstance(dataset_obj, HFDataset):
                raise ValueError("stratified mode requires a non-streaming Hugging Face Dataset.")
            if max_samples is not None:
                dataset_obj = dataset_obj.train_test_split(
                    train_size=max_samples,
                    stratify_by_column=label_column,
                    seed=shuffle_seed,
                )["train"]
                max_samples = None
        if shuffle:
            if isinstance(dataset_obj, IterableDataset):
                dataset_obj = dataset_obj.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer_size)
            else:
                dataset_obj = dataset_obj.shuffle(seed=shuffle_seed)

        label_names = None
        if hasattr(dataset_obj, "features") and label_column in dataset_obj.features:
            label_feature = dataset_obj.features[label_column]
            label_names = getattr(label_feature, "names", None)

        if isinstance(dataset_obj, IterableDataset):
            dataset_for_eval = dataset_obj.take(max_samples) if max_samples is not None else dataset_obj
            total_samples = max_samples
        else:
            if max_samples is not None:
                dataset_obj = dataset_obj.select(range(min(max_samples, len(dataset_obj))))
            total_samples = len(dataset_obj)
            dataset_for_eval = dataset_obj.to_iterable_dataset()

        batched_iter = dataset_for_eval.batch(batch_size=batch_size)
        total_batches = math.ceil(total_samples / batch_size) if total_samples is not None else None
        rows: list[tuple[str, str, bool]] = []
        for batch in tqdm(batched_iter, total=total_batches, desc="Evaluating"):
            image_values = batch[image_column]
            label_values = batch[label_column]
            batch_images = [self._coerce_image(image_value) for image_value in image_values]
            batch_pred = self(batch_images)

            if batch_pred and isinstance(batch_pred[0], dict):
                pred_labels = [str(batch_pred[0]["label"])]
            else:
                pred_labels = [str(pred[0]["label"]) for pred in batch_pred]

            for true_raw, pred_label in zip(label_values, pred_labels, strict=True):
                true_label = str(self._normalize_label(true_raw, label_names))
                rows.append((true_label, pred_label, true_label == pred_label))

        if not rows:
            raise ValueError("Dataset is empty; cannot evaluate.")

        correct = sum(int(r[2]) for r in rows)
        total = len(rows)
        accuracy = correct / total
        y_true = [r[0] for r in rows]
        y_pred = [r[1] for r in rows]
        report_text = classification_report(y_true, y_pred, zero_division=0)
        true_counts = dict(Counter(r[0] for r in rows))
        pred_counts = dict(Counter(r[1] for r in rows))

        metrics = {
            "split": split,
            "samples": total,
            "top1_accuracy": accuracy,
            "correct": correct,
            "classification_report": report_text,
            "true_label_counts": true_counts,
            "pred_label_counts": pred_counts,
        }
        logger.info("Evaluation complete: split=%s samples=%d top1_accuracy=%.4f", split, total, accuracy)
        logger.info("Classification report:\n%s", report_text)
        return metrics

    def _extract_embeddings(self, model_outputs: Any) -> torch.Tensor:
        """Extract a 2D embedding tensor from transformer model outputs."""
        if getattr(model_outputs, "pooler_output", None) is not None:
            embeddings = cast(torch.Tensor, model_outputs.pooler_output)
        elif getattr(model_outputs, "last_hidden_state", None) is not None:
            embeddings = cast(torch.Tensor, model_outputs.last_hidden_state[:, 0, :])
        else:
            raise ValueError("Model outputs do not contain pooler_output or last_hidden_state.")
        embeddings = embeddings.flatten(start_dim=1)
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {tuple(embeddings.shape)}")
        return embeddings

    def _forward(self, model_inputs: dict[str, Any], **forward_params: Any) -> Any:
        if self.knn_model is None:
            raise ValueError("KNN head is not loaded. Provide knn_model_path or call train(...) first.")
        logger.debug("Running forward pass with KNN head")
        model_outputs = self.model(**model_inputs, **forward_params)
        embeddings = self._extract_embeddings(model_outputs)

        probs_np = self.knn_model.predict_proba(embeddings.detach().cpu().numpy())
        probs = torch.from_numpy(np.asarray(probs_np, dtype=np.float32))
        return {"probs": probs}

    def postprocess(
        self,
        model_outputs: Any,
        function_to_apply: str | None = None,
        top_k: int | None = 1,
        _legacy: bool = True,
        **postprocess_parameters: Any,
    ) -> Any:
        # function_to_apply is ignored because KNN returns calibrated probabilities.
        del function_to_apply, _legacy
        del postprocess_parameters
        if self.knn_model is None:
            raise ValueError("KNN head is not loaded. Provide knn_model_path or call train(...) first.")

        probs = model_outputs["probs"]
        if probs.ndim == 1:
            probs = probs.unsqueeze(0)

        classes = self.knn_model.classes_
        num_classes = probs.shape[-1]
        if top_k is None:
            top_k = num_classes
        top_k = max(1, min(int(top_k), num_classes))

        batched_results: list[list[dict[str, float | str]]] = []
        for row in probs:
            values, indices = torch.topk(row, k=top_k)
            results: list[dict[str, float | str]] = [
                {"label": str(classes[int(idx)]), "score": float(score)}
                for score, idx in zip(values.tolist(), indices.tolist(), strict=True)
            ]
            batched_results.append(results)

        if len(batched_results) == 1:
            return batched_results[0]
        return batched_results


def pipeline(
    task: str,
    model_path: str | Path,
    *,
    knn_model_path: str | Path,
    **kwargs: Any,
) -> KNNImageClassificationPipeline:
    """Build a KNN-backed image classification pipeline.

    Parameters match the familiar Transformers entrypoint, with one required
    extension: ``knn_model_path``.
    """
    if task != "image-classification":
        raise ValueError("Only task='image-classification' is supported.")

    model = AutoModel.from_pretrained(str(model_path))
    image_processor = AutoImageProcessor.from_pretrained(str(model_path))

    return KNNImageClassificationPipeline(
        model=model,
        image_processor=image_processor,
        knn_model_path=knn_model_path,
        **kwargs,
    )


__all__ = ["KNNImageClassificationPipeline", "pipeline"]


def _validate_cli_train_args(args: argparse.Namespace) -> None:
    """Validate CLI argument combinations for training-related commands."""
    if args.grid_search and args.n_neighbors is not None:
        raise ValueError("--n-neighbors is mutually exclusive with --grid-search.")
    if args.grid_search_splits is not None and not args.grid_search:
        raise ValueError("--grid-search-splits can only be used with --grid-search.")
    if args.grid_search_repeats is not None and not args.grid_search:
        raise ValueError("--grid-search-repeats can only be used with --grid-search.")
    if args.grid_search_scoring is not None and not args.grid_search:
        raise ValueError("--grid-search-scoring can only be used with --grid-search.")
    if args.grid_search and args.grid_search_scoring is None:
        raise ValueError("--grid-search-scoring is required when --grid-search is enabled.")


def _train_pipeline_from_args(args: argparse.Namespace) -> KNNImageClassificationPipeline:
    """Build and train a pipeline instance from parsed CLI arguments."""
    _validate_cli_train_args(args)

    clf = pipeline(
        "image-classification",
        model_path=args.model,
        knn_model_path=args.knn_model_path,
        device=args.device,
        top_k=args.top_k,
    )

    logger.info(
        "CLI train using dataset=%s split=%s image_column=%s label_column=%s",
        args.dataset,
        args.split,
        args.image_column,
        args.label_column,
    )
    clf.train(
        dataset=args.dataset,
        split=args.split,
        image_column=args.image_column,
        label_column=args.label_column,
        batch_size=args.batch_size,
        streaming=args.stream,
        stratified=args.stratified,
        shuffle=args.shuffle,
        shuffle_seed=args.shuffle_seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
        max_samples=args.max_samples,
        n_neighbors=args.n_neighbors if args.n_neighbors is not None else DEFAULT_N_NEIGHBORS,
        grid_search=args.grid_search,
        grid_search_splits=args.grid_search_splits if args.grid_search_splits is not None else 3,
        grid_search_repeats=args.grid_search_repeats if args.grid_search_repeats is not None else 2,
        grid_search_scoring=args.grid_search_scoring,
        save_knn_model_path=args.knn_model_path,
    )
    return clf


def _run_cli_train(args: argparse.Namespace) -> None:
    """Handle the `train` CLI command."""
    _train_pipeline_from_args(args)

    logger.info("Training complete. Saved KNN model to %s", args.knn_model_path)


def _run_cli_infer(args: argparse.Namespace) -> None:
    """Handle the `infer` CLI command using an already-trained KNN model."""
    if args.inference_batch_size <= 0:
        raise ValueError("--inference-batch-size must be > 0.")
    clf = pipeline(
        "image-classification",
        model_path=args.model,
        knn_model_path=args.knn_model_path,
        device=args.device,
        top_k=args.top_k,
    )
    image_input = args.image
    image_batch = [image_input for _ in range(args.inference_batch_size)]
    single_result = clf(image_input)
    batch_result = clf(image_batch)
    logger.info("Single-image inference input: %s", image_input)
    logger.info("Single-image inference result: %s", single_result)
    logger.info("Batch inference URLs (count=%d): %s", args.inference_batch_size, image_batch)
    logger.info("Batch inference result: %s", batch_result)


def _run_cli_predict(args: argparse.Namespace) -> None:
    """Handle the `predict` CLI command."""
    clf = pipeline(
        "image-classification",
        model_path=args.model,
        knn_model_path=args.knn_model_path,
        device=args.device,
        top_k=args.top_k,
    )
    result = clf(args.image)
    logger.info("Prediction result: %s", result)


def _run_cli_eval(args: argparse.Namespace) -> None:
    """Handle the `eval` CLI command."""
    clf = pipeline(
        "image-classification",
        model_path=args.model,
        knn_model_path=args.knn_model_path,
        device=args.device,
        top_k=args.top_k,
    )
    metrics = clf.evaluate(
        dataset=args.dataset,
        split=args.split,
        image_column=args.image_column,
        label_column=args.label_column,
        batch_size=args.batch_size,
        streaming=args.stream,
        stratified=args.stratified,
        shuffle=args.shuffle,
        shuffle_seed=args.shuffle_seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
        max_samples=args.max_samples,
    )
    logger.info("Eval metrics: %s", metrics)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(description="KNN image pipeline CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train KNN head and save it.")
    train_parser.add_argument("--model", required=True, help="HF model id/path for feature extraction.")
    train_parser.add_argument("--knn-model-path", required=True, help="Path to save/load KNN model (.joblib).")
    train_parser.add_argument("--dataset", required=True, help="HF dataset name (example: timm/mini-imagenet).")
    train_parser.add_argument("--split", default="train", help="Dataset split when --dataset is provided.")
    train_parser.add_argument("--image-column", default="image", help="Dataset image column.")
    train_parser.add_argument("--label-column", default="label", help="Dataset label column.")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size.")
    stream_group = train_parser.add_mutually_exclusive_group()
    stream_group.add_argument("--stream", action="store_true", help="Enable HF dataset streaming mode.")
    stream_group.add_argument(
        "--stratified",
        action="store_true",
        help="Enable stratified sampling (non-streaming only; uses --max-samples as subset size).",
    )
    train_parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset before sampling/training.")
    train_parser.add_argument("--shuffle-seed", type=int, default=42, help="Shuffle seed.")
    train_parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=1000,
        help="Streaming shuffle buffer size (used with --stream --shuffle).",
    )
    train_parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on training samples.")
    neighbors_group = train_parser.add_mutually_exclusive_group()
    neighbors_group.add_argument("--n-neighbors", type=int, default=None, help="KNN neighbors.")
    neighbors_group.add_argument(
        "--grid-search",
        action="store_true",
        help="Run GridSearchCV over n_neighbors=[1,2,4,8,16,32] and multiple distance metrics.",
    )
    train_parser.add_argument(
        "--grid-search-splits",
        type=int,
        default=None,
        help="Number of stratified splits per repeat for --grid-search.",
    )
    train_parser.add_argument(
        "--grid-search-repeats",
        type=int,
        default=None,
        help="Number of repeats for repeated stratified splits in --grid-search.",
    )
    train_parser.add_argument(
        "--grid-search-scoring",
        choices=sorted(GRID_SEARCH_ALLOWED_SCORING),
        default=None,
        help="GridSearchCV scoring metric (required with --grid-search).",
    )
    train_parser.add_argument("--top-k", type=int, default=2, help="Top-k at inference time.")
    train_parser.add_argument("--device", type=int, default=-1, help="Transformers device index (-1 for CPU).")

    infer_parser = subparsers.add_parser(
        "infer",
        help="Run inference on one image and a list of images using a trained KNN head.",
    )
    infer_parser.add_argument("--model", required=True, help="HF model id/path for feature extraction.")
    infer_parser.add_argument("--knn-model-path", required=True, help="Path to save/load KNN model (.joblib).")
    infer_parser.add_argument("--top-k", type=int, default=3, help="Top-k predictions.")
    infer_parser.add_argument("--device", type=int, default=-1, help="Transformers device index (-1 for CPU).")
    infer_parser.add_argument(
        "--image",
        default="https://picsum.photos/200",
        help="Image input used for single and batched inference (file path or URL).",
    )
    infer_parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=5,
        help="Number of images to send as batched inference requests.",
    )

    predict_parser = subparsers.add_parser("predict", help="Run inference using trained KNN head.")
    predict_parser.add_argument("--model", required=True, help="HF model id/path for feature extraction.")
    predict_parser.add_argument("--knn-model-path", required=True, help="Path to trained KNN model (.joblib).")
    predict_parser.add_argument("--image", required=True, help="Image path/URL accepted by transformers image pipeline.")
    predict_parser.add_argument("--top-k", type=int, default=3, help="Top-k predictions.")
    predict_parser.add_argument("--device", type=int, default=-1, help="Transformers device index (-1 for CPU).")

    eval_parser = subparsers.add_parser("eval", help="Evaluate trained KNN head on a dataset split.")
    eval_parser.add_argument("--model", required=True, help="HF model id/path for feature extraction.")
    eval_parser.add_argument("--knn-model-path", required=True, help="Path to trained KNN model (.joblib).")
    eval_parser.add_argument("--dataset", required=True, help="HF dataset name (example: timm/mini-imagenet).")
    eval_parser.add_argument("--split", default="validation", help="Dataset split.")
    eval_parser.add_argument("--image-column", default="image", help="Dataset image column.")
    eval_parser.add_argument("--label-column", default="label", help="Dataset label column.")
    eval_parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size.")
    eval_stream_group = eval_parser.add_mutually_exclusive_group()
    eval_stream_group.add_argument("--stream", action="store_true", help="Enable HF dataset streaming mode.")
    eval_stream_group.add_argument(
        "--stratified",
        action="store_true",
        help="Enable stratified sampling (non-streaming only; uses --max-samples as subset size).",
    )
    eval_parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset before evaluation.")
    eval_parser.add_argument("--shuffle-seed", type=int, default=42, help="Shuffle seed.")
    eval_parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=1000,
        help="Streaming shuffle buffer size (used with --stream --shuffle).",
    )
    eval_parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on evaluation samples.")
    eval_parser.add_argument("--top-k", type=int, default=1, help="Top-k predictions (evaluation uses top-1).")
    eval_parser.add_argument("--device", type=int, default=-1, help="Transformers device index (-1 for CPU).")

    return parser


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.command == "train":
        _run_cli_train(args)
    elif args.command == "infer":
        _run_cli_infer(args)
    elif args.command == "predict":
        _run_cli_predict(args)
    elif args.command == "eval":
        _run_cli_eval(args)
    else:  # pragma: no cover - argparse enforces valid commands
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
