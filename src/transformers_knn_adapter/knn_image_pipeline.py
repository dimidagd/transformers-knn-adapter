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
import re
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
from transformers import pipeline as hf_pipeline
from transformers.image_utils import load_image
from transformers.pipelines import ImageClassificationPipeline
from transformers.pipelines.pt_utils import KeyDataset

from .dinov2_arcface import Dinov2ForImageClassificationWithArcFaceLoss

logger = logging.getLogger(__name__)
GRID_SEARCH_ALLOWED_SCORING = {"f1_macro", "precision_macro", "recall_macro"}
DEFAULT_N_NEIGHBORS = 5


class KNNImageClassificationPipeline(ImageClassificationPipeline):
    """Image classification pipeline using transformer embeddings + sklearn KNN."""

    def __init__(
        self,
        *args: Any,
        knn_model_path: str | Path,
        pad_to_square: bool = False,
        skip_channel_information: str | None = None,
        embedding_source: str = "cls_mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.knn_model_path = str(knn_model_path)
        self.pad_to_square = pad_to_square
        self.skip_channel_information = skip_channel_information
        self.embedding_source = self._resolve_embedding_source(embedding_source)
        feature_device = -1
        if isinstance(self.device, torch.device) and self.device.type == "cuda":
            feature_device = 0 if self.device.index is None else int(self.device.index)
        elif isinstance(self.device, int):
            feature_device = self.device
        self.feature_extraction_pipeline: Any = hf_pipeline(
            "image-feature-extraction",
            model=self.model,
            image_processor=self.image_processor,
            framework="pt",
            device=feature_device,
        )
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
        if isinstance(image_value, str):
            return load_image(image_value).convert("RGB")
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

    @staticmethod
    def _pad_image_to_square(image: Image.Image, fill_color: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
        width, height = image.size
        if width == height:
            return image
        side = max(width, height)
        canvas = Image.new("RGB", (side, side), fill_color)
        offset_x = (side - width) // 2
        offset_y = (side - height) // 2
        canvas.paste(image, (offset_x, offset_y))
        return canvas

    @staticmethod
    def _clone_channel_to_rgb(image: Image.Image, channel: str) -> Image.Image:
        channel_to_index = {"R": 0, "G": 1, "B": 2}
        if channel not in channel_to_index:
            raise ValueError("skip_channel_information must be one of: R, G, B.")
        rgb_image = image.convert("RGB")
        source = rgb_image.split()[channel_to_index[channel]]
        return Image.merge("RGB", (source, source, source))

    @staticmethod
    def _apply_image_transforms(
        image: Image.Image,
        *,
        pad_to_square: bool,
        skip_channel_information: str | None,
    ) -> Image.Image:
        if skip_channel_information is not None:
            image = KNNImageClassificationPipeline._clone_channel_to_rgb(image, skip_channel_information)
        if pad_to_square:
            image = KNNImageClassificationPipeline._pad_image_to_square(image)
        return image

    @staticmethod
    def _prepare_image_static(
        image_value: Any,
        *,
        pad_to_square: bool,
        skip_channel_information: str | None,
    ) -> Image.Image:
        image = KNNImageClassificationPipeline._coerce_image(image_value)
        return KNNImageClassificationPipeline._apply_image_transforms(
            image,
            pad_to_square=pad_to_square,
            skip_channel_information=skip_channel_information,
        )

    def _prepare_image(
        self,
        image_value: Any,
        *,
        pad_to_square: bool,
        skip_channel_information: str | None,
    ) -> Image.Image:
        return self._prepare_image_static(
            image_value,
            pad_to_square=pad_to_square,
            skip_channel_information=skip_channel_information,
        )

    def _resolve_pad_to_square(self, pad_to_square: bool | None) -> bool:
        if pad_to_square is None:
            return self.pad_to_square
        return bool(pad_to_square)

    def _resolve_skip_channel_information(self, skip_channel_information: str | None) -> str | None:
        if skip_channel_information is None:
            return self.skip_channel_information
        if skip_channel_information not in {"R", "G", "B"}:
            raise ValueError("skip_channel_information must be one of: R, G, B.")
        return skip_channel_information

    def _resolve_embedding_source(self, embedding_source: str | None) -> str:
        return Dinov2ForImageClassificationWithArcFaceLoss.resolve_embedding_source(
            embedding_source,
            default=cast(str, getattr(self, "embedding_source", "cls_mean")),
        )

    def _sanitize_parameters(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        custom_pad = kwargs.pop("pad_to_square", None)
        custom_skip_channel = kwargs.pop("skip_channel_information", None)
        custom_embedding_source = kwargs.pop("embedding_source", None)
        preprocess_params, forward_params, postprocess_params = super()._sanitize_parameters(**kwargs)
        if custom_pad is not None:
            preprocess_params["pad_to_square"] = bool(custom_pad)
        if custom_skip_channel is not None:
            preprocess_params["skip_channel_information"] = str(custom_skip_channel)
        if custom_embedding_source is not None:
            forward_params["embedding_source"] = str(custom_embedding_source)
        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, image: Any, **preprocess_params: Any) -> dict[str, Any]:
        pad_to_square = bool(preprocess_params.pop("pad_to_square", self.pad_to_square))
        skip_channel_information = self._resolve_skip_channel_information(
            preprocess_params.pop("skip_channel_information", None)
        )
        prepared_image = self._prepare_image(
            image,
            pad_to_square=pad_to_square,
            skip_channel_information=skip_channel_information,
        )
        return {"prepared_image": prepared_image}

    def _resolve_dataset(
        self,
        dataset: Any,
        split: str,
        streaming: bool = False,
        *,
        pre_shuffle: bool = False,
        shuffle_seed: int = 42,
    ) -> HFDataset | IterableDataset:
        if streaming:
            return self._resolve_dataset_no_preshuffle(dataset=dataset, split=split, streaming=True)
        return self._resolve_dataset_with_optional_preshuffle(
            dataset=dataset,
            split=split,
            streaming=False,
            pre_shuffle=pre_shuffle,
            shuffle_seed=shuffle_seed,
        )

    @staticmethod
    def _parse_split_slice(split: str) -> tuple[str, str] | None:
        match = re.match(r"^\s*([^\[\]]+)\[([^\[\]]*)\]\s*$", split)
        if match is None:
            return None
        return match.group(1).strip(), match.group(2).strip()

    @staticmethod
    def _slice_bound_to_index(bound: str, *, size: int, is_start: bool) -> int:
        if bound == "":
            return 0 if is_start else size
        if bound.endswith("%"):
            percent = float(bound[:-1])
            if not 0.0 <= percent <= 100.0:
                raise ValueError(f"Split percent bound out of range: {bound}")
            return int(math.floor((percent / 100.0) * size))
        return int(bound)

    def _apply_slice_spec(self, dataset_obj: HFDataset, *, split_spec: str) -> HFDataset:
        if ":" not in split_spec:
            raise ValueError(
                "pre_shuffle currently supports slice expressions with ':' "
                "(example: train[:80%], train[80%:], train[10%:90%])."
            )
        start_raw, end_raw = split_spec.split(":", 1)
        total = len(dataset_obj)
        start = self._slice_bound_to_index(start_raw.strip(), size=total, is_start=True)
        end = self._slice_bound_to_index(end_raw.strip(), size=total, is_start=False)
        start = max(0, min(start, total))
        end = max(0, min(end, total))
        if start > end:
            raise ValueError(f"Invalid split slice after pre_shuffle: start ({start}) > end ({end}).")
        return dataset_obj.select(range(start, end))

    def _resolve_dataset_no_preshuffle(self, dataset: Any, split: str, streaming: bool) -> HFDataset | IterableDataset:
        if isinstance(dataset, (str, os.PathLike)):
            dataset_ref = os.fspath(dataset)
            try:
                from datasets import load_dataset
            except Exception as exc:  # pragma: no cover - dependency/runtime detail
                raise ImportError("`datasets` package is required when dataset is a string name.") from exc
            dataset_path = Path(dataset_ref).expanduser()
            if dataset_path.is_dir():
                dataset_obj = load_dataset("imagefolder", data_dir=str(dataset_path), split=split, streaming=streaming)
            else:
                dataset_obj = load_dataset(dataset_ref, split=split, streaming=streaming)
            if not isinstance(dataset_obj, (HFDataset, IterableDataset)):
                raise TypeError("Only Hugging Face Dataset/IterableDataset is supported.")
            return dataset_obj

        dataset_obj = dataset[split] if hasattr(dataset, "keys") and split in dataset else dataset
        if not isinstance(dataset_obj, (HFDataset, IterableDataset)):
            raise TypeError("Only Hugging Face Dataset/IterableDataset is supported.")
        return dataset_obj

    def _resolve_dataset_with_optional_preshuffle(
        self,
        dataset: Any,
        split: str,
        streaming: bool,
        *,
        pre_shuffle: bool = False,
        shuffle_seed: int = 42,
    ) -> HFDataset | IterableDataset:
        if not pre_shuffle:
            return self._resolve_dataset_no_preshuffle(dataset=dataset, split=split, streaming=streaming)

        parsed = self._parse_split_slice(split)
        if parsed is None:
            return self._resolve_dataset_no_preshuffle(dataset=dataset, split=split, streaming=streaming)
        base_split, split_spec = parsed

        if isinstance(dataset, (str, os.PathLike)):
            dataset_ref = os.fspath(dataset)
            try:
                from datasets import load_dataset
            except Exception as exc:  # pragma: no cover - dependency/runtime detail
                raise ImportError("`datasets` package is required when dataset is a string name.") from exc
            dataset_path = Path(dataset_ref).expanduser()
            if dataset_path.is_dir():
                dataset_obj = load_dataset("imagefolder", data_dir=str(dataset_path), split=base_split, streaming=False)
            else:
                dataset_obj = load_dataset(dataset_ref, split=base_split, streaming=False)
        else:
            if hasattr(dataset, "keys") and base_split in dataset:
                dataset_obj = dataset[base_split]
            else:
                raise ValueError(
                    f"pre_shuffle with split slice requires base split '{base_split}' to exist in the provided dataset."
                )

        if not isinstance(dataset_obj, HFDataset):
            raise TypeError("pre_shuffle with split slicing requires a non-streaming Hugging Face Dataset.")

        shuffled = dataset_obj.shuffle(seed=shuffle_seed)
        sliced = self._apply_slice_spec(shuffled, split_spec=split_spec)
        logger.info("Applied deterministic pre-slice shuffle: split=%s seed=%d", split, shuffle_seed)
        return sliced

    @staticmethod
    def _normalize_label(raw_label: Any, label_names: list[str] | None) -> Any:
        if label_names is not None and isinstance(raw_label, (int, np.integer)) and 0 <= int(raw_label) < len(label_names):
            return label_names[int(raw_label)]
        return raw_label

    def _extract_string_labels(
        self,
        *,
        dataset_obj: HFDataset,
        label_column: str,
        label_names: list[str] | None,
    ) -> list[str]:
        raw_labels = cast(list[Any], dataset_obj[label_column])
        return [str(self._normalize_label(raw_label, label_names)) for raw_label in raw_labels]

    def _apply_min_class_filter(
        self,
        *,
        dataset_obj: HFDataset,
        labels: list[str],
        min_class_instances: int,
    ) -> tuple[HFDataset, list[str]]:
        class_counts = Counter(labels)
        keep_labels = {label for label, count in class_counts.items() if count >= min_class_instances}
        if not keep_labels:
            raise ValueError(
                f"No classes meet min_class_instances={min_class_instances}. "
                "Lower threshold or disable this filter."
            )
        if len(keep_labels) == len(class_counts):
            return dataset_obj, labels

        keep_indices = [idx for idx, label in enumerate(labels) if label in keep_labels]
        dropped = len(labels) - len(keep_indices)
        filtered_dataset = dataset_obj.select(keep_indices)
        filtered_labels = [labels[idx] for idx in keep_indices]
        logger.info(
            "Dropped %d eval samples from classes below min_class_instances=%d",
            dropped,
            min_class_instances,
        )
        return filtered_dataset, filtered_labels

    def _apply_positive_population_ratio(
        self,
        *,
        dataset_obj: HFDataset,
        labels: list[str],
        negative_classes: list[str],
        positive_classes_population_ratio: float,
        shuffle_seed: int,
    ) -> tuple[HFDataset, list[str]]:
        negative_label_set = {str(label) for label in negative_classes}
        positive_indices = [idx for idx, label in enumerate(labels) if label not in negative_label_set]
        negative_indices = [idx for idx, label in enumerate(labels) if label in negative_label_set]
        positive_count = len(positive_indices)
        negative_count = len(negative_indices)
        if positive_count == 0 or negative_count == 0:
            logger.warning(
                "Requested positive_classes_population_ratio=%.6f cannot be adjusted because only one group "
                "(positive or negative) is present.",
                positive_classes_population_ratio,
            )
            return dataset_obj, labels

        rng = np.random.default_rng(shuffle_seed)
        target_ratio = positive_classes_population_ratio
        if target_ratio == 1.0:
            keep_indices = positive_indices
        elif target_ratio == 0.0:
            keep_indices = negative_indices
        else:
            current_ratio = positive_count / (positive_count + negative_count)
            if current_ratio > target_ratio:
                target_positive_count = int(math.floor((target_ratio / (1.0 - target_ratio)) * negative_count))
                kept_positive = (
                    set(int(i) for i in rng.choice(positive_indices, size=target_positive_count, replace=False))
                    if target_positive_count > 0
                    else set()
                )
                keep_indices = [idx for idx in range(len(labels)) if idx in kept_positive or idx in negative_indices]
            else:
                target_negative_count = int(math.floor(((1.0 - target_ratio) / target_ratio) * positive_count))
                kept_negative = (
                    set(int(i) for i in rng.choice(negative_indices, size=target_negative_count, replace=False))
                    if target_negative_count > 0
                    else set()
                )
                keep_indices = [idx for idx in range(len(labels)) if idx in positive_indices or idx in kept_negative]

        subsampled_dataset = dataset_obj.select(keep_indices)
        subsampled_labels = [labels[idx] for idx in keep_indices]
        positive_after = sum(1 for label in subsampled_labels if label not in negative_label_set)
        total_after = len(subsampled_labels)
        if total_after > 0:
            achieved_ratio = positive_after / total_after
            if not math.isclose(achieved_ratio, target_ratio, rel_tol=0.0, abs_tol=1e-12):
                logger.warning(
                    "Requested positive_classes_population_ratio=%.6f but achieved ratio=%.6f "
                    "(positive_samples=%d total_samples=%d)",
                    target_ratio,
                    achieved_ratio,
                    positive_after,
                    total_after,
                )
        return subsampled_dataset, subsampled_labels

    def _apply_eval_class_controls(
        self,
        *,
        dataset_obj: HFDataset,
        label_column: str,
        label_names: list[str] | None,
        min_class_instances: int | None,
        negative_classes: list[str],
        positive_classes_population_ratio: float | None,
        shuffle_seed: int,
    ) -> HFDataset:
        labels = self._extract_string_labels(dataset_obj=dataset_obj, label_column=label_column, label_names=label_names)
        if min_class_instances is not None:
            dataset_obj, labels = self._apply_min_class_filter(
                dataset_obj=dataset_obj,
                labels=labels,
                min_class_instances=min_class_instances,
            )
        if positive_classes_population_ratio is not None:
            dataset_obj, _ = self._apply_positive_population_ratio(
                dataset_obj=dataset_obj,
                labels=labels,
                negative_classes=negative_classes,
                positive_classes_population_ratio=positive_classes_population_ratio,
                shuffle_seed=shuffle_seed,
            )
        return dataset_obj

    def _extract_embedding_from_feature_output(
        self,
        feature_output: Any,
        *,
        embedding_source: str | None = None,
    ) -> np.ndarray:
        resolved_embedding_source = self._resolve_embedding_source(embedding_source)
        features_np = np.asarray(feature_output, dtype=np.float32)
        while features_np.ndim > 2 and features_np.shape[0] == 1:
            features_np = features_np[0]
        if features_np.ndim == 2:
            batched_features = features_np[np.newaxis, :, :]
            embeddings = Dinov2ForImageClassificationWithArcFaceLoss.calculate_embeddings_from_numpy(
                batched_features,
                embedding_source=resolved_embedding_source,
            )
            return embeddings[0]
        if features_np.ndim == 1:
            return features_np
        raise ValueError(f"Unexpected feature output shape: {features_np.shape}")

    @staticmethod
    def _save_debug_transformed_samples(
        *,
        images: list[Any],
        output_dir: str | Path,
        prefix: str,
    ) -> int:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        saved = 0
        for idx, image_value in enumerate(images):
            image = KNNImageClassificationPipeline._coerce_image(image_value)
            image.save(output_path / f"{prefix}_{idx:03d}.png")
            saved += 1
        return saved

    def _extract_embeddings_from_images(
        self,
        images: list[Image.Image],
        *,
        embedding_source: str | None = None,
    ) -> np.ndarray:
        resolved_embedding_source = self._resolve_embedding_source(embedding_source)
        raw_features = self.feature_extraction_pipeline(images, batch_size=len(images))
        features_np = np.asarray(raw_features, dtype=np.float32)
        # Some HF backends return an extra singleton axis: (B, 1, T, D).
        if features_np.ndim == 4 and features_np.shape[1] == 1:
            features_np = features_np[:, 0, :, :]
        if features_np.ndim == 3:
            return Dinov2ForImageClassificationWithArcFaceLoss.calculate_embeddings_from_numpy(
                features_np,
                embedding_source=resolved_embedding_source,
            )
        if features_np.ndim == 2:
            if len(images) == 1:
                return features_np
            return features_np
        raise ValueError(
            "Expected 2D/3D features (or 4D with singleton axis) from extraction pipeline, "
            f"got shape {features_np.shape}"
        )

    def _materialize_training_features_and_labels(
        self,
        *,
        dataset_obj: HFDataset | IterableDataset,
        image_column: str,
        label_column: str,
        batch_size: int,
        num_workers: int,
        max_samples: int | None,
        label_names: list[str] | None,
        pad_to_square: bool,
        skip_channel_information: str | None,
        debug_save_transformed_samples_dir: str | Path | None = None,
        debug_save_transformed_samples_count: int = 0,
        embedding_source: str = "cls_mean",
    ) -> tuple[np.ndarray, np.ndarray, np.memmap | None, str | None, int]:
        """Extract embeddings and labels, optionally using memmap storage for known sample counts."""
        self.model.eval()
        resolved_embedding_source = self._resolve_embedding_source(embedding_source)
        embedding_batches: list[np.ndarray] = []
        embeddings_mm: np.memmap | None = None
        memmap_path: str | None = None
        write_idx = 0
        labels: list[Any] = []
        loaded = 0
        if isinstance(dataset_obj, HFDataset):
            if max_samples is not None:
                dataset_obj = dataset_obj.select(range(min(max_samples, len(dataset_obj))))
            labels = [
                self._normalize_label(raw_label, label_names) for raw_label in cast(list[Any], dataset_obj[label_column])
            ]
            total_samples: int | None = len(dataset_obj)

            def _transform_row(row: dict[str, Any]) -> dict[str, Any]:
                image_value = row[image_column]
                if isinstance(image_value, list):
                    transformed_images = [
                        self._prepare_image(
                            img,
                            pad_to_square=pad_to_square,
                            skip_channel_information=skip_channel_information,
                        )
                        for img in image_value
                    ]
                    row[image_column] = transformed_images
                    return row
                row[image_column] = self._prepare_image(
                    image_value,
                    pad_to_square=pad_to_square,
                    skip_channel_information=skip_channel_information,
                )
                return row

            transformed_dataset = dataset_obj.with_transform(_transform_row)
            if debug_save_transformed_samples_dir is not None and debug_save_transformed_samples_count > 0:
                sample_count = min(debug_save_transformed_samples_count, len(transformed_dataset))
                transformed_images = [transformed_dataset[idx][image_column] for idx in range(sample_count)]
                saved = self._save_debug_transformed_samples(
                    images=transformed_images,
                    output_dir=debug_save_transformed_samples_dir,
                    prefix="train",
                )
                logger.info("Saved %d transformed train debug samples to %s", saved, debug_save_transformed_samples_dir)
            feature_iter = self.feature_extraction_pipeline(
                KeyDataset(transformed_dataset, image_column),
                batch_size=batch_size,
                num_workers=num_workers,
            )
        else:
            dataset_for_iter = dataset_obj.take(max_samples) if max_samples is not None else dataset_obj
            total_samples = max_samples
            if debug_save_transformed_samples_dir is not None and debug_save_transformed_samples_count > 0:
                logger.warning(
                    "debug_save_transformed_samples_* for train is currently supported only for non-streaming HFDataset."
                )

            def image_iter() -> Any:
                for row in dataset_for_iter:
                    labels.append(self._normalize_label(row[label_column], label_names))
                    image = self._prepare_image(
                        row[image_column],
                        pad_to_square=pad_to_square,
                        skip_channel_information=skip_channel_information,
                    )
                    yield image

            feature_iter = self.feature_extraction_pipeline(
                image_iter(),
                batch_size=batch_size,
                num_workers=num_workers,
            )

        for feature_output in tqdm(feature_iter, total=total_samples, desc="Extracting train embeddings"):
            embedding = self._extract_embedding_from_feature_output(
                feature_output,
                embedding_source=resolved_embedding_source,
            ).reshape(1, -1)
            loaded += 1
            if total_samples is not None:
                if embeddings_mm is None:
                    with tempfile.NamedTemporaryFile(prefix="knn_emb_", suffix=".mmap", delete=False) as f:
                        memmap_path = f.name
                    embeddings_mm = np.memmap(
                        memmap_path,
                        dtype=embedding.dtype,
                        mode="w+",
                        shape=(int(total_samples), embedding.shape[1]),
                    )
                    logger.info("Using memory-mapped embedding storage at %s", memmap_path)
                embeddings_mm[write_idx] = embedding[0]
                write_idx += 1
            else:
                embedding_batches.append(embedding)

        if loaded == 0:
            raise ValueError("Dataset is empty; cannot train KNN.")
        logger.info("Loaded %d items for training", loaded)

        if loaded != len(labels):
            raise ValueError(
                f"Mismatch between extracted embeddings ({loaded}) and labels ({len(labels)})."
            )

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

    @staticmethod
    def _compute_reid_metrics_from_rank_positions(
        rank_positions: list[int],
        *,
        cmc_ranks: tuple[int, ...],
    ) -> dict[str, Any]:
        if not rank_positions:
            raise ValueError("rank_positions must not be empty for re-identification metrics.")
        if not cmc_ranks:
            raise ValueError("cmc_ranks must not be empty for re-identification metrics.")
        if any(rank <= 0 for rank in cmc_ranks):
            raise ValueError("All cmc_ranks must be positive integers.")
        cmc_ranks_sorted = tuple(sorted(set(int(rank) for rank in cmc_ranks)))
        total = len(rank_positions)
        # For closed-set class ranking there is exactly one relevant identity/class per query.
        # AP for each query is therefore reciprocal rank.
        map_score = float(np.mean([1.0 / float(rank) for rank in rank_positions]))
        cmc = {
            f"cmc@{rank}": float(sum(1 for pos in rank_positions if pos <= rank) / total)
            for rank in cmc_ranks_sorted
        }
        return {
            "queries": total,
            "mAP": map_score,
            "cmc": cmc,
            "mean_rank": float(np.mean(rank_positions)),
        }

    def train(
        self,
        dataset: Any,
        *,
        split: str = "train",
        image_column: str = "image",
        label_column: str = "label",
        batch_size: int = 16,
        num_workers: int = 1,
        streaming: bool = False,
        stratified: bool = False,
        pre_shuffle: bool = False,
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
        pad_to_square: bool | None = None,
        skip_channel_information: str | None = None,
        embedding_source: str | None = None,
        debug_save_transformed_samples_dir: str | Path | None = None,
        debug_save_transformed_samples_count: int = 0,
        save_knn_model_path: str | Path | None = None,
    ) -> ClassifierMixin:
        """Train and attach a KNN head from extracted embeddings.

        `dataset` can be:
        - a Hugging Face dataset name (e.g. ``"timm/mini-imagenet"``), or
        - a local directory path for `imagefolder` data, or
        - a loaded `datasets.Dataset` / `datasets.DatasetDict`, or
        - a loaded `datasets.IterableDataset` / `datasets.IterableDatasetDict`.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        if shuffle_buffer_size <= 0:
            raise ValueError("shuffle_buffer_size must be > 0")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be > 0")
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be > 0")
        if debug_save_transformed_samples_count < 0:
            raise ValueError("debug_save_transformed_samples_count must be >= 0")
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
        if stratified and max_samples is None:
            raise ValueError("stratified mode requires max_samples to be set.")
        if streaming and pre_shuffle:
            raise ValueError("pre_shuffle is only supported when streaming=False.")
        resolved_pad_to_square = self._resolve_pad_to_square(pad_to_square)
        resolved_skip_channel_information = self._resolve_skip_channel_information(skip_channel_information)
        resolved_embedding_source = self._resolve_embedding_source(embedding_source)

        logger.info(
            "Starting KNN training: split=%s image_column=%s label_column=%s batch_size=%d n_neighbors=%d streaming=%s stratified=%s pre_shuffle=%s shuffle=%s shuffle_seed=%d max_samples=%s",
            split,
            image_column,
            label_column,
            batch_size,
            n_neighbors,
            streaming,
            stratified,
            pre_shuffle,
            shuffle,
            shuffle_seed,
            max_samples,
        )
        dataset_obj = self._resolve_dataset(
            dataset=dataset,
            split=split,
            streaming=streaming,
            pre_shuffle=pre_shuffle,
            shuffle_seed=shuffle_seed,
        )

        if stratified:
            if not isinstance(dataset_obj, HFDataset):
                raise ValueError("stratified mode requires a non-streaming Hugging Face Dataset.")
            dataset_obj = dataset_obj.train_test_split(
                train_size=max_samples,
                stratify_by_column=label_column,
                seed=shuffle_seed,
            )["train"]
            max_samples = None
            logger.info("Applied stratified sampling for training subset")
            
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
            num_workers=num_workers,
            max_samples=max_samples,
            label_names=label_names,
            pad_to_square=resolved_pad_to_square,
            skip_channel_information=resolved_skip_channel_information,
            debug_save_transformed_samples_dir=debug_save_transformed_samples_dir,
            debug_save_transformed_samples_count=debug_save_transformed_samples_count,
            embedding_source=resolved_embedding_source,
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
        num_workers: int = 1,
        streaming: bool = False,
        stratified: bool = False,
        pre_shuffle: bool = False,
        shuffle: bool = False,
        shuffle_seed: int = 42,
        shuffle_buffer_size: int = 1000,
        max_samples: int | None = None,
        min_class_instances: int | None = None,
        negative_classes: tuple[str, ...] = ("other",),
        positive_classes_population_ratio: float | None = None,
        pad_to_square: bool | None = None,
        skip_channel_information: str | None = None,
        embedding_source: str | None = None,
        debug_save_transformed_samples_dir: str | Path | None = None,
        debug_save_transformed_samples_count: int = 0,
        reid_cmc_ranks: tuple[int, ...] = (1, 5, 10),
    ) -> dict[str, Any]:
        """Evaluate top-1 accuracy on a dataset or dataset split."""
        if self.knn_model is None:
            raise ValueError("KNN head is not loaded. Provide knn_model_path or call train(...) first.")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        if shuffle_buffer_size <= 0:
            raise ValueError("shuffle_buffer_size must be > 0")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be > 0")
        if min_class_instances is not None and min_class_instances <= 0:
            raise ValueError("min_class_instances must be > 0")
        if debug_save_transformed_samples_count < 0:
            raise ValueError("debug_save_transformed_samples_count must be >= 0")
        if not reid_cmc_ranks:
            raise ValueError("reid_cmc_ranks must not be empty.")
        if any(int(rank) <= 0 for rank in reid_cmc_ranks):
            raise ValueError("All reid_cmc_ranks must be positive integers.")
        if positive_classes_population_ratio is not None and not (0.0 <= positive_classes_population_ratio <= 1.0):
            raise ValueError("positive_classes_population_ratio must be in [0.0, 1.0]")
        if streaming and stratified:
            raise ValueError("stratified mode is only supported when streaming=False.")
        if stratified and max_samples is None:
            raise ValueError("stratified mode requires max_samples to be set.")
        if streaming and pre_shuffle:
            raise ValueError("pre_shuffle is only supported when streaming=False.")
        if streaming and (min_class_instances is not None or positive_classes_population_ratio is not None):
            raise ValueError(
                "min_class_instances and positive_classes_population_ratio require streaming=False "
                "(class-aware filtering/subsampling needs materialized datasets)."
            )
        resolved_pad_to_square = self._resolve_pad_to_square(pad_to_square)
        resolved_skip_channel_information = self._resolve_skip_channel_information(skip_channel_information)
        resolved_embedding_source = self._resolve_embedding_source(embedding_source)
        dataset_obj = self._resolve_dataset(
            dataset=dataset,
            split=split,
            streaming=streaming,
            pre_shuffle=pre_shuffle,
            shuffle_seed=shuffle_seed,
        )
        if stratified:
            if not isinstance(dataset_obj, HFDataset):
                raise ValueError("stratified mode requires a non-streaming Hugging Face Dataset.")
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

        if not isinstance(dataset_obj, IterableDataset) and (
            min_class_instances is not None or positive_classes_population_ratio is not None
        ):
            dataset_obj = self._apply_eval_class_controls(
                dataset_obj=dataset_obj,
                label_column=label_column,
                label_names=label_names,
                min_class_instances=min_class_instances,
                negative_classes=list(negative_classes),
                positive_classes_population_ratio=positive_classes_population_ratio,
                shuffle_seed=shuffle_seed,
            )
        rank_positions: list[int] = []
        eval_top_k = len(self.knn_model.classes_)
        rows: list[tuple[str, str, bool]] = []
        if isinstance(dataset_obj, HFDataset):
            if max_samples is not None:
                dataset_obj = dataset_obj.select(range(min(max_samples, len(dataset_obj))))
            if debug_save_transformed_samples_dir is not None and debug_save_transformed_samples_count > 0:
                sample_count = min(debug_save_transformed_samples_count, len(dataset_obj))
                raw_sample_images = cast(list[Any], dataset_obj[image_column][:sample_count])
                transformed_samples = [
                    self._prepare_image(
                        image_value,
                        pad_to_square=resolved_pad_to_square,
                        skip_channel_information=resolved_skip_channel_information,
                    )
                    for image_value in raw_sample_images
                ]
                saved = self._save_debug_transformed_samples(
                    images=transformed_samples,
                    output_dir=debug_save_transformed_samples_dir,
                    prefix="eval",
                )
                logger.info("Saved %d transformed eval debug samples to %s", saved, debug_save_transformed_samples_dir)
            y_true = [
                str(self._normalize_label(raw_label, label_names)) for raw_label in cast(list[Any], dataset_obj[label_column])
            ]
            image_inputs: Any = KeyDataset(dataset_obj, image_column)
            pred_iter = self(
                image_inputs,
                batch_size=batch_size,
                num_workers=num_workers,
                top_k=eval_top_k,
                pad_to_square=resolved_pad_to_square,
                skip_channel_information=resolved_skip_channel_information,
            )
            for true_label, pred in tqdm(zip(y_true, pred_iter, strict=True), total=len(y_true), desc="Evaluating"):
                pred_list = pred if isinstance(pred, list) else [pred]
                pred_label = str(pred_list[0]["label"])
                rows.append((true_label, pred_label, true_label == pred_label))
                rank = next(
                    (idx + 1 for idx, item in enumerate(pred_list) if str(item["label"]) == true_label),
                    len(pred_list) + 1,
                )
                rank_positions.append(rank)
        else:
            dataset_for_eval = dataset_obj.take(max_samples) if max_samples is not None else dataset_obj
            true_labels: list[str] = []
            debug_saved = 0

            def image_iter() -> Any:
                nonlocal debug_saved
                for row in dataset_for_eval:
                    true_labels.append(str(self._normalize_label(row[label_column], label_names)))
                    if (
                        debug_save_transformed_samples_dir is not None
                        and debug_save_transformed_samples_count > 0
                        and debug_saved < debug_save_transformed_samples_count
                    ):
                        transformed = self._prepare_image(
                            row[image_column],
                            pad_to_square=resolved_pad_to_square,
                            skip_channel_information=resolved_skip_channel_information,
                        )
                        self._save_debug_transformed_samples(
                            images=[transformed],
                            output_dir=debug_save_transformed_samples_dir,
                            prefix=f"eval_{debug_saved:03d}",
                        )
                        debug_saved += 1
                    yield row[image_column]

            pred_iter = self(
                image_iter(),
                batch_size=batch_size,
                num_workers=num_workers,
                top_k=eval_top_k,
                pad_to_square=resolved_pad_to_square,
                skip_channel_information=resolved_skip_channel_information,
                embedding_source=resolved_embedding_source,
            )
            total = max_samples
            for idx, pred in enumerate(tqdm(pred_iter, total=total, desc="Evaluating")):
                true_label = true_labels[idx]
                pred_list = pred if isinstance(pred, list) else [pred]
                pred_label = str(pred_list[0]["label"])
                rows.append((true_label, pred_label, true_label == pred_label))
                rank = next(
                    (rank_idx + 1 for rank_idx, item in enumerate(pred_list) if str(item["label"]) == true_label),
                    len(pred_list) + 1,
                )
                rank_positions.append(rank)

        if not rows:
            raise ValueError("Dataset is empty; cannot evaluate.")

        correct = sum(int(r[2]) for r in rows)
        total = len(rows)
        accuracy = correct / total
        y_true = [r[0] for r in rows]
        y_pred = [r[1] for r in rows]
        # Restrict report classes to labels present in the evaluation ground truth.
        eval_labels = list(dict.fromkeys(y_true))
        report_text = classification_report(y_true, y_pred, labels=eval_labels, zero_division=0)
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
        reid = self._compute_reid_metrics_from_rank_positions(rank_positions, cmc_ranks=reid_cmc_ranks)
        metrics["reid_metrics"] = reid
        logger.info("Re-identification metrics: %s", reid)
        logger.info("Evaluation complete: split=%s samples=%d top1_accuracy=%.4f", split, total, accuracy)
        logger.info("Classification report:\n%s", report_text)
        return metrics

    def _extract_embeddings(self, model_outputs: Any, *, embedding_source: str | None = None) -> torch.Tensor:
        """Extract a 2D embedding tensor from transformer model outputs."""
        resolved_embedding_source = self._resolve_embedding_source(embedding_source)
        if getattr(model_outputs, "last_hidden_state", None) is not None:
            sequence_output = cast(torch.Tensor, model_outputs.last_hidden_state)
            embeddings = Dinov2ForImageClassificationWithArcFaceLoss.calculate_embeddings(
                sequence_output,
                embedding_source=resolved_embedding_source,
            )
        elif getattr(model_outputs, "pooler_output", None) is not None and resolved_embedding_source == "cls":
            embeddings = cast(torch.Tensor, model_outputs.pooler_output)
        else:
            raise ValueError("Model outputs do not contain hidden states suitable for embedding extraction.")
        embeddings = embeddings.flatten(start_dim=1)
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {tuple(embeddings.shape)}")
        return embeddings

    def _forward(self, model_inputs: dict[str, Any], **forward_params: Any) -> Any:
        if self.knn_model is None:
            raise ValueError("KNN head is not loaded. Provide knn_model_path or call train(...) first.")
        resolved_embedding_source = self._resolve_embedding_source(forward_params.pop("embedding_source", None))
        del forward_params
        prepared_images = model_inputs["prepared_image"]
        if isinstance(prepared_images, list):
            images = cast(list[Image.Image], prepared_images)
        else:
            images = [cast(Image.Image, prepared_images)]
        embeddings_np = self._extract_embeddings_from_images(images, embedding_source=resolved_embedding_source)
        probs_np = self.knn_model.predict_proba(embeddings_np)
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


def _resolve_cli_image_options(args: argparse.Namespace) -> tuple[bool, str | None]:
    pad_to_square = bool(getattr(args, "pad_to_square", False))
    skip_channel_information = cast(str | None, getattr(args, "skip_channel_information", None))
    return pad_to_square, skip_channel_information


def _build_pipeline_from_args(args: argparse.Namespace) -> tuple[KNNImageClassificationPipeline, bool, str | None, str]:
    pad_to_square, skip_channel_information = _resolve_cli_image_options(args)
    embedding_source = cast(str, getattr(args, "embedding_source", "cls_mean"))
    clf = pipeline(
        "image-classification",
        model_path=args.model,
        knn_model_path=args.knn_model_path,
        device=args.device,
        top_k=args.top_k,
        pad_to_square=pad_to_square,
        skip_channel_information=skip_channel_information,
        embedding_source=embedding_source,
    )
    return clf, pad_to_square, skip_channel_information, embedding_source


def _train_pipeline_from_args(args: argparse.Namespace) -> KNNImageClassificationPipeline:
    """Build and train a pipeline instance from parsed CLI arguments."""
    _validate_cli_train_args(args)
    clf, pad_to_square, skip_channel_information, embedding_source = _build_pipeline_from_args(args)

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
        num_workers=args.num_workers,
        streaming=args.stream,
        stratified=args.stratified,
        pre_shuffle=args.pre_shuffle,
        shuffle=args.shuffle,
        shuffle_seed=args.shuffle_seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
        max_samples=args.max_samples,
        n_neighbors=args.n_neighbors if args.n_neighbors is not None else DEFAULT_N_NEIGHBORS,
        grid_search=args.grid_search,
        grid_search_splits=args.grid_search_splits if args.grid_search_splits is not None else 3,
        grid_search_repeats=args.grid_search_repeats if args.grid_search_repeats is not None else 2,
        grid_search_scoring=args.grid_search_scoring,
        pad_to_square=pad_to_square,
        skip_channel_information=skip_channel_information,
        embedding_source=embedding_source,
        debug_save_transformed_samples_dir=args.debug_save_transformed_samples_dir,
        debug_save_transformed_samples_count=args.debug_save_transformed_samples_count,
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
    clf, pad_to_square, skip_channel_information, embedding_source = _build_pipeline_from_args(args)
    image_input = args.image
    image_batch = [image_input for _ in range(args.inference_batch_size)]
    single_result = clf(
        image_input,
        pad_to_square=pad_to_square,
        skip_channel_information=skip_channel_information,
        embedding_source=embedding_source,
    )
    batch_result = clf(
        image_batch,
        pad_to_square=pad_to_square,
        skip_channel_information=skip_channel_information,
        embedding_source=embedding_source,
    )
    logger.info("Single-image inference input: %s", image_input)
    logger.info("Single-image inference result: %s", single_result)
    logger.info("Batch inference URLs (count=%d): %s", args.inference_batch_size, image_batch)
    logger.info("Batch inference result: %s", batch_result)


def _run_cli_predict(args: argparse.Namespace) -> None:
    """Handle the `predict` CLI command."""
    clf, pad_to_square, skip_channel_information, embedding_source = _build_pipeline_from_args(args)
    result = clf(
        args.image,
        pad_to_square=pad_to_square,
        skip_channel_information=skip_channel_information,
        embedding_source=embedding_source,
    )
    logger.info("Prediction result: %s", result)


def _run_cli_eval(args: argparse.Namespace) -> None:
    """Handle the `eval` CLI command."""
    clf, pad_to_square, skip_channel_information, embedding_source = _build_pipeline_from_args(args)
    metrics = clf.evaluate(
        dataset=args.dataset,
        split=args.split,
        image_column=args.image_column,
        label_column=args.label_column,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        streaming=args.stream,
        stratified=args.stratified,
        pre_shuffle=args.pre_shuffle,
        shuffle=args.shuffle,
        shuffle_seed=args.shuffle_seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
        max_samples=args.max_samples,
        min_class_instances=args.min_class_instances,
        negative_classes=tuple(args.negative_classes),
        positive_classes_population_ratio=args.positive_classes_population_ratio,
        pad_to_square=pad_to_square,
        skip_channel_information=skip_channel_information,
        embedding_source=embedding_source,
        debug_save_transformed_samples_dir=args.debug_save_transformed_samples_dir,
        debug_save_transformed_samples_count=args.debug_save_transformed_samples_count,
        reid_cmc_ranks=tuple(args.reid_cmc_ranks),
    )
    logger.info("Eval metrics: %s", metrics)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(description="KNN image pipeline CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train KNN head and save it.")
    train_parser.add_argument("--model", required=True, help="HF model id/path for feature extraction.")
    train_parser.add_argument("--knn-model-path", required=True, help="Path to save/load KNN model (.joblib).")
    train_parser.add_argument(
        "--dataset",
        required=True,
        help="HF dataset name (example: timm/mini-imagenet) or local imagefolder path.",
    )
    train_parser.add_argument("--split", default="train", help="Dataset split when --dataset is provided.")
    train_parser.add_argument("--image-column", default="image", help="Dataset image column.")
    train_parser.add_argument("--label-column", default="label", help="Dataset label column.")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size.")
    train_parser.add_argument("--num-workers", type=int, default=1, help="HF pipeline dataloader workers.")
    stream_group = train_parser.add_mutually_exclusive_group()
    stream_group.add_argument("--stream", action="store_true", help="Enable HF dataset streaming mode.")
    stream_group.add_argument(
        "--stratified",
        action="store_true",
        help="Enable stratified sampling (non-streaming only; uses --max-samples as subset size).",
    )
    train_parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset before sampling/training.")
    train_parser.add_argument(
        "--pre-shuffle",
        action="store_true",
        help=(
            "Shuffle base split deterministically before applying split-slice expressions "
            "(example: train[80%%:]). Uses --shuffle-seed."
        ),
    )
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
    train_parser.add_argument(
        "--embedding-source",
        choices=["cls", "cls_mean"],
        default="cls_mean",
        help="Embedding reduction used for KNN features: CLS only or CLS plus mean patch tokens.",
    )
    train_parser.add_argument("--device", type=int, default=-1, help="Transformers device index (-1 for CPU).")
    train_parser.add_argument(
        "--pad-to-square",
        action="store_true",
        help="Pad images with black pixels to square shape before preprocessing.",
    )
    train_parser.add_argument(
        "--skip-channel-information",
        choices=["R", "G", "B"],
        default=None,
        help="Clone selected channel into all RGB channels before optional square padding.",
    )
    train_parser.add_argument(
        "--debug-save-transformed-samples-dir",
        default=None,
        help="Optional output directory to save a few transformed train images for debugging.",
    )
    train_parser.add_argument(
        "--debug-save-transformed-samples-count",
        type=int,
        default=0,
        help="Number of transformed train images to save to --debug-save-transformed-samples-dir.",
    )

    infer_parser = subparsers.add_parser(
        "infer",
        help="Run inference on one image and a list of images using a trained KNN head.",
    )
    infer_parser.add_argument("--model", required=True, help="HF model id/path for feature extraction.")
    infer_parser.add_argument("--knn-model-path", required=True, help="Path to save/load KNN model (.joblib).")
    infer_parser.add_argument("--top-k", type=int, default=3, help="Top-k predictions.")
    infer_parser.add_argument(
        "--embedding-source",
        choices=["cls", "cls_mean"],
        default="cls_mean",
        help="Embedding reduction used for KNN features: CLS only or CLS plus mean patch tokens.",
    )
    infer_parser.add_argument("--device", type=int, default=-1, help="Transformers device index (-1 for CPU).")
    infer_parser.add_argument(
        "--pad-to-square",
        action="store_true",
        help="Pad images with black pixels to square shape before preprocessing.",
    )
    infer_parser.add_argument(
        "--skip-channel-information",
        choices=["R", "G", "B"],
        default=None,
        help="Clone selected channel into all RGB channels before optional square padding.",
    )
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
    predict_parser.add_argument(
        "--embedding-source",
        choices=["cls", "cls_mean"],
        default="cls_mean",
        help="Embedding reduction used for KNN features: CLS only or CLS plus mean patch tokens.",
    )
    predict_parser.add_argument("--device", type=int, default=-1, help="Transformers device index (-1 for CPU).")
    predict_parser.add_argument(
        "--pad-to-square",
        action="store_true",
        help="Pad images with black pixels to square shape before preprocessing.",
    )
    predict_parser.add_argument(
        "--skip-channel-information",
        choices=["R", "G", "B"],
        default=None,
        help="Clone selected channel into all RGB channels before optional square padding.",
    )

    eval_parser = subparsers.add_parser("eval", help="Evaluate trained KNN head on a dataset split.")
    eval_parser.add_argument("--model", required=True, help="HF model id/path for feature extraction.")
    eval_parser.add_argument("--knn-model-path", required=True, help="Path to trained KNN model (.joblib).")
    eval_parser.add_argument(
        "--dataset",
        required=True,
        help="HF dataset name (example: timm/mini-imagenet) or local imagefolder path.",
    )
    eval_parser.add_argument("--split", default="validation", help="Dataset split.")
    eval_parser.add_argument("--image-column", default="image", help="Dataset image column.")
    eval_parser.add_argument("--label-column", default="label", help="Dataset label column.")
    eval_parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size.")
    eval_parser.add_argument("--num-workers", type=int, default=1, help="HF pipeline dataloader workers.")
    eval_stream_group = eval_parser.add_mutually_exclusive_group()
    eval_stream_group.add_argument("--stream", action="store_true", help="Enable HF dataset streaming mode.")
    eval_stream_group.add_argument(
        "--stratified",
        action="store_true",
        help="Enable stratified sampling (non-streaming only; uses --max-samples as subset size).",
    )
    eval_parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset before evaluation.")
    eval_parser.add_argument(
        "--pre-shuffle",
        action="store_true",
        help=(
            "Shuffle base split deterministically before applying split-slice expressions "
            "(example: train[80%%:]). Uses --shuffle-seed."
        ),
    )
    eval_parser.add_argument("--shuffle-seed", type=int, default=42, help="Shuffle seed.")
    eval_parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=1000,
        help="Streaming shuffle buffer size (used with --stream --shuffle).",
    )
    eval_parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on evaluation samples.")
    eval_parser.add_argument(
        "--min-class-instances",
        type=int,
        default=None,
        help="Drop classes with fewer than this number of eval instances (non-streaming only).",
    )
    eval_parser.add_argument(
        "--negative-classes",
        default=["other"],
        type=lambda value: [item.strip() for item in value.split(",") if item.strip()],
        help="Comma-separated class labels treated as negative classes (default: other).",
    )
    eval_parser.add_argument(
        "--positive-classes-population-ratio",
        type=float,
        default=None,
        help=(
            "Target ratio of positive samples to total samples after subsampling "
            "(non-streaming only). Positive classes are inferred as labels not in --negative-classes."
        ),
    )
    eval_parser.add_argument("--top-k", type=int, default=1, help="Top-k predictions (evaluation uses top-1).")
    eval_parser.add_argument(
        "--embedding-source",
        choices=["cls", "cls_mean"],
        default="cls_mean",
        help="Embedding reduction used for KNN features: CLS only or CLS plus mean patch tokens.",
    )
    eval_parser.add_argument("--device", type=int, default=-1, help="Transformers device index (-1 for CPU).")
    eval_parser.add_argument(
        "--pad-to-square",
        action="store_true",
        help="Pad images with black pixels to square shape before preprocessing.",
    )
    eval_parser.add_argument(
        "--skip-channel-information",
        choices=["R", "G", "B"],
        default=None,
        help="Clone selected channel into all RGB channels before optional square padding.",
    )
    eval_parser.add_argument(
        "--debug-save-transformed-samples-dir",
        default=None,
        help="Optional output directory to save a few transformed eval images for debugging.",
    )
    eval_parser.add_argument(
        "--debug-save-transformed-samples-count",
        type=int,
        default=0,
        help="Number of transformed eval images to save to --debug-save-transformed-samples-dir.",
    )
    eval_parser.add_argument(
        "--reid-cmc-ranks",
        type=lambda value: [int(item.strip()) for item in value.split(",") if item.strip()],
        default=[1, 5, 10],
        help="Comma-separated CMC ranks for re-identification metrics (default: 1,5,10).",
    )

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
