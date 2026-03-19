from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import TrainerCallback
from transformers.utils import logging

logger = logging.get_logger(__name__)


class KNNCallback(TrainerCallback):
    """Evaluate image embeddings with a KNN classifier during Trainer evaluation.

    The callback assumes the train/eval datasets already yield model-ready
    examples, either because they were preprocessed upfront or because a dataset
    transform is applied lazily. Each example must expose `pixel_values` and the
    configured label column.
    """

    def __init__(
        self,
        *,
        trainer: Any,
        image_column: str = "image",
        label_column: str = "label",
        ks: Sequence[int] = (1, 5),
        batch_size: int | None = None,
        average: str = "macro",
        zero_division: float | str = np.nan,
    ) -> None:
        self.trainer = trainer
        self.image_column = image_column
        self.label_column = label_column
        self.ks = tuple(int(k) for k in ks)
        self.batch_size = batch_size
        self.average = average
        self.zero_division = zero_division

    def _collate_batch(self, examples: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        if any("pixel_values" not in example for example in examples):
            raise ValueError(
                "KNNCallback expects datasets to yield `pixel_values`. "
                "Preprocess the dataset upfront or apply a dataset transform before evaluation."
            )
        pixel_values = torch.stack([torch.as_tensor(example["pixel_values"]) for example in examples])
        labels = torch.tensor([int(example[self.label_column]) for example in examples], dtype=torch.long)
        return pixel_values, labels

    @staticmethod
    def _extract_embeddings(model_outputs: Any) -> torch.Tensor:
        pooled_output = getattr(model_outputs, "pooler_output", None)
        if pooled_output is not None:
            return pooled_output
        raise ValueError("Model outputs must expose pooler_output for KNN embedding extraction.")

    def _build_loader(self, dataset: Any, batch_size: int) -> DataLoader:
        num_workers = int(getattr(self.trainer.args, "dataloader_num_workers", 0))
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda batch: batch,
        )

    @staticmethod
    def _ensure_finite(name: str, values: torch.Tensor | np.ndarray) -> None:
        if isinstance(values, torch.Tensor):
            finite = torch.isfinite(values)
            bad_count = int((~finite).sum().item())
        else:
            finite = np.isfinite(values)
            bad_count = int((~finite).sum())
        if bad_count:
            raise ValueError(f"{name} contains non-finite values ({bad_count} entries).")

    @staticmethod
    def _compute_retrieval_metrics_from_neighbor_labels(
        neighbor_labels: np.ndarray,
        true_labels: np.ndarray,
    ) -> tuple[float, float]:
        hits = neighbor_labels == true_labels[:, None]
        recall_at_k = float(hits.any(axis=1).mean())

        first_match_indices = hits.argmax(axis=1) + 1
        has_match = hits.any(axis=1)
        reciprocal_ranks = np.zeros(len(true_labels), dtype=np.float64)
        reciprocal_ranks[has_match] = 1.0 / first_match_indices[has_match]
        mrr = float(reciprocal_ranks.mean())
        return recall_at_k, mrr

    @classmethod
    def _compute_averaged_retrieval_metrics_from_neighbor_labels(
        cls,
        neighbor_labels: np.ndarray,
        true_labels: np.ndarray,
        *,
        average: str,
    ) -> tuple[float, float]:
        if average not in {"macro", "weighted"}:
            return cls._compute_retrieval_metrics_from_neighbor_labels(neighbor_labels, true_labels)

        per_class_recall_at_k: list[float] = []
        per_class_mrr: list[float] = []
        per_class_weights: list[float] = []
        for class_label in np.unique(true_labels):
            mask = true_labels == class_label
            recall_at_k, mrr = cls._compute_retrieval_metrics_from_neighbor_labels(
                neighbor_labels[mask],
                true_labels[mask],
            )
            per_class_recall_at_k.append(recall_at_k)
            per_class_mrr.append(mrr)
            per_class_weights.append(float(mask.sum()))

        if average == "macro":
            return float(np.mean(per_class_recall_at_k)), float(np.mean(per_class_mrr))

        return (
            float(np.average(per_class_recall_at_k, weights=per_class_weights)),
            float(np.average(per_class_mrr, weights=per_class_weights)),
        )

    @staticmethod
    def _predict_from_neighbor_labels(
        neighbor_labels: np.ndarray,
        neighbor_distances: np.ndarray,
        *,
        classes: np.ndarray,
    ) -> np.ndarray:
        class_to_index = {label: idx for idx, label in enumerate(classes.tolist())}
        encoded = np.vectorize(class_to_index.__getitem__, otypes=[np.int64])(neighbor_labels)
        predictions = np.empty(encoded.shape[0], dtype=classes.dtype)
        for row_idx, row in enumerate(encoded):
            distances = neighbor_distances[row_idx]
            if np.any(distances == 0.0):
                weights = np.zeros_like(distances, dtype=np.float64)
                weights[distances == 0.0] = 1.0
            else:
                weights = 1.0 / distances
            class_weights = np.bincount(row, weights=weights, minlength=len(classes))
            predictions[row_idx] = classes[int(np.argmax(class_weights))]
        return predictions

    def _collect_embeddings_and_labels(
        self,
        *,
        model: torch.nn.Module,
        dataset: Any,
        batch_size: int,
        split_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        loader = self._build_loader(dataset, batch_size)
        device = next(model.parameters()).device
        embeddings_batches: list[np.ndarray] = []
        labels_batches: list[np.ndarray] = []
        logger.info("Collecting KNN embeddings for %s split: %d samples with batch_size=%d", split_name, len(dataset), batch_size)

        progress = tqdm(
            loader,
            total=len(loader),
            desc=f"KNN embeddings ({split_name})",
            disable=bool(getattr(self.trainer.args, "disable_tqdm", False)),
        )
        for examples in progress:
            pixel_values, labels = self._collate_batch(examples)
            pixel_values = pixel_values.to(device)
            with torch.no_grad():
                outputs = model(pixel_values=pixel_values, output_hidden_states=True)
            embeddings = self._extract_embeddings(outputs)
            self._ensure_finite(f"{split_name} embeddings", embeddings)
            embeddings_batches.append(embeddings.detach().cpu().numpy())
            labels_batches.append(labels.numpy())

        embeddings_array = np.concatenate(embeddings_batches, axis=0)
        labels_array = np.concatenate(labels_batches, axis=0)
        self._ensure_finite(f"{split_name} embeddings", embeddings_array)
        return embeddings_array, labels_array

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs.get("metrics")
        if metrics is None:
            return control

        model = self.trainer.model
        train_dataset = self.trainer.train_dataset
        eval_dataset = self.trainer.eval_dataset
        if model is None or train_dataset is None or eval_dataset is None:
            raise ValueError("KNNCallback requires trainer.model, train_dataset, and eval_dataset.")

        batch_size = self.batch_size or int(self.trainer.args.per_device_eval_batch_size)
        logger.info(
            "Running KNN callback evaluation for ks=%s average=%s zero_division=%s batch_size=%d num_workers=%d",
            self.ks,
            self.average,
            self.zero_division,
            batch_size,
            int(getattr(self.trainer.args, "dataloader_num_workers", 0)),
        )
        was_training = model.training
        model.eval()
        try:
            train_X, train_y = self._collect_embeddings_and_labels(
                model=model,
                dataset=train_dataset,
                batch_size=batch_size,
                split_name="train",
            )
            eval_X, eval_y = self._collect_embeddings_and_labels(
                model=model,
                dataset=eval_dataset,
                batch_size=batch_size,
                split_name="eval",
            )
        finally:
            if was_training:
                model.train()

        knn_metrics: dict[str, float] = {}
        max_k = max(self.ks)
        logger.info("Fitting KNN classifier once with max_k=%d on %d train embeddings", max_k, len(train_y))
        knn = KNeighborsClassifier(n_neighbors=max_k, weights="distance")
        knn.fit(train_X, train_y)
        neighbor_distances, neighbor_indices = knn.kneighbors(eval_X, return_distance=True)
        all_neighbor_labels = train_y[neighbor_indices]
        classes = LabelEncoder().fit(train_y).classes_

        for k in self.ks:
            neighbor_labels = all_neighbor_labels[:, :k]
            distances = neighbor_distances[:, :k]
            pred_y = self._predict_from_neighbor_labels(neighbor_labels, distances, classes=classes)
            recall_at_k, mrr = self._compute_averaged_retrieval_metrics_from_neighbor_labels(
                neighbor_labels,
                eval_y,
                average=self.average,
            )
            precision, recall, f1, _ = precision_recall_fscore_support(
                eval_y,
                pred_y,
                average=self.average,
                zero_division=self.zero_division,
            )
            knn_metrics[f"eval_knn_{k}/f1/{self.average}"] = float(f1)
            knn_metrics[f"eval_knn_{k}/precision/{self.average}"] = float(precision)
            knn_metrics[f"eval_knn_{k}/recall/{self.average}"] = float(recall)
            knn_metrics[f"eval_knn_{k}/recall_at_{k}/{self.average}"] = recall_at_k
            knn_metrics[f"eval_knn_{k}/mrr/{self.average}"] = mrr
            logger.info(
                "KNN metrics for k=%d: f1=%0.4f precision=%0.4f recall=%0.4f recall_at_%d=%0.4f mrr=%0.4f average=%s",
                k,
                float(f1),
                float(precision),
                float(recall),
                k,
                recall_at_k,
                mrr,
                self.average,
            )
        metrics.update(knn_metrics)
        self.trainer.log(knn_metrics)
        return control
