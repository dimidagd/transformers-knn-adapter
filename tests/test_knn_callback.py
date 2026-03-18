from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.trainer_callback import TrainerControl, TrainerState

import transformers_knn_adapter.knn_callback as knn_callback_module
from transformers_knn_adapter.knn_callback import KNNCallback


class _FakeEmbeddingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(embedding_source="cls_mean")

    def forward(self, pixel_values: torch.Tensor, output_hidden_states: bool = False, **kwargs: Any) -> BaseModelOutputWithPooling:
        del kwargs
        values = pixel_values[:, 0, 0, 0].view(-1, 1)
        cls = torch.cat([values, values * 0 + 1.0], dim=1)
        patch_a = torch.cat([values * 0 + 2.0, values * 0], dim=1)
        patch_b = torch.cat([values * 0 + 4.0, values * 0], dim=1)
        sequence_output = torch.stack([cls, patch_a, patch_b], dim=1)
        hidden_states = (sequence_output,) if output_hidden_states else None
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=sequence_output[:, 0, :],
            hidden_states=hidden_states,
        )


class _FakeWrapperModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dinov2 = _FakeEmbeddingModel()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(embedding_source="cls_mean")

    def forward(self, pixel_values: torch.Tensor, output_hidden_states: bool = False, **kwargs: Any) -> BaseModelOutputWithPooling:
        return self.dinov2(pixel_values, output_hidden_states=output_hidden_states, **kwargs)


class _FakeNaNEmbeddingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(embedding_source="cls_mean")

    def forward(self, pixel_values: torch.Tensor, output_hidden_states: bool = False, **kwargs: Any) -> BaseModelOutputWithPooling:
        del pixel_values, kwargs
        sequence_output = torch.tensor([[[float("nan"), 1.0], [2.0, 3.0], [4.0, 5.0]]], dtype=torch.float32)
        hidden_states = (sequence_output,) if output_hidden_states else None
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=sequence_output[:, 0, :],
            hidden_states=hidden_states,
        )

def _make_example(value: int, label: int) -> dict[str, torch.Tensor | int]:
    pixel_values = torch.tensor([[[float(value)]]], dtype=torch.float32)
    return {"pixel_values": pixel_values, "label": label}


def test_knn_callback_logs_macro_metrics_from_trainer_embeddings() -> None:
    train_dataset = [
        _make_example(10, 0),
        _make_example(12, 0),
        _make_example(200, 1),
        _make_example(202, 1),
    ]
    eval_dataset = [
        _make_example(11, 0),
        _make_example(201, 1),
    ]
    logged_metrics: list[dict[str, float]] = []
    trainer = SimpleNamespace(
        model=_FakeEmbeddingModel().train(),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SimpleNamespace(per_device_eval_batch_size=2),
        log=lambda payload: logged_metrics.append(payload),
    )
    callback = KNNCallback(trainer=trainer, ks=(1,))
    metrics: dict[str, float] = {}

    control = callback.on_evaluate(
        trainer.args,
        TrainerState(),
        TrainerControl(),
        metrics=metrics,
    )

    assert control is not None
    assert metrics["eval_knn_1/f1/macro"] == 1.0
    assert metrics["eval_knn_1/precision/macro"] == 1.0
    assert metrics["eval_knn_1/recall/macro"] == 1.0
    assert metrics["eval_knn_1/recall_at_1/macro"] == 1.0
    assert metrics["eval_knn_1/mrr/macro"] == 1.0
    assert logged_metrics == [metrics]
    assert trainer.model.training is True


def test_knn_callback_uses_average_in_metric_name() -> None:
    train_dataset = [
        _make_example(10, 0),
        _make_example(12, 0),
        _make_example(200, 1),
        _make_example(202, 1),
    ]
    eval_dataset = [
        _make_example(11, 0),
        _make_example(201, 1),
    ]
    trainer = SimpleNamespace(
        model=_FakeEmbeddingModel().train(),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SimpleNamespace(per_device_eval_batch_size=2),
        log=lambda payload: None,
    )
    callback = KNNCallback(trainer=trainer, ks=(1,), average="weighted")
    metrics: dict[str, float] = {}

    callback.on_evaluate(
        trainer.args,
        TrainerState(),
        TrainerControl(),
        metrics=metrics,
    )

    assert "eval_knn_1/f1/weighted" in metrics
    assert "eval_knn_1/precision/weighted" in metrics
    assert "eval_knn_1/recall/weighted" in metrics


def test_knn_callback_logs_recall_at_k_and_mrr() -> None:
    train_dataset = [
        _make_example(0, 0),
        _make_example(100, 1),
        _make_example(101, 1),
    ]
    eval_dataset = [_make_example(99, 0)]
    trainer = SimpleNamespace(
        model=_FakeEmbeddingModel().train(),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SimpleNamespace(per_device_eval_batch_size=1),
        log=lambda payload: None,
    )
    callback = KNNCallback(trainer=trainer, ks=(2,))
    metrics: dict[str, float] = {}

    callback.on_evaluate(
        trainer.args,
        TrainerState(),
        TrainerControl(),
        metrics=metrics,
    )

    assert metrics["eval_knn_2/recall_at_2/macro"] == 0.0
    assert metrics["eval_knn_2/mrr/macro"] == 0.0


def test_knn_callback_retrieval_metrics_capture_rank_order() -> None:
    neighbor_labels = torch.tensor([[1, 0], [1, 1]], dtype=torch.long).numpy()
    true_labels = torch.tensor([0, 1], dtype=torch.long).numpy()

    recall_at_k, mrr = KNNCallback._compute_retrieval_metrics_from_neighbor_labels(neighbor_labels, true_labels)

    assert recall_at_k == 1.0
    assert mrr == 0.75


def test_knn_callback_averaged_retrieval_metrics_balance_classes() -> None:
    neighbor_labels = torch.tensor(
        [
            [0, 1],
            [0, 1],
            [0, 0],
            [0, 0],
        ],
        dtype=torch.long,
    ).numpy()
    true_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long).numpy()

    macro_recall_at_k, macro_mrr = KNNCallback._compute_averaged_retrieval_metrics_from_neighbor_labels(
        neighbor_labels,
        true_labels,
        average="macro",
    )

    assert macro_recall_at_k == 0.5
    assert macro_mrr == 0.5

    weighted_recall_at_k, weighted_mrr = KNNCallback._compute_averaged_retrieval_metrics_from_neighbor_labels(
        neighbor_labels,
        true_labels,
        average="weighted",
    )

    assert weighted_recall_at_k == 0.5
    assert weighted_mrr == 0.5


def test_knn_callback_extract_embeddings_uses_cls_mean_by_default() -> None:
    outputs = BaseModelOutputWithPooling(
        last_hidden_state=torch.tensor([[[3.0, 4.0], [5.0, 6.0], [9.0, 10.0]]]),
        pooler_output=torch.tensor([[3.0, 4.0]]),
        hidden_states=(
            torch.tensor([[[3.0, 4.0], [5.0, 6.0], [9.0, 10.0]]]),
        ),
    )

    embeddings = KNNCallback._extract_embeddings(outputs, embedding_source="cls_mean")

    assert torch.equal(embeddings, torch.tensor([[3.0, 4.0, 7.0, 8.0]]))


def test_knn_callback_defaults_to_model_embedding_source() -> None:
    trainer = SimpleNamespace(
        model=_FakeEmbeddingModel().train(),
        train_dataset=[],
        eval_dataset=[],
        args=SimpleNamespace(per_device_eval_batch_size=2),
        log=lambda payload: None,
    )
    callback = KNNCallback(trainer=trainer)

    assert callback._resolve_embedding_source(trainer.model) == "cls_mean"


def test_knn_callback_extract_embeddings_uses_cls_only_in_cls_mode() -> None:
    outputs = BaseModelOutputWithPooling(
        last_hidden_state=torch.tensor([[[3.0, 4.0], [5.0, 6.0], [9.0, 10.0]]]),
        pooler_output=torch.tensor([[3.0, 4.0]]),
        hidden_states=(
            torch.tensor([[[3.0, 4.0], [5.0, 6.0], [9.0, 10.0]]]),
        ),
    )

    embeddings = KNNCallback._extract_embeddings(outputs, embedding_source="cls")

    assert torch.equal(embeddings, torch.tensor([[3.0, 4.0]]))


def test_knn_callback_cls_mode_logs_metrics() -> None:
    train_dataset = [
        _make_example(10, 0),
        _make_example(12, 0),
        _make_example(200, 1),
        _make_example(202, 1),
    ]
    eval_dataset = [
        _make_example(11, 0),
        _make_example(201, 1),
    ]
    logged_metrics: list[dict[str, float]] = []
    trainer = SimpleNamespace(
        model=_FakeWrapperModel().train(),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SimpleNamespace(per_device_eval_batch_size=2, disable_tqdm=True),
        log=lambda payload: logged_metrics.append(payload),
    )
    callback = KNNCallback(trainer=trainer, ks=(1,), embedding_source="cls")
    metrics: dict[str, float] = {}

    callback.on_evaluate(
        trainer.args,
        TrainerState(),
        TrainerControl(),
        metrics=metrics,
    )

    assert metrics["eval_knn_1/recall_at_1/macro"] == 1.0
    assert metrics["eval_knn_1/mrr/macro"] == 1.0
    assert logged_metrics == [metrics]


def test_knn_callback_warns_when_callback_and_model_embedding_sources_differ(caplog: pytest.LogCaptureFixture) -> None:
    train_dataset = [
        _make_example(10, 0),
        _make_example(12, 0),
        _make_example(200, 1),
        _make_example(202, 1),
    ]
    eval_dataset = [
        _make_example(11, 0),
        _make_example(201, 1),
    ]
    model = _FakeWrapperModel().train()
    model.config.embedding_source = "cls_mean"
    trainer = SimpleNamespace(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SimpleNamespace(per_device_eval_batch_size=2, disable_tqdm=True, dataloader_num_workers=0),
        log=lambda payload: None,
    )
    callback = KNNCallback(trainer=trainer, ks=(1,), embedding_source="cls")

    with caplog.at_level("WARNING"):
        callback.on_evaluate(
            trainer.args,
            TrainerState(),
            TrainerControl(),
            metrics={},
        )

    assert "does not match model.config.embedding_source" in caplog.text


def test_knn_callback_requires_pixel_values_from_dataset() -> None:
    trainer = SimpleNamespace(
        model=_FakeEmbeddingModel().train(),
        train_dataset=[{"label": 0}],
        eval_dataset=[{"label": 0}],
        args=SimpleNamespace(per_device_eval_batch_size=1),
        log=lambda payload: None,
    )
    callback = KNNCallback(trainer=trainer, ks=(1,))

    with pytest.raises(ValueError, match="expects datasets to yield `pixel_values`"):
        callback.on_evaluate(
            trainer.args,
            TrainerState(),
            TrainerControl(),
            metrics={},
        )


def test_knn_callback_raises_on_non_finite_embeddings() -> None:
    trainer = SimpleNamespace(
        model=_FakeNaNEmbeddingModel().train(),
        train_dataset=[_make_example(1, 0)],
        eval_dataset=[_make_example(1, 0)],
        args=SimpleNamespace(per_device_eval_batch_size=1, disable_tqdm=True),
        log=lambda payload: None,
    )
    callback = KNNCallback(trainer=trainer, ks=(1,))

    with pytest.raises(ValueError, match="train embeddings contains non-finite values"):
        callback.on_evaluate(
            trainer.args,
            TrainerState(),
            TrainerControl(),
            metrics={},
        )


def test_knn_callback_fits_knn_once_for_multiple_k(monkeypatch: pytest.MonkeyPatch) -> None:
    train_dataset = [
        _make_example(10, 0),
        _make_example(12, 0),
        _make_example(200, 1),
        _make_example(202, 1),
    ]
    eval_dataset = [
        _make_example(11, 0),
        _make_example(201, 1),
    ]
    trainer = SimpleNamespace(
        model=_FakeEmbeddingModel().train(),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SimpleNamespace(per_device_eval_batch_size=2, disable_tqdm=True, dataloader_num_workers=0),
        log=lambda payload: None,
    )

    fit_calls = 0
    real_knn = knn_callback_module.KNeighborsClassifier

    class _CountingKNN(real_knn):
        def fit(self, X, y):
            nonlocal fit_calls
            fit_calls += 1
            return super().fit(X, y)

    monkeypatch.setattr(knn_callback_module, "KNeighborsClassifier", _CountingKNN)
    callback = KNNCallback(trainer=trainer, ks=(1, 2))
    metrics: dict[str, float] = {}

    callback.on_evaluate(
        trainer.args,
        TrainerState(),
        TrainerControl(),
        metrics=metrics,
    )

    assert fit_calls == 1
    assert "eval_knn_1/f1/macro" in metrics
    assert "eval_knn_2/f1/macro" in metrics
