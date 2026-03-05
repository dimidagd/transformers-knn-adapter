"""Pytest smoke tests for KNN head training from local dummy datasets.

This module intentionally avoids remote HF downloads. It validates training
with a dummy `datasets.Dataset` object (HF datasets format).

How to run:
    uv run pytest tests/test_knn_pipeline.py -q
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from datasets import ClassLabel, Dataset, Features, Image as HFImage
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor, ViTModel

from transformers_knn_adapter.knn_image_pipeline import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

EXPECTED_CLASSES = ("class_a", "class_b")
TRAIN_DEFAULTS = {
    "image_column": "image",
    "label_column": "label",
    "batch_size": 4,
    "n_neighbors": 1,
}
ORDERED_CLASS_A_COUNT = 60
ORDERED_CLASS_B_COUNT = 60
SUBSET_MAX_SAMPLES = 20
TEST_SHUFFLE_SEED = 1234
GRID_SEARCH_K_VALUES = {1, 2, 4, 8, 16, 32}
GRID_SEARCH_METRICS = {"euclidean", "manhattan", "chebyshev", "minkowski", "cosine"}


def _build_local_pipeline(
    workdir: Path,
    top_k: int = 2,
    model_class: type[ViTModel] | type[ViTForImageClassification] = ViTModel,
):
    """Create a local untrained tiny ViT/checkpoint + KNN pipeline (no network)."""
    model_dir = workdir / "tiny-local-vit"
    model_dir.mkdir(parents=True, exist_ok=True)
    knn_path = workdir / "knn_head.joblib"

    config = ViTConfig(
        image_size=32,
        patch_size=16,
        num_channels=3,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
    )
    model = model_class(config)
    model.save_pretrained(model_dir)

    processor = ViTImageProcessor(
        do_resize=True,
        size={"height": 32, "width": 32},
        do_rescale=True,
        do_normalize=False,
    )
    processor.save_pretrained(model_dir)

    return pipeline(
        "image-classification",
        model_path=str(model_dir),
        knn_model_path=str(knn_path),
        device=-1,
        top_k=top_k,
    ), knn_path


def _assert_inference_shape(clf, sample: Image.Image) -> None:
    """Check single/batch outputs match expected pipeline shape."""
    single = clf(sample)
    batch = clf([sample, sample])

    assert isinstance(single, list), "Single inference should return a list"
    assert single and isinstance(single[0], dict), "Single inference items should be dicts"
    assert "label" in single[0] and "score" in single[0], "Inference dict should have label+score"
    assert isinstance(batch, list) and len(batch) == 2, "Batch inference should return list of length 2"
    assert isinstance(batch[0], list), "Each batch item should be a list of predictions"


def _run_training_case(
    pipeline_factory,
    dataset,
    *,
    split: str | None = None,
    model_class: type[ViTModel] | type[ViTForImageClassification] = ViTModel,
    train_kwargs: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Create and train a pipeline with shared defaults plus optional overrides."""
    clf, knn_path = pipeline_factory(model_class=model_class)
    kwargs = dict(TRAIN_DEFAULTS)
    if train_kwargs:
        kwargs.update(train_kwargs)
    clf.train(dataset=dataset, split=split, **kwargs)
    return {
        "clf": clf,
        "knn_path": knn_path,
        "dataset": dataset,
        "split": split,
    }


@pytest.fixture()
def rng() -> np.random.Generator:
    """Provide deterministic RNG for synthetic test images."""
    return np.random.default_rng(1234)


def get_hf_dataset(rng: np.random.Generator, num_samples: int = 100) -> tuple[Dataset, Image.Image]:
    """Build a dummy HF Dataset and return it with one sample image."""
    rows = []
    for i in range(num_samples):
        rows.append(
            {
                "image": Image.fromarray(rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8), mode="RGB"),
                "label": int(rng.integers(0, len(EXPECTED_CLASSES))),
            }
        )

    features = Features(
        {
            "image": HFImage(),
            "label": ClassLabel(names=list(EXPECTED_CLASSES)),
        }
    )
    return Dataset.from_list(rows, features=features), rows[0]["image"]


def _build_ordered_dataset(
    rng: np.random.Generator,
    *,
    first_label_count: int,
    second_label_count: int,
    first_label: str = "class_a",
    second_label: str = "class_b",
) -> Dataset:
    """Build an ordered two-class dataset with first_label rows first."""
    label_to_id = {label: i for i, label in enumerate(EXPECTED_CLASSES)}
    if first_label not in label_to_id or second_label not in label_to_id:
        raise ValueError("first_label and second_label must be members of EXPECTED_CLASSES")

    rows = []
    for _ in range(first_label_count):
        rows.append(
            {
                "image": Image.fromarray(rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8), mode="RGB"),
                "label": label_to_id[first_label],
            }
        )
    for _ in range(second_label_count):
        rows.append(
            {
                "image": Image.fromarray(rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8), mode="RGB"),
                "label": label_to_id[second_label],
            }
        )

    features = Features(
        {
            "image": HFImage(),
            "label": ClassLabel(names=list(EXPECTED_CLASSES)),
        }
    )
    return Dataset.from_list(rows, features=features)


def _build_hf_case(rng: np.random.Generator) -> tuple[Dataset, Image.Image, str | None]:
    """Build dataset case for HF Dataset input."""
    dataset, sample_image = get_hf_dataset(rng=rng)
    return dataset, sample_image, None


def _get_training_samples_and_labels(dataset: Dataset) -> tuple[list[Image.Image], list[str]]:
    """Extract training images and normalized string labels for assertions."""
    label_names = dataset.features["label"].names
    images = [row["image"] for row in dataset]
    labels = [label_names[int(row["label"])] for row in dataset]
    return images, labels


def _assert_predictions_match_training_labels(clf, images: list[Image.Image], labels: list[str]) -> None:
    """Assert top-1 predictions exactly match the provided labels."""
    predictions = clf(images)
    predicted_labels = [pred[0]["label"] for pred in predictions]
    assert predicted_labels == labels, "Top-1 predictions should match training labels for all training samples"


def _assert_eval_metrics(metrics: dict[str, object], labels: list[str]) -> None:
    """Assert evaluation metrics are internally consistent and perfect on training data."""
    expected_count = len(labels)
    expected_label_counts = {label: labels.count(label) for label in sorted(set(labels))}

    assert metrics["samples"] == expected_count, "Eval sample count should match dataset size"
    assert metrics["correct"] == expected_count, "All training samples should be predicted correctly"
    assert metrics["top1_accuracy"] == 1.0, "Top-1 accuracy should be perfect on training data"
    assert metrics["true_label_counts"] == expected_label_counts, "True label counts should match expected labels"
    assert metrics["pred_label_counts"] == expected_label_counts, "Pred label counts should match expected labels"
    report_text = metrics["classification_report"]
    assert isinstance(report_text, str) and "precision" in report_text, "Classification report should be present"


def _assert_artifact_and_model(case: dict[str, object]) -> None:
    """Assert training produced a persisted artifact and a fitted model."""
    clf = case["clf"]
    knn_path = case["knn_path"]
    assert knn_path.exists(), "KNN artifact should be created"
    assert clf.knn_model is not None, "KNN model should be fitted"


def _get_knn_classes(clf) -> set[str]:
    """Return KNN classes as a string set for stable assertions."""
    assert clf.knn_model is not None, "KNN model should be fitted"
    return {str(label) for label in clf.knn_model.classes_}


def _assert_knn_classes(clf, expected: set[str]) -> None:
    """Assert the trained KNN class set exactly matches expectation."""
    assert _get_knn_classes(clf) == expected, "KNN classes should match expected labels"


def _build_sampling_bias_dataset(rng: np.random.Generator) -> Dataset:
    """Build a deterministic ordered dataset used for sampling/shuffle/stratified tests."""
    return _build_ordered_dataset(
        rng=rng,
        first_label_count=ORDERED_CLASS_A_COUNT,
        second_label_count=ORDERED_CLASS_B_COUNT,
    )


def _evaluate_true_label_set(
    clf,
    dataset: Dataset,
    *,
    max_samples: int = SUBSET_MAX_SAMPLES,
    **eval_kwargs: Any,
) -> set[str]:
    """Run eval and return the set of true labels observed in the sampled subset."""
    metrics = clf.evaluate(
        dataset=dataset,
        split="train",
        image_column="image",
        label_column="label",
        max_samples=max_samples,
        **eval_kwargs,
    )
    return set(metrics["true_label_counts"])


def _build_subset_sampling_train_kwargs(*, stratified: bool, shuffle: bool) -> dict[str, Any]:
    """Build train kwargs for max-samples sampling behavior tests."""
    kwargs: dict[str, Any] = {
        "max_samples": SUBSET_MAX_SAMPLES,
        "stratified": stratified,
        "shuffle": shuffle,
    }
    if stratified or shuffle:
        kwargs["shuffle_seed"] = TEST_SHUFFLE_SEED
    if stratified:
        kwargs["streaming"] = False
    if shuffle:
        kwargs["shuffle_buffer_size"] = ORDERED_CLASS_A_COUNT + ORDERED_CLASS_B_COUNT
    return kwargs


def _build_subset_sampling_eval_kwargs(*, stratified: bool, shuffle: bool) -> dict[str, Any]:
    """Build eval kwargs for max-samples sampling behavior tests."""
    kwargs: dict[str, Any] = {
        "stratified": stratified,
        "shuffle": shuffle,
    }
    if stratified or shuffle:
        kwargs["shuffle_seed"] = TEST_SHUFFLE_SEED
    if shuffle:
        kwargs["shuffle_buffer_size"] = ORDERED_CLASS_A_COUNT + ORDERED_CLASS_B_COUNT
    return kwargs


@pytest.fixture()
def pipeline_factory():
    """Provide a callable that creates an isolated local pipeline + KNN artifact path."""
    tempdir = tempfile.TemporaryDirectory()
    workdir = Path(tempdir.name)
    try:
        yield lambda top_k=2, model_class=ViTModel: _build_local_pipeline(
            workdir, top_k=top_k, model_class=model_class
        )
    finally:
        tempdir.cleanup()


@pytest.fixture(
    params=[
        pytest.param(_build_hf_case, id="hf_dataset"),
    ]
)
def dataset_case(
    request, rng: np.random.Generator
) -> tuple[Dataset, Image.Image, str | None]:
    """Provide (dataset, sample_image, split) for each supported dataset format."""
    builder = request.param
    assert isinstance(builder, Callable)
    return builder(rng)


@pytest.fixture(
    params=[
        pytest.param(ViTModel, id="feature_checkpoint"),
        pytest.param(ViTForImageClassification, id="classification_checkpoint"),
    ]
)
def model_class(request) -> type[ViTModel] | type[ViTForImageClassification]:
    """Provide model class variant to validate pipeline compatibility."""
    cls = request.param
    assert isinstance(cls, type)
    return cls


@pytest.fixture()
def trained_case(
    pipeline_factory,
    dataset_case: tuple[Dataset, Image.Image, str | None],
    model_class: type[ViTModel] | type[ViTForImageClassification],
) -> dict[str, object]:
    """Build and train a pipeline for each supported dummy dataset format."""
    dataset, sample_image, split = dataset_case

    logger.info("Preparing trained test case")
    case = _run_training_case(
        pipeline_factory=pipeline_factory,
        dataset=dataset,
        split=split,
        model_class=model_class,
    )
    train_images, train_labels = _get_training_samples_and_labels(dataset=dataset)
    case.update(
        {
            "sample_image": sample_image,
            "train_images": train_images,
            "train_labels": train_labels,
        }
    )
    return case


def test_train_with_dummy_datasets(trained_case: dict[str, object]) -> None:
    """Test KNN training artifacts and classes for both dummy dataset formats."""
    clf = trained_case["clf"]
    _assert_artifact_and_model(trained_case)
    _assert_knn_classes(clf=clf, expected=set(EXPECTED_CLASSES))


def test_inference_with_dummy_datasets(trained_case: dict[str, object]) -> None:
    """Test inference output shape and exact training-set predictions."""
    clf = trained_case["clf"]
    sample_image = trained_case["sample_image"]
    train_images = trained_case["train_images"]
    train_labels = trained_case["train_labels"]

    _assert_inference_shape(clf, sample_image)
    _assert_predictions_match_training_labels(clf=clf, images=train_images, labels=train_labels)


def test_evaluate_with_dummy_datasets(trained_case: dict[str, object]) -> None:
    """Test evaluate metrics for both dummy dataset formats."""
    clf = trained_case["clf"]
    dataset = trained_case["dataset"]
    split = trained_case["split"]
    train_labels = trained_case["train_labels"]

    eval_split = split or "train"
    metrics = clf.evaluate(
        dataset=dataset,
        split=eval_split,
        image_column="image",
        label_column="label",
    )
    _assert_eval_metrics(metrics=metrics, labels=train_labels)


@pytest.mark.parametrize("batch_size", [1, 2], ids=["bs1", "bs2"])
def test_evaluate_with_parametrized_batch_size(
    trained_case: dict[str, object],
    batch_size: int,
) -> None:
    """Evaluate metrics should remain correct across configured eval batch sizes."""
    clf = trained_case["clf"]
    dataset = trained_case["dataset"]
    split = trained_case["split"] or "train"
    train_labels = trained_case["train_labels"]

    metrics = clf.evaluate(
        dataset=dataset,
        split=split,
        image_column="image",
        label_column="label",
        batch_size=batch_size,
    )
    _assert_eval_metrics(metrics=metrics, labels=train_labels)


def test_train_streaming_iterable_dataset(
    pipeline_factory,
    rng: np.random.Generator,
    model_class: type[ViTModel] | type[ViTForImageClassification],
) -> None:
    """Train successfully with streaming enabled on an iterable dataset."""
    dataset, _ = get_hf_dataset(rng=rng)
    iterable_dataset = dataset.to_iterable_dataset()
    case = _run_training_case(
        pipeline_factory=pipeline_factory,
        dataset=iterable_dataset,
        model_class=model_class,
        train_kwargs={"streaming": True},
    )
    _assert_artifact_and_model(case)
    _assert_knn_classes(case["clf"], set(EXPECTED_CLASSES))


def test_train_with_grid_search_selects_knn_hyperparameters(
    pipeline_factory,
    rng: np.random.Generator,
) -> None:
    """Grid search should fit KNN with one of the configured k/metric values."""
    dataset = _build_sampling_bias_dataset(rng=rng)
    case = _run_training_case(
        pipeline_factory=pipeline_factory,
        dataset=dataset,
        train_kwargs={
            "max_samples": 40,
            "grid_search": True,
            "grid_search_splits": 3,
            "grid_search_repeats": 1,
            "grid_search_scoring": "f1_macro",
        },
    )
    _assert_artifact_and_model(case)
    knn_model = case["clf"].knn_model
    assert knn_model is not None
    assert int(knn_model.n_neighbors) in GRID_SEARCH_K_VALUES
    assert str(knn_model.metric) in GRID_SEARCH_METRICS


@pytest.mark.parametrize(
    ("stratified", "shuffle", "expected_classes"),
    [
        pytest.param(False, False, {"class_a"}, id="unstratified-unshuffled"),
        pytest.param(False, True, set(EXPECTED_CLASSES), id="unstratified-shuffled"),
        pytest.param(True, False, set(EXPECTED_CLASSES), id="stratified-unshuffled"),
        pytest.param(True, True, set(EXPECTED_CLASSES), id="stratified-shuffled"),
    ],
)
def test_train_sampling_modes_with_max_samples(
    pipeline_factory,
    rng: np.random.Generator,
    model_class: type[ViTModel] | type[ViTForImageClassification],
    stratified: bool,
    shuffle: bool,
    expected_classes: set[str],
) -> None:
    """Train sampling modes should yield expected class coverage under max_samples."""
    dataset = _build_sampling_bias_dataset(rng=rng)
    case = _run_training_case(
        pipeline_factory=pipeline_factory,
        dataset=dataset,
        model_class=model_class,
        train_kwargs=_build_subset_sampling_train_kwargs(
            stratified=stratified,
            shuffle=shuffle,
        ),
    )
    _assert_artifact_and_model(case)
    _assert_knn_classes(case["clf"], expected_classes)


@pytest.mark.parametrize(
    ("stratified", "shuffle", "expected_true_labels"),
    [
        pytest.param(False, False, {"class_a"}, id="unstratified-unshuffled"),
        pytest.param(False, True, set(EXPECTED_CLASSES), id="unstratified-shuffled"),
        pytest.param(True, False, set(EXPECTED_CLASSES), id="stratified-unshuffled"),
        pytest.param(True, True, set(EXPECTED_CLASSES), id="stratified-shuffled"),
    ],
)
def test_evaluate_sampling_modes_with_max_samples(
    pipeline_factory,
    rng: np.random.Generator,
    model_class: type[ViTModel] | type[ViTForImageClassification],
    stratified: bool,
    shuffle: bool,
    expected_true_labels: set[str],
) -> None:
    """Eval sampling modes should yield expected true-label coverage under max_samples."""
    dataset = _build_sampling_bias_dataset(rng=rng)
    train_case = _run_training_case(
        pipeline_factory=pipeline_factory,
        dataset=dataset,
        model_class=model_class,
    )
    clf = train_case["clf"]
    true_labels = _evaluate_true_label_set(
        clf=clf,
        dataset=dataset,
        **_build_subset_sampling_eval_kwargs(
            stratified=stratified,
            shuffle=shuffle,
        ),
    )
    assert true_labels == expected_true_labels
