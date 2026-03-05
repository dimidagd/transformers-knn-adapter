"""Public package exports for transformers_knn_adapter."""

from __future__ import annotations

from typing import Any

__all__ = ["KNNImageClassificationPipeline", "pipeline"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .knn_image_pipeline import KNNImageClassificationPipeline, pipeline

        exports = {
            "KNNImageClassificationPipeline": KNNImageClassificationPipeline,
            "pipeline": pipeline,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
