"""Post-release smoke test for inference using installed PyPI package.

This script generates a tiny local transformer model + KNN head and runs
single and batched inference through the package API (no CLI usage).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from transformers import ViTConfig, ViTImageProcessor, ViTModel

from transformers_knn_adapter.knn_image_pipeline import pipeline


def main() -> None:
    workdir = Path("/tmp/pypi-check-fixtures")
    model_dir = workdir / "tiny-local-vit"
    knn_path = workdir / "knn_head.joblib"
    image_path = workdir / "fake-image.png"
    workdir.mkdir(parents=True, exist_ok=True)

    config = ViTConfig(
        image_size=32,
        patch_size=16,
        num_channels=3,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
    )
    model = ViTModel(config)
    model.save_pretrained(model_dir)

    processor = ViTImageProcessor(
        do_resize=True,
        size={"height": 32, "width": 32},
        do_rescale=True,
        do_normalize=False,
    )
    processor.save_pretrained(model_dir)

    features = np.random.default_rng(1234).normal(size=(12, 64))
    labels = np.array(["class_a"] * 6 + ["class_b"] * 6, dtype=object)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(features, labels)
    joblib.dump(knn, knn_path)

    image = Image.fromarray(
        np.random.default_rng(4321).integers(0, 255, size=(32, 32, 3), dtype=np.uint8),
        mode="RGB",
    )
    image.save(image_path)

    clf = pipeline(
        "image-classification",
        model_path=str(model_dir),
        knn_model_path=str(knn_path),
        device=-1,
        top_k=2,
    )

    single_result = clf(str(image_path))
    batch_result = clf([str(image_path)] * 5)

    assert isinstance(single_result, list) and single_result and isinstance(single_result[0], dict)
    assert isinstance(batch_result, list) and len(batch_result) == 5
    assert all(isinstance(item, list) and item for item in batch_result)

    print("PyPI inference smoke check passed")


if __name__ == "__main__":
    main()
