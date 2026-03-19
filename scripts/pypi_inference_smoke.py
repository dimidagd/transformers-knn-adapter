"""Post-release smoke test for inference using the installed PyPI package."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from transformers import ViTConfig, ViTImageProcessor, ViTModel

from transformers_knn_adapter.knn_image_pipeline import pipeline


def extract_pooler_embeddings(
    model: ViTModel,
    processor: ViTImageProcessor,
    images: list[Image.Image],
) -> np.ndarray:
    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.cpu().numpy()


def main() -> None:
    rng = np.random.default_rng(1234)
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

    train_images = [
        Image.fromarray(
            rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8),
            mode="RGB",
        )
        for _ in range(4)
    ]
    features = extract_pooler_embeddings(model, processor, train_images)
    labels = np.array(["class_a", "class_b", "class_a", "class_b"], dtype=object)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(features, labels)
    joblib.dump(knn, knn_path)

    image = train_images[0]
    image.save(image_path)

    clf = pipeline(
        "image-classification",
        model_path=str(model_dir),
        knn_model_path=str(knn_path),
        device=-1,
        top_k=2,
    )

    result = clf(str(image_path))
    assert isinstance(result, list) and result and isinstance(result[0], dict)
    assert "label" in result[0] and "score" in result[0]

    print("PyPI inference smoke check passed")


if __name__ == "__main__":
    main()
