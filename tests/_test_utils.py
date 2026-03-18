from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from datasets import ClassLabel, Dataset, Features
from datasets import Image as HFImage
from PIL import Image
from transformers import BitImageProcessor, Dinov2Config, Dinov2Model

EXPECTED_CLASSES = ("class_a", "class_b", "class_c")


def get_hf_dataset(rng: np.random.Generator, num_samples: int = 100) -> tuple[Dataset, Image.Image]:
    """Build a dummy HF Dataset and return it with one sample image."""
    rows: list[dict[str, Any]] = []
    for i in range(num_samples):
        rows.append(
            {
                "image": Image.fromarray(rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8), mode="RGB"),
                "label": i % len(EXPECTED_CLASSES),
            }
        )

    features = Features(
        {
            "image": HFImage(),
            "label": ClassLabel(names=list(EXPECTED_CLASSES)),
        }
    )
    return Dataset.from_list(rows, features=features), rows[0]["image"]


def build_local_imagefolder_dataset(
    root: Path,
    rng: np.random.Generator,
    *,
    samples_per_class: int = 8,
) -> Path:
    """Create an on-disk imagefolder dataset with one subdirectory per class."""
    root.mkdir(parents=True, exist_ok=True)
    for class_name in EXPECTED_CLASSES:
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(samples_per_class):
            image = Image.fromarray(rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8), mode="RGB")
            image.save(class_dir / f"{class_name}_{idx}.png")
    return root


def build_local_dinov2_checkpoint(root: Path) -> Path:
    """Create a tiny local Dinov2 backbone checkpoint and matching processor."""
    root.mkdir(parents=True, exist_ok=True)
    config = Dinov2Config(
        image_size=32,
        patch_size=16,
        num_channels=3,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        mlp_ratio=2,
        intermediate_size=64,
    )
    model = Dinov2Model(config)
    model.save_pretrained(root)
    processor = BitImageProcessor(
        do_resize=True,
        size={"height": 32, "width": 32},
        do_rescale=True,
        do_normalize=False,
    )
    processor.save_pretrained(root)
    return root
