"""Integration test for Dinov2 embeddings + ArcFace custom loss computation.

This test is opt-in because it downloads `facebook/dinov2-small` weights.

Run manually:
    RUN_DINOV2_ARCFACE_TEST=1 uv run --with pytorch-metric-learning \
      pytest tests/test_dinov2_arcface_integration.py -q
"""

from __future__ import annotations

import os

import pytest
import torch
from transformers import AutoModel


def compute_loss_and_logits(
    *,
    arcface_loss: torch.nn.Module,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ArcFace loss and logits using ArcFaceLoss internal components."""
    arcface_loss.cast_types(embeddings.dtype, embeddings.device)

    mask = arcface_loss.get_target_mask(embeddings, labels)
    cosine = arcface_loss.get_cosine(embeddings)
    cosine_of_targets = cosine[mask == 1]
    modified_cosine_of_targets = arcface_loss.modify_cosine_of_target_classes(cosine_of_targets)
    diff = (modified_cosine_of_targets - cosine_of_targets).unsqueeze(1)

    logits = cosine + (mask * diff)
    logits = arcface_loss.scale_logits(logits, embeddings)
    loss = arcface_loss.cross_entropy(logits, labels).mean()
    return loss, logits


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_DINOV2_ARCFACE_TEST") != "1",
    reason="Set RUN_DINOV2_ARCFACE_TEST=1 to run Dinov2 + ArcFace integration test.",
)
def test_dinov2_arcface_custom_compute_loss_matches_module_forward() -> None:
    pml_losses = pytest.importorskip("pytorch_metric_learning.losses")
    arcface_cls = pml_losses.ArcFaceLoss

    torch.manual_seed(0)

    model = AutoModel.from_pretrained("facebook/dinov2-small")
    model.eval()

    batch_size = 4
    num_classes = 8
    pixel_values = torch.rand(batch_size, 3, 224, 224)
    labels = torch.tensor([0, 3, 5, 7], dtype=torch.long)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        embeddings = outputs.last_hidden_state[:, 0, :]

    arcface = arcface_cls(
        num_classes=num_classes,
        embedding_size=embeddings.shape[1],
        margin=28.6,
        scale=64,
    )

    custom_loss, custom_logits = compute_loss_and_logits(
        arcface_loss=arcface,
        embeddings=embeddings,
        labels=labels,
    )
    module_loss = arcface(embeddings, labels)

    assert custom_logits.shape == (batch_size, num_classes)
    assert torch.isfinite(custom_logits).all()
    assert torch.isfinite(custom_loss)
    assert torch.isclose(custom_loss, module_loss, atol=1e-6), (
        f"custom_loss={custom_loss.item():.8f}, module_loss={module_loss.item():.8f}"
    )
