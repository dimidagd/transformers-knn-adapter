from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import Dinov2Config, Dinov2ForImageClassification
from transformers.modeling_outputs import ImageClassifierOutput
from transformers.trainer_callback import TrainerControl, TrainerState

from transformers_knn_adapter.dinov2_arcface import Dinov2ForImageClassificationWithArcFaceLoss
from transformers_knn_adapter.freeze_schedule_callback import FreezeScheduleCallback


def _require_pml() -> None:
    pytest.importorskip("pytorch_metric_learning.losses", exc_type=ImportError)


def _count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class _GenericFreezeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.head = torch.nn.Linear(4, 2)
        self.config = SimpleNamespace(
            freeze_schedule=[
                {"epoch": 0.0, "freeze_modules": ["backbone"], "unfreeze_modules": []},
                {"epoch": 1.0, "freeze_modules": [], "unfreeze_modules": ["backbone"]},
            ]
        )


def test_dinov2_arcface_model_matches_image_classifier_contract() -> None:
    _require_pml()

    config = Dinov2Config(
        image_size=32,
        patch_size=16,
        num_channels=3,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=48,
        num_labels=5,
    )
    config.arcface_margin = 28.6
    config.arcface_scale = 64.0

    model = Dinov2ForImageClassificationWithArcFaceLoss(config)
    pixel_values = torch.randn(3, 3, 32, 32)
    labels = torch.tensor([0, 1, 4], dtype=torch.long)

    outputs = model(pixel_values=pixel_values, labels=labels)

    assert isinstance(outputs, ImageClassifierOutput)
    assert outputs.loss is not None
    assert outputs.logits.shape == (3, 5)
    assert torch.isfinite(outputs.loss)
    assert torch.isfinite(outputs.logits).all()


def test_dinov2_arcface_model_returns_scaled_cosine_logits_without_labels() -> None:
    _require_pml()

    config = Dinov2Config(
        image_size=32,
        patch_size=16,
        num_channels=3,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=48,
        num_labels=4,
    )

    model = Dinov2ForImageClassificationWithArcFaceLoss(config)
    pixel_values = torch.randn(2, 3, 32, 32)

    outputs = model(pixel_values=pixel_values)

    assert outputs.loss is None
    assert outputs.logits.shape == (2, 4)
    assert torch.isfinite(outputs.logits).all()


def test_get_embedding_size_from_model_conf_matches_cls_plus_patch_mean() -> None:
    config = Dinov2Config(
        image_size=32,
        patch_size=16,
        num_channels=3,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=48,
        num_labels=4,
    )

    embedding_size = Dinov2ForImageClassificationWithArcFaceLoss.get_embedding_size_from_model_conf(config)

    assert embedding_size == 48


def test_get_embedding_size_from_model_conf_supports_cls_only() -> None:
    config = Dinov2Config(
        image_size=32,
        patch_size=16,
        num_channels=3,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=48,
        num_labels=4,
    )
    config.embedding_source = "cls"

    embedding_size = Dinov2ForImageClassificationWithArcFaceLoss.get_embedding_size_from_model_conf(config)

    assert embedding_size == 24


def test_calculate_embeddings_uses_cls_plus_mean_patch_tokens() -> None:
    sequence_output = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        ]
    )

    embeddings = Dinov2ForImageClassificationWithArcFaceLoss.calculate_embeddings(sequence_output)

    assert embeddings.tolist() == [[1.0, 2.0, 4.0, 5.0]]


def test_calculate_embeddings_supports_cls_only() -> None:
    sequence_output = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        ]
    )

    embeddings = Dinov2ForImageClassificationWithArcFaceLoss.calculate_embeddings(
        sequence_output,
        embedding_source="cls",
    )

    assert embeddings.tolist() == [[1.0, 2.0]]


def test_extract_embeddings_from_image_classifier_output_uses_last_hidden_state_entry() -> None:
    sequence_output = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        ]
    )
    outputs = ImageClassifierOutput(hidden_states=(torch.zeros_like(sequence_output), sequence_output))

    embeddings = Dinov2ForImageClassificationWithArcFaceLoss.extract_embeddings_from_image_classifier_output(outputs)

    assert embeddings.tolist() == [[1.0, 2.0, 4.0, 5.0]]


def test_extract_embeddings_from_image_classifier_output_supports_cls_only() -> None:
    sequence_output = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        ]
    )
    outputs = ImageClassifierOutput(hidden_states=(torch.zeros_like(sequence_output), sequence_output))

    embeddings = Dinov2ForImageClassificationWithArcFaceLoss.extract_embeddings_from_image_classifier_output(
        outputs,
        embedding_source="cls",
    )

    assert embeddings.tolist() == [[1.0, 2.0]]


def test_extract_embeddings_from_image_classifier_output_requires_hidden_states() -> None:
    outputs = ImageClassifierOutput()

    with pytest.raises(ValueError, match="output_hidden_states=True"):
        Dinov2ForImageClassificationWithArcFaceLoss.extract_embeddings_from_image_classifier_output(outputs)


def test_extract_embeddings_from_dinov2_for_image_classification_output() -> None:
    config = Dinov2Config(
        image_size=32,
        patch_size=16,
        num_channels=3,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=48,
        num_labels=4,
    )
    model = Dinov2ForImageClassification(config)
    pixel_values = torch.randn(2, 3, 32, 32)

    outputs = model(pixel_values=pixel_values, output_hidden_states=True)
    embeddings = Dinov2ForImageClassificationWithArcFaceLoss.extract_embeddings_from_image_classifier_output(outputs)
    expected = Dinov2ForImageClassificationWithArcFaceLoss.calculate_embeddings(outputs.hidden_states[-1])

    assert outputs.hidden_states is not None
    assert embeddings.shape == (2, 48)
    assert torch.allclose(embeddings, expected)


def test_model_supports_cls_embedding_source() -> None:
    _require_pml()

    config = Dinov2Config(
        image_size=32,
        patch_size=16,
        num_channels=3,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=48,
        num_labels=4,
    )
    config.embedding_source = "cls"

    model = Dinov2ForImageClassificationWithArcFaceLoss(config)
    pixel_values = torch.randn(2, 3, 32, 32)

    outputs = model(pixel_values=pixel_values)

    assert outputs.logits.shape == (2, 4)
    assert model.arcface_loss.W.shape == (24, 4)


def test_compute_arcface_loss_and_logits_matches_arcface_module_forward() -> None:
    _require_pml()

    config = Dinov2Config(
        image_size=32,
        patch_size=16,
        num_channels=3,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=48,
        num_labels=4,
    )

    model = Dinov2ForImageClassificationWithArcFaceLoss(config)
    embeddings = torch.randn(3, model.get_embedding_size_from_model_conf(config))
    labels = torch.tensor([0, 1, 3], dtype=torch.long)

    loss, logits = model.compute_arcface_loss_and_logits(embeddings=embeddings, labels=labels)
    module_loss = model.arcface_loss(embeddings, labels)

    assert loss is not None
    assert logits.shape == (3, 4)
    assert torch.isfinite(logits).all()
    assert torch.isclose(loss, module_loss, atol=1e-6), (
        f"loss={loss.item():.8f}, module_loss={module_loss.item():.8f}"
    )


def test_compute_arcface_loss_and_logits_supports_sigmoid_focal_loss() -> None:
    _require_pml()
    pytest.importorskip("torchvision.ops", exc_type=ImportError)
    from torchvision.ops import sigmoid_focal_loss

    config = Dinov2Config(
        image_size=32,
        patch_size=16,
        num_channels=3,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=48,
        num_labels=4,
    )
    config.use_focal_loss = True
    config.focal_loss_alpha = 0.25
    config.focal_loss_gamma = 2.0

    model = Dinov2ForImageClassificationWithArcFaceLoss(config)
    embeddings = torch.randn(3, model.get_embedding_size_from_model_conf(config))
    labels = torch.tensor([0, 1, 3], dtype=torch.long)

    loss, logits = model.compute_arcface_loss_and_logits(embeddings=embeddings, labels=labels)
    expected = sigmoid_focal_loss(
        inputs=logits,
        targets=F.one_hot(labels, num_classes=config.num_labels).to(dtype=logits.dtype),
        alpha=config.focal_loss_alpha,
        gamma=config.focal_loss_gamma,
        reduction="mean",
    )

    assert loss is not None
    assert torch.isfinite(loss)
    assert torch.isclose(loss, expected, atol=1e-6), (
        f"loss={loss.item():.8f}, expected={expected.item():.8f}"
    )


def test_from_pretrained_reinitializes_arcface_weights_after_load(tmp_path) -> None:
    _require_pml()

    base_config = Dinov2Config(
        image_size=32,
        patch_size=16,
        num_channels=3,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=48,
        num_labels=4,
    )
    base_model = Dinov2ForImageClassification(base_config)
    checkpoint_dir = tmp_path / "dinov2-base"
    base_model.save_pretrained(checkpoint_dir)

    arcface_config = Dinov2Config.from_pretrained(checkpoint_dir)
    arcface_config.num_labels = 4

    model = Dinov2ForImageClassificationWithArcFaceLoss.from_pretrained(
        checkpoint_dir,
        config=arcface_config,
        ignore_mismatched_sizes=True,
    )

    assert torch.isfinite(model.arcface_loss.W).all()


def test_from_pretrained_preserves_finite_arcface_weights_from_checkpoint(tmp_path) -> None:
    _require_pml()

    config = Dinov2Config(
        image_size=32,
        patch_size=16,
        num_channels=3,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=48,
        num_labels=4,
    )
    model = Dinov2ForImageClassificationWithArcFaceLoss(config)
    original_weight = model.arcface_loss.W.detach().clone()
    checkpoint_dir = tmp_path / "arcface-model"
    model.save_pretrained(checkpoint_dir)

    reloaded = Dinov2ForImageClassificationWithArcFaceLoss.from_pretrained(checkpoint_dir)

    assert torch.isfinite(reloaded.arcface_loss.W).all()
    assert torch.allclose(reloaded.arcface_loss.W, original_weight)


def test_freeze_schedule_freezes_and_unfreezes_modules_by_epoch() -> None:
    _require_pml()

    config = Dinov2Config(
        image_size=32,
        patch_size=16,
        num_channels=3,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=48,
        num_labels=4,
    )
    config.freeze_schedule = [
        {"epoch": 0.0, "freeze_modules": ["dinov2"], "unfreeze_modules": []},
        {"epoch": 1.0, "freeze_modules": [], "unfreeze_modules": ["dinov2"]},
    ]

    model = Dinov2ForImageClassificationWithArcFaceLoss(config)

    FreezeScheduleCallback.apply_freeze_schedule(model, epoch=0.0)
    assert all(not parameter.requires_grad for parameter in model.dinov2.parameters())
    assert all(parameter.requires_grad for parameter in model.arcface_loss.parameters())
    frozen_trainable_params = _count_trainable_parameters(model)
    assert frozen_trainable_params > 0

    FreezeScheduleCallback.apply_freeze_schedule(model, epoch=1.0)
    assert all(parameter.requires_grad for parameter in model.dinov2.parameters())
    assert all(parameter.requires_grad for parameter in model.arcface_loss.parameters())
    assert _count_trainable_parameters(model) > frozen_trainable_params


def test_freeze_schedule_callback_applies_model_schedule() -> None:
    _require_pml()

    config = Dinov2Config(
        image_size=32,
        patch_size=16,
        num_channels=3,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=48,
        num_labels=4,
    )
    config.freeze_schedule = [
        {"epoch": 0.0, "freeze_modules": ["dinov2"], "unfreeze_modules": []},
        {"epoch": 2.0, "freeze_modules": [], "unfreeze_modules": ["dinov2"]},
    ]

    model = Dinov2ForImageClassificationWithArcFaceLoss(config)
    logged_metrics: list[dict[str, float]] = []
    trainer = SimpleNamespace(log=lambda payload: logged_metrics.append(payload))
    callback = FreezeScheduleCallback(trainer=trainer)

    callback.on_train_begin(None, TrainerState(), TrainerControl(), model=model)
    assert all(not parameter.requires_grad for parameter in model.dinov2.parameters())
    assert logged_metrics
    assert "train/trainable_parameters" in logged_metrics[-1]

    state = TrainerState(epoch=2.0)
    callback.on_epoch_begin(None, state, TrainerControl(), model=model)
    assert all(parameter.requires_grad for parameter in model.dinov2.parameters())
    assert "train/trainable_parameters" in logged_metrics[-1]


def test_freeze_schedule_callback_applies_to_generic_model_class() -> None:
    model = _GenericFreezeModel()
    callback = FreezeScheduleCallback()

    callback.on_train_begin(None, TrainerState(), TrainerControl(), model=model)
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())

    callback.on_epoch_begin(None, TrainerState(epoch=1.0), TrainerControl(), model=model)
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())


def test_freeze_schedule_callback_accepts_inline_schedule_without_model_config() -> None:
    model = _GenericFreezeModel()
    model.config.freeze_schedule = None

    callback = FreezeScheduleCallback(
        freeze_schedule=[
            {"epoch": 0.0, "freeze_modules": ["backbone"], "unfreeze_modules": []},
            {"epoch": 1.0, "freeze_modules": [], "unfreeze_modules": ["backbone"]},
        ]
    )

    callback.on_train_begin(None, TrainerState(), TrainerControl(), model=model)
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())

    callback.on_epoch_begin(None, TrainerState(epoch=1.0), TrainerControl(), model=model)
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())


def test_freeze_schedule_rejects_unknown_module_names() -> None:
    _require_pml()

    config = Dinov2Config(
        image_size=32,
        patch_size=16,
        num_channels=3,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=48,
        num_labels=4,
    )
    config.freeze_schedule = [
        {"epoch": 0.0, "freeze_modules": ["does_not_exist"], "unfreeze_modules": []},
    ]

    model = Dinov2ForImageClassificationWithArcFaceLoss(config)

    with pytest.raises(ValueError, match="Unknown module name"):
        FreezeScheduleCallback.apply_freeze_schedule(model, epoch=0.0)


def test_full_model_freeze_yields_zero_trainable_parameters() -> None:
    _require_pml()

    config = Dinov2Config(
        image_size=32,
        patch_size=16,
        num_channels=3,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=48,
        num_labels=4,
    )
    config.freeze_schedule = [
        {"epoch": 0.0, "freeze_modules": ["."], "unfreeze_modules": []},
    ]

    model = Dinov2ForImageClassificationWithArcFaceLoss(config)
    assert _count_trainable_parameters(model) > 0

    FreezeScheduleCallback.apply_freeze_schedule(model, epoch=0.0)
    assert _count_trainable_parameters(model) == 0
