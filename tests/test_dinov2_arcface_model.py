from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import Dinov2Config, Dinov2ForImageClassification, Dinov2Model
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


def test_get_embedding_size_from_model_conf_matches_hidden_size() -> None:
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

    assert embedding_size == 24


def test_calculate_embeddings_returns_pooler_output() -> None:
    pooled_output = torch.tensor([[1.0, 2.0]])
    embeddings = Dinov2ForImageClassificationWithArcFaceLoss.calculate_embeddings(pooled_output)

    assert embeddings.tolist() == [[1.0, 2.0]]


def test_extract_embeddings_from_image_classifier_output_uses_pooler_output() -> None:
    outputs = ImageClassifierOutput()
    outputs.pooler_output = torch.tensor([[1.0, 2.0]])

    embeddings = Dinov2ForImageClassificationWithArcFaceLoss.extract_embeddings_from_image_classifier_output(outputs)

    assert embeddings.tolist() == [[1.0, 2.0]]


def test_extract_embeddings_from_image_classifier_output_requires_post_layernorm_outputs() -> None:
    outputs = ImageClassifierOutput()

    with pytest.raises(ValueError, match="must expose pooler_output"):
        Dinov2ForImageClassificationWithArcFaceLoss.extract_embeddings_from_image_classifier_output(outputs)


def test_extract_embeddings_from_dinov2_for_image_classification_output_uses_post_layernorm_tokens() -> None:
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

    backbone_outputs = model.dinov2(pixel_values=pixel_values, output_hidden_states=True)
    embeddings = Dinov2ForImageClassificationWithArcFaceLoss.extract_embeddings_from_image_classifier_output(
        backbone_outputs
    )
    expected = Dinov2ForImageClassificationWithArcFaceLoss.calculate_embeddings(backbone_outputs.pooler_output)

    assert embeddings.shape == (2, 24)
    assert torch.allclose(embeddings, expected)


def test_dinov2_post_layernorm_outputs_match_cls_token_and_not_hidden_state_tail() -> None:
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
    pixel_values = torch.randn(2, 3, 32, 32)

    model = Dinov2Model(config)
    outputs = model(pixel_values=pixel_values, output_hidden_states=True)

    assert outputs.hidden_states is not None
    assert torch.allclose(outputs.pooler_output, outputs.last_hidden_state[:, 0, :])
    assert not torch.allclose(outputs.hidden_states[-1], outputs.last_hidden_state)
    assert not torch.allclose(outputs.hidden_states[-1][:, 0, :], outputs.pooler_output)


def test_checkpoint_loaded_dinov2_and_classifier_backbone_use_same_post_layernorm_embeddings(tmp_path) -> None:
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
    checkpoint_dir = tmp_path / "dinov2-shared"
    Dinov2ForImageClassification(config).save_pretrained(checkpoint_dir)

    raw_model = Dinov2Model.from_pretrained(checkpoint_dir)
    classifier_model = Dinov2ForImageClassification.from_pretrained(checkpoint_dir)
    pixel_values = torch.randn(2, 3, 32, 32)

    raw_outputs = raw_model(pixel_values=pixel_values, output_hidden_states=True)
    classifier_backbone_outputs = classifier_model.dinov2(
        pixel_values=pixel_values,
        output_hidden_states=True,
    )

    raw_embeddings = Dinov2ForImageClassificationWithArcFaceLoss.calculate_embeddings(raw_outputs.pooler_output)
    classifier_embeddings = Dinov2ForImageClassificationWithArcFaceLoss.calculate_embeddings(
        classifier_backbone_outputs.pooler_output
    )

    assert raw_outputs.hidden_states is not None
    assert classifier_backbone_outputs.hidden_states is not None
    assert torch.allclose(raw_outputs.pooler_output, raw_outputs.last_hidden_state[:, 0, :])
    assert torch.allclose(
        classifier_backbone_outputs.pooler_output,
        classifier_backbone_outputs.last_hidden_state[:, 0, :],
    )
    assert not torch.allclose(raw_outputs.hidden_states[-1], raw_outputs.last_hidden_state)
    assert not torch.allclose(
        classifier_backbone_outputs.hidden_states[-1],
        classifier_backbone_outputs.last_hidden_state,
    )
    assert torch.allclose(raw_embeddings, classifier_embeddings)


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
