from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import ImageClassifierOutput
from transformers.models.dinov2.modeling_dinov2 import Dinov2ForImageClassification


class Dinov2ForImageClassificationWithArcFaceLoss(Dinov2ForImageClassification):
    """Dinov2 image classifier with ArcFace logits/loss.

    This class preserves the public forward contract of
    ``Dinov2ForImageClassification`` while replacing the classifier loss path
    with an ArcFace objective from ``pytorch-metric-learning``.

    Extra ArcFace hyperparameters can be provided on ``config`` via the
    optional attributes ``arcface_margin`` and ``arcface_scale``.
    The classification loss can optionally be switched to sigmoid focal loss
    with ``use_focal_loss``, ``focal_loss_alpha``, and ``focal_loss_gamma``.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        embedding_size = self.get_embedding_size_from_model_conf(config)
        self.classifier = nn.Identity()
        self.embedding_source = self._resolve_embedding_source(config)
        self.arcface_margin = float(getattr(config, "arcface_margin", 28.6))
        self.arcface_scale = float(getattr(config, "arcface_scale", 64.0))
        self.use_focal_loss = bool(getattr(config, "use_focal_loss", False))
        self.focal_loss_alpha = getattr(config, "focal_loss_alpha", 0.25)
        self.focal_loss_gamma = float(getattr(config, "focal_loss_gamma", 2.0))
        self.arcface_loss = self._build_arcface_loss(
            num_classes=config.num_labels,
            embedding_size=embedding_size,
            margin=self.arcface_margin,
            scale=self.arcface_scale,
        )
        self._reset_arcface_weights()

    def _reset_arcface_weights(self) -> None:
        weight = getattr(self.arcface_loss, "W", None)
        if weight is not None:
            nn.init.normal_(weight)
        if weight is not None and not torch.isfinite(weight).all():
            raise ValueError("arcface_loss.W contains non-finite values after reinitialization.")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike | None, *model_args, **kwargs):
        output_loading_info = kwargs.pop("output_loading_info", False)
        model, loading_info = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            output_loading_info=True,
            **kwargs,
        )
        weight = getattr(model.arcface_loss, "W", None)
        missing_keys = set(loading_info.get("missing_keys", ()))
        if "arcface_loss.W" in missing_keys or weight is None or not torch.isfinite(weight).all():
            model._reset_arcface_weights()
        if output_loading_info:
            return model, loading_info
        return model

    @staticmethod
    def _build_arcface_loss(*, num_classes: int, embedding_size: int, margin: float, scale: float) -> nn.Module:
        try:
            from pytorch_metric_learning.losses import ArcFaceLoss
        except ImportError as exc:
            raise ImportError(
                "Dinov2ForImageClassificationWithArcFaceLoss requires "
                "`pytorch-metric-learning`. Install it with `uv add pytorch-metric-learning` "
                "or run with `uv run --with pytorch-metric-learning ...`."
            ) from exc

        if num_classes <= 0:
            raise ValueError("ArcFace classification requires config.num_labels > 0.")

        return ArcFaceLoss(
            num_classes=num_classes,
            embedding_size=embedding_size,
            margin=margin,
            scale=scale,
        )

    @staticmethod
    def _resolve_embedding_source(config: Any) -> str:
        embedding_source = str(getattr(config, "embedding_source", "cls_mean"))
        if embedding_source not in {"cls", "cls_mean"}:
            raise ValueError("embedding_source must be one of: cls, cls_mean.")
        return embedding_source

    @staticmethod
    def get_embedding_size_from_model_conf(config: Any) -> int:
        hidden_size = int(getattr(config, "hidden_size"))
        embedding_source = Dinov2ForImageClassificationWithArcFaceLoss._resolve_embedding_source(config)
        if embedding_source == "cls":
            return hidden_size
        return hidden_size * 2

    def _compute_classification_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if not self.use_focal_loss:
            return self.arcface_loss.cross_entropy(logits, labels).mean()

        try:
            from torchvision.ops import sigmoid_focal_loss
        except ImportError as exc:
            raise ImportError(
                "Sigmoid focal loss requires torchvision. Install it with "
                "`uv add torchvision` or run with `uv run --with torchvision ...`."
            ) from exc

        targets = F.one_hot(labels, num_classes=logits.shape[1]).to(dtype=logits.dtype)
        loss = sigmoid_focal_loss(
            inputs=logits,
            targets=targets,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="mean",
        )
        return loss

    def compute_arcface_loss_and_logits(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Return ArcFace loss and logits from embeddings.

        When ``labels`` are omitted, the logits are the scaled cosine scores.
        When ``labels`` are provided, the logits include the ArcFace margin on
        the target class before the cross-entropy loss is computed.
        """
        self.arcface_loss.cast_types(embeddings.dtype, embeddings.device)

        cosine = self.arcface_loss.get_cosine(embeddings)
        if labels is None:
            logits = self.arcface_loss.scale_logits(cosine, embeddings)
            return None, logits

        mask = self.arcface_loss.get_target_mask(embeddings, labels)
        cosine_of_targets = cosine[mask == 1]
        modified_targets = self.arcface_loss.modify_cosine_of_target_classes(cosine_of_targets)
        diff = (modified_targets - cosine_of_targets).unsqueeze(1)
        logits = cosine + (mask * diff)
        logits = self.arcface_loss.scale_logits(logits, embeddings)
        loss = self._compute_classification_loss(logits, labels)
        return loss, logits

    def compute_inference_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        self.arcface_loss.cast_types(embeddings.dtype, embeddings.device)
        cosine = self.arcface_loss.get_cosine(embeddings)
        return self.arcface_loss.scale_logits(cosine, embeddings)

    @staticmethod
    def calculate_embeddings(
        sequence_output: torch.Tensor,
        embedding_source: str = "cls_mean",
    ) -> torch.Tensor:
        cls_token = sequence_output[:, 0]
        if embedding_source == "cls":
            return cls_token
        if embedding_source != "cls_mean":
            raise ValueError("embedding_source must be one of: cls, cls_mean.")
        patch_tokens = sequence_output[:, 1:]
        if patch_tokens.shape[1] == 0:
            patch_mean = torch.zeros_like(cls_token)
        else:
            patch_mean = patch_tokens.mean(dim=1)
        return torch.cat([cls_token, patch_mean], dim=1)

    @staticmethod
    def extract_embeddings_from_image_classifier_output(
        model_output: ImageClassifierOutput,
        embedding_source: str = "cls_mean",
    ) -> torch.Tensor:
        hidden_states = model_output.hidden_states
        if hidden_states is None or len(hidden_states) == 0:
            raise ValueError(
                "ImageClassifierOutput.hidden_states must be populated to extract embeddings. "
                "Call the model with output_hidden_states=True."
            )
        sequence_output = hidden_states[-1]
        return Dinov2ForImageClassificationWithArcFaceLoss.calculate_embeddings(
            sequence_output,
            embedding_source=embedding_source,
        )

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ImageClassifierOutput:
        outputs = self.dinov2(pixel_values, **kwargs)

        sequence_output = outputs.last_hidden_state
        embeddings = self.calculate_embeddings(sequence_output, embedding_source=self.embedding_source)

        loss, _training_logits = self.compute_arcface_loss_and_logits(embeddings=embeddings, labels=labels)
        logits = self.compute_inference_logits(embeddings)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
