from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from transformers import TrainerCallback
from transformers.utils import logging

logger = logging.get_logger(__name__)


class FreezeScheduleCallback(TrainerCallback):
    """Apply model freeze/unfreeze schedule based on Trainer epoch progress."""

    def __init__(
        self,
        *,
        trainer: Any | None = None,
        metric_name: str = "train/trainable_parameters",
        freeze_schedule: Any | None = None,
    ) -> None:
        self.trainer = trainer
        self.metric_name = metric_name
        self.freeze_schedule = freeze_schedule

    @staticmethod
    def normalize_freeze_schedule(freeze_schedule: Any) -> list[dict[str, Any]]:
        if freeze_schedule in (None, ()):
            return []
        if not isinstance(freeze_schedule, Sequence) or isinstance(freeze_schedule, (str, bytes)):
            raise ValueError("freeze_schedule must be a sequence of rule dictionaries.")

        normalized_rules: list[dict[str, Any]] = []
        for raw_rule in freeze_schedule:
            if not isinstance(raw_rule, dict):
                raise ValueError("Each freeze_schedule entry must be a dictionary.")
            if "epoch" not in raw_rule:
                raise ValueError("Each freeze_schedule entry must include an 'epoch' key.")
            epoch = float(raw_rule["epoch"])
            freeze_modules = [str(module_name) for module_name in raw_rule.get("freeze_modules", ())]
            unfreeze_modules = [str(module_name) for module_name in raw_rule.get("unfreeze_modules", ())]
            normalized_rules.append(
                {
                    "epoch": epoch,
                    "freeze_modules": freeze_modules,
                    "unfreeze_modules": unfreeze_modules,
                }
            )
        normalized_rules.sort(key=lambda rule: rule["epoch"])
        return normalized_rules

    @staticmethod
    def count_trainable_parameters(model: Any) -> int:
        if not hasattr(model, "parameters"):
            return 0
        return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    @staticmethod
    def _set_module_trainable(model: Any, module_name: str, *, trainable: bool) -> None:
        if module_name in {"", "."}:
            target_module = model
        else:
            modules = dict(model.named_modules())
            target_module = modules.get(module_name)
            if target_module is None:
                raise ValueError(f"Unknown module name in freeze_schedule: {module_name!r}")
        for parameter in target_module.parameters():
            parameter.requires_grad = trainable

    @classmethod
    def apply_freeze_schedule(
        cls,
        model: Any,
        *,
        epoch: float,
        freeze_schedule: Any | None = None,
        initial_state_attr: str = "_freeze_schedule_initial_trainable_state",
    ) -> list[dict[str, Any]]:
        if freeze_schedule is None:
            freeze_schedule = getattr(getattr(model, "config", None), "freeze_schedule", ())

        normalized_rules = cls.normalize_freeze_schedule(freeze_schedule)
        if not normalized_rules:
            return []

        initial_state = getattr(model, initial_state_attr, None)
        if initial_state is None:
            initial_state = {
                name: parameter.requires_grad for name, parameter in model.named_parameters()
            }
            setattr(model, initial_state_attr, initial_state)

        for parameter_name, initial_requires_grad in initial_state.items():
            model.get_parameter(parameter_name).requires_grad = initial_requires_grad

        for rule in normalized_rules:
            if float(rule["epoch"]) > float(epoch):
                break
            for module_name in rule["freeze_modules"]:
                cls._set_module_trainable(model, module_name, trainable=False)
            for module_name in rule["unfreeze_modules"]:
                cls._set_module_trainable(model, module_name, trainable=True)
        return normalized_rules

    def _apply(self, model: Any, epoch: float) -> None:
        if model is None:
            return
        freeze_schedule = (
            self.freeze_schedule
            if self.freeze_schedule is not None
            else getattr(getattr(model, "config", None), "freeze_schedule", None)
        )
        if freeze_schedule in (None, ()):
            return

        self.apply_freeze_schedule(model, epoch=epoch, freeze_schedule=freeze_schedule)

        trainable_params = self.count_trainable_parameters(model)
        logger.info(
            "Applied freeze schedule at epoch %.4f. Trainable parameters: %d",
            epoch,
            trainable_params,
        )
        if self.trainer is not None:
            self.trainer.log({self.metric_name: float(trainable_params)})

    def on_train_begin(self, args, state, control, **kwargs):  # type: ignore[override]
        del args
        model = kwargs.get("model")
        self._apply(model, epoch=0.0)
        return control

    def on_epoch_begin(self, args, state, control, **kwargs):  # type: ignore[override]
        del args
        model = kwargs.get("model")
        current_epoch = 0.0 if state.epoch is None else float(state.epoch)
        self._apply(model, epoch=current_epoch)
        return control
