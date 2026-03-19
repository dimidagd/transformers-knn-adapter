from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import Dinov2Config, Trainer, TrainingArguments

from transformers_knn_adapter.dinov2_arcface import Dinov2ForImageClassificationWithArcFaceLoss


class _TinyImageDataset(TorchDataset):
    def __init__(self) -> None:
        self.samples = [
            {
                "pixel_values": torch.full((3, 32, 32), 0.1, dtype=torch.float32),
                "labels": torch.tensor(0, dtype=torch.long),
            },
            {
                "pixel_values": torch.full((3, 32, 32), 0.2, dtype=torch.float32),
                "labels": torch.tensor(1, dtype=torch.long),
            },
            {
                "pixel_values": torch.full((3, 32, 32), 0.3, dtype=torch.float32),
                "labels": torch.tensor(0, dtype=torch.long),
            },
            {
                "pixel_values": torch.full((3, 32, 32), 0.4, dtype=torch.float32),
                "labels": torch.tensor(1, dtype=torch.long),
            },
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


@pytest.mark.integration
def test_dinov2_arcface_trainer_runs_two_epochs_on_tiny_dataset(tmp_path) -> None:
    pytest.importorskip("accelerate", exc_type=ImportError)
    pytest.importorskip("pytorch_metric_learning.losses", exc_type=ImportError)

    config = Dinov2Config(
        image_size=32,
        patch_size=16,
        num_channels=3,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=48,
        num_labels=2,
    )
    model = Dinov2ForImageClassificationWithArcFaceLoss(config)
    train_dataset = _TinyImageDataset()
    training_args = TrainingArguments(
        output_dir=str(tmp_path / "trainer-output"),
        do_train=True,
        do_eval=False,
        per_device_train_batch_size=2,
        num_train_epochs=2.0,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        logging_strategy="no",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        use_cpu=True,
        disable_tqdm=True,
        seed=1234,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    result = trainer.train()

    assert result.training_loss >= 0.0
    assert trainer.state.epoch == pytest.approx(2.0)
    assert trainer.state.global_step == 4
