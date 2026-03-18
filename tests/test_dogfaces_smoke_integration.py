from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tests._test_utils import build_local_dinov2_checkpoint, build_local_imagefolder_dataset

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "dogfaces_smoke.py"
SPEC = importlib.util.spec_from_file_location("dogfaces_smoke", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
DOGFACES_SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOGFACES_SMOKE)


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_DOGFACES_SMOKE_INTEGRATION") != "1",
    reason="Set RUN_DOGFACES_SMOKE_INTEGRATION=1 to run the local Dinov2 smoke integration test.",
)
def test_dogfaces_smoke_with_local_dinov2_and_imagefolder(tmp_path: Path) -> None:
    pytest.importorskip("accelerate", exc_type=ImportError)
    pytest.importorskip("pytorch_metric_learning.losses", exc_type=ImportError)

    rng = np.random.default_rng(1234)
    dataset_dir = build_local_imagefolder_dataset(tmp_path / "imagefolder", rng, samples_per_class=3)
    model_dir = build_local_dinov2_checkpoint(tmp_path / "tiny-dinov2")
    output_dir = tmp_path / "run-output"
    freeze_schedule_config = tmp_path / "freeze_schedule.json"
    freeze_schedule_config.write_text(
        json.dumps(
            {
                "freeze_schedule": [
                    {"epoch": 0.0, "freeze_modules": ["dinov2"], "unfreeze_modules": []},
                    {"epoch": 1.0, "freeze_modules": [], "unfreeze_modules": ["dinov2"]},
                ]
            }
        )
    )

    args = DOGFACES_SMOKE.build_parser().parse_args(
        [
            "--dataset",
            str(dataset_dir),
            "--model",
            str(model_dir),
            "--output-dir",
            str(output_dir),
            "--train-samples",
            "4",
            "--eval-samples",
            "2",
            "--per-device-train-batch-size",
            "2",
            "--per-device-eval-batch-size",
            "2",
            "--max-steps",
            "1",
            "--eval-steps",
            "1",
            "--logging-steps",
            "1",
            "--freeze-schedule-config",
            str(freeze_schedule_config),
            "--use-cpu",
        ]
    )

    summary = DOGFACES_SMOKE.run(args)

    assert summary["train_samples"] == 4
    assert summary["eval_samples"] == 2
    assert summary["saved_model_class"] == "Dinov2ForImageClassificationWithArcFaceLoss"
    assert summary["auto_model_reload_class"] == "Dinov2Model"
    assert summary["freeze_schedule"] == [
        {"epoch": 0.0, "freeze_modules": ["dinov2"], "unfreeze_modules": []},
        {"epoch": 1.0, "freeze_modules": [], "unfreeze_modules": ["dinov2"]},
    ]
    assert summary["use_focal_loss"] is False
    assert summary["focal_loss_alpha"] == 0.25
    assert summary["focal_loss_gamma"] == 2.0
    assert "eval_knn_1/f1/macro" in summary["pretrain_eval_metrics"]
    assert "eval_knn_1/recall_at_1/macro" in summary["pretrain_eval_metrics"]
    assert "eval_knn_1/mrr/macro" in summary["pretrain_eval_metrics"]
    assert "eval_knn_1/f1/macro" in summary["eval_metrics"]
    assert "eval_knn_1/recall_at_1/macro" in summary["eval_metrics"]
    assert "eval_knn_1/mrr/macro" in summary["eval_metrics"]
    assert summary["feature_extraction_probe"]["status"] == "ok"
    assert summary["feature_extraction_probe"]["output_shape"] == [1, 197, 32]
    assert summary["custom_pipeline_probe"]["status"] == "ok"
    assert summary["saved_model_custom_pipeline_probe"]["status"] == "ok"
    assert Path(summary["output_dir"]).is_dir()
    assert Path(summary["auto_model_export_dir"]).is_dir()


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_DOGFACES_SMOKE_INTEGRATION") != "1",
    reason="Set RUN_DOGFACES_SMOKE_INTEGRATION=1 to run the local Dinov2 smoke integration test.",
)
def test_dogfaces_smoke_can_disable_knn_callback(tmp_path: Path) -> None:
    pytest.importorskip("accelerate", exc_type=ImportError)
    pytest.importorskip("pytorch_metric_learning.losses", exc_type=ImportError)

    rng = np.random.default_rng(1234)
    dataset_dir = build_local_imagefolder_dataset(tmp_path / "imagefolder", rng, samples_per_class=3)
    model_dir = build_local_dinov2_checkpoint(tmp_path / "tiny-dinov2")
    output_dir = tmp_path / "run-output"

    args = DOGFACES_SMOKE.build_parser().parse_args(
        [
            "--dataset",
            str(dataset_dir),
            "--model",
            str(model_dir),
            "--output-dir",
            str(output_dir),
            "--train-samples",
            "4",
            "--eval-samples",
            "2",
            "--per-device-train-batch-size",
            "2",
            "--per-device-eval-batch-size",
            "2",
            "--max-steps",
            "1",
            "--eval-steps",
            "1",
            "--logging-steps",
            "1",
            "--disable-knn-callback",
            "--use-cpu",
        ]
    )

    summary = DOGFACES_SMOKE.run(args)

    assert summary["knn_callback_enabled"] is False
    assert summary["use_focal_loss"] is False
    assert "eval_knn_1/f1/macro" not in summary["pretrain_eval_metrics"]
    assert "eval_knn_1/recall_at_1/macro" not in summary["pretrain_eval_metrics"]
    assert "eval_knn_1/mrr/macro" not in summary["pretrain_eval_metrics"]
    assert "eval_knn_1/f1/macro" not in summary["eval_metrics"]
    assert "eval_knn_1/recall_at_1/macro" not in summary["eval_metrics"]
    assert "eval_knn_1/mrr/macro" not in summary["eval_metrics"]
