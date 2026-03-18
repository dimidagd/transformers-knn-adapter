from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np

from tests._test_utils import build_local_imagefolder_dataset

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "dogfaces_smoke.py"
SPEC = importlib.util.spec_from_file_location("dogfaces_smoke", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
DOGFACES_SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOGFACES_SMOKE)


def test_load_train_eval_datasets_uses_stratified_70_30_split(tmp_path: Path) -> None:
    rng = np.random.default_rng(1234)
    dataset_dir = build_local_imagefolder_dataset(tmp_path / "imagefolder", rng, samples_per_class=10)

    train_dataset, eval_dataset = DOGFACES_SMOKE.load_train_eval_datasets(
        dataset_name=str(dataset_dir),
        base_split="train",
        train_fraction=0.7,
        shuffle_seed=42,
        train_samples=None,
        eval_samples=None,
    )

    assert len(train_dataset) == 21
    assert len(eval_dataset) == 9
    assert Counter(train_dataset["label"]) == Counter({0: 7, 1: 7, 2: 7})
    assert Counter(eval_dataset["label"]) == Counter({0: 3, 1: 3, 2: 3})


def test_load_freeze_schedule_config_reads_json_object(tmp_path: Path) -> None:
    config_path = tmp_path / "freeze_schedule.json"
    expected = [
        {"epoch": 0.0, "freeze_modules": ["dinov2"], "unfreeze_modules": []},
        {"epoch": 1.0, "freeze_modules": [], "unfreeze_modules": ["dinov2"]},
    ]
    config_path.write_text(json.dumps({"freeze_schedule": expected}))

    loaded = DOGFACES_SMOKE.load_freeze_schedule_config(config_path)

    assert loaded == expected


def test_build_parser_accepts_wandb_reporting_args() -> None:
    args = DOGFACES_SMOKE.build_parser().parse_args(
        [
            "--report-to",
            "wandb",
            "--run-name",
            "smoke-run",
            "--wandb-project",
            "dogfaces",
            "--wandb-entity",
            "team-name",
            "--wandb-name",
            "trial-1",
            "--wandb-tags",
            "smoke",
            "arcface",
        ]
    )

    assert list(args.report_to) == ["wandb"]
    assert args.run_name == "smoke-run"
    assert args.wandb_project == "dogfaces"
    assert args.wandb_entity == "team-name"
    assert args.wandb_name == "trial-1"
    assert args.wandb_tags == ["smoke", "arcface"]
