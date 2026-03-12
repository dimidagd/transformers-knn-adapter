---
name: test-runner
description: Run tests in this repository using uv and pytest, including targeted and full-suite test commands.
---

# test-runner

Use this skill when asked about testing.

## Goal

Execute the right test scope quickly, then expand only if needed.

## Prerequisites

- Run from repository root.
- Dependencies installed with `uv`.

## Commands

### Run a focused test file

```bash
uv run pytest tests/test_knn_image_pipeline.py -q
```

### Run full test suite

```bash
uv run pytest
```

### Run one specific test

```bash
uv run pytest tests/test_knn_image_pipeline.py -k "grid_search" -q
```

## Workflow

1. Start with the most targeted test(s) relevant to changed files.
2. If targeted tests pass, run the full suite.
3. Report command, pass/fail, and key failures.

## Reporting

- Always include the exact command executed.
- Summarize failures with test id and error headline.
- If tests were not run, state that explicitly.
