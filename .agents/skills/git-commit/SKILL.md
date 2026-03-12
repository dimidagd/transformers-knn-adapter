---
name: git-commit
description: Commit changes in this repository using Conventional Commits with clear staging and verification steps.
---

# git-commit

Use this skill when asked to create commits in this repository.

## Goal

Create clean, reviewable commits that follow this project's commit rules.

## Core rule

- Use atomic commits: one logical change per commit.
- Keep scope isolated between commits: do not mix unrelated changes.
- Every commit message must follow Conventional Commits.

## Commit format

Use Conventional Commits:

```text
<type>(<optional-scope>): <short imperative summary>
```

Examples:

- `feat(pipeline): add stratified sampling for eval`
- `fix(ci): run ruff before pytest`
- `docs(readme): add train and eval usage examples`

## Recommended workflow

1. Inspect current changes.
2. Group changes by logical unit.
3. Stage only files for one logical unit.
4. Verify staged diff contains one isolated scope.
5. Commit with a compliant Conventional Commit message.
6. Repeat for the next logical unit.

## Commands

### Inspect changes

```bash
git status --short
git diff
```

### Stage selected files

```bash
git add <file1> <file2>
```

### Verify staged content

```bash
git diff --staged
```

### Commit

```bash
git commit -m "docs(docs): add commit workflow skill"
```

## Scope guidance

Prefer scopes used in this repo when relevant:

- `pipeline`
- `knn`
- `tests`
- `ci`
- `packaging`
- `docs`

## Safety rules

- Do not commit unrelated modified files.
- Keep each commit atomic and scoped to one logical change.
- Never commit secrets (API keys, tokens, credentials, private keys, `.env`
  secrets, or other sensitive data).
- Keep subject line <= 72 chars.
- Use imperative mood (`add`, `fix`, `remove`), not past tense.
- Do not end commit subject with a period.
