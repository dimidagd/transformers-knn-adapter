# AGENTS.md

## Commit Convention

Use [Conventional Commits](https://www.conventionalcommits.org/) for all commits and PR titles.

Required format:

```text
<type>(<optional-scope>): <short imperative summary>
```

Examples:

- `feat(pipeline): add stratified sampling for eval`
- `fix(ci): run ruff before pytest`
- `docs(readme): add train and eval usage examples`
- `test(knn): cover grid-search parameter validation`
- `chore(deps): bump transformers to latest compatible version`

## Allowed Types

- `feat`: new user-facing functionality
- `fix`: bug fix
- `docs`: documentation-only changes
- `test`: add/update tests
- `refactor`: code restructuring without behavior change
- `perf`: performance improvement
- `build`: packaging/build system changes
- `ci`: CI/CD workflow changes
- `chore`: maintenance that does not fit above

## Breaking Changes

For breaking changes, use either:

- `feat!:` / `fix!:` syntax, or
- `BREAKING CHANGE:` in the commit body

Example:

- `feat!: rename pipeline factory arguments`

## Scope Guidance

Prefer one of these scopes when relevant:

- `pipeline`
- `knn`
- `tests`
- `ci`
- `packaging`
- `docs`

## Additional Rules

- Keep subject line <= 72 characters.
- Use imperative mood (`add`, `fix`, `remove`), not past tense.
- Do not end subject line with a period.
- Squash/fixup local commits before merge when practical.
- When executing code, prefer showing progress when feasible. For Python, use
  `tqdm` (or equivalent) to expose task progress.

## Why

This repository uses automated release/versioning workflows. Conventional Commits keep changelogs and semantic version bumps predictable.

## Build Wheel Locally

Use `uv` to build the wheel from the repository root:

```bash
uv build --wheel
```

The built wheel is written to `dist/`.

## Skills

Repository-local skills are stored under `.agents/skills/`.

### Available skills

- `knn-cli-training`: Run KNN training from the CLI using either a Hugging Face
  dataset ID or a local `imagefolder` path, including validation via `eval`.
  (file: `.agents/skills/knn-cli-training/SKILL.md`)
- `test-runner`: Run repository tests with `uv` including targeted and full
  pytest commands.
  (file: `.agents/skills/test-runner/SKILL.md`)
- `git-commit`: Commit project changes with Conventional Commit messages and a
  clean staged diff.
  (file: `.agents/skills/git-commit/SKILL.md`)
