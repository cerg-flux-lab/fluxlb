# Project setup

A runbook for standing the repository up from the scaffold files and creating the board (labels, milestones, and all 77 issues with sub-tasks). Steps are ordered; run them from the repository root unless noted.

## 0. Prerequisites

- `git` and Python 3.12+
- GitHub CLI (`gh`), authenticated: `gh auth login`
- Only if you will use a Projects (v2) board: `gh auth refresh -s project`

## 1. Create the repository and place the files

From the folder holding the downloaded scaffold files:

```bash
git init
mkdir -p docs scripts .github/ISSUE_TEMPLATE

# root files (skip any already at root)
mv README.md CLAUDE.md CONTRIBUTING.md CITATION.cff .

# docs, scripts, templates
mv lbm-sciml-roadmap.md issue-checklists.md docs/
mv create-lbm-sciml-board.sh scripts/
mv research_task.yml bug_report.yml feature_request.yml config.yml .github/ISSUE_TEMPLATE/

chmod +x scripts/create-lbm-sciml-board.sh
```

Target layout:

```
(root)  README.md  CLAUDE.md  CONTRIBUTING.md  CITATION.cff  LICENSE
docs/   lbm-sciml-roadmap.md  issue-checklists.md  setup.md
scripts/ create-lbm-sciml-board.sh
.github/ISSUE_TEMPLATE/ research_task.yml  bug_report.yml  feature_request.yml  config.yml
```

## 2. Licence and placeholder fills

- Done: `LICENSE` (Apache-2.0, copyright 2026 Muaaz Bhamjee), `NOTICE`, `CITATION.cff` licence field, and the README Licence section.

## 3. First commit and push

Choose `--private` while building, `--public` when ready.

```bash
git add .
git commit -m "Initial scaffold"
gh repo create cerg-flux-lab/fluxlb \
  --private --source=. --remote=origin --push \
  --description "Differentiable, quantum-ready Lattice Boltzmann in PyTorch"
```

## 4. Python project

The package build config is yours to write (see the README extras: `[dev]`, `[sciml]`, `[quantum]`). Once `pyproject.toml` and the package skeleton exist:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -m smoke
```

## 5. Create the board and issues

The script creates labels, the six milestones, and all 77 issues, appending each issue's sub-tasks from `docs/issue-checklists.md`.

1. (Optional) Create a Projects v2 board and note its number:
   ```bash
   gh project create --owner cerg-flux-lab --title "fluxlb roadmap"
   gh project list --owner cerg-flux-lab      # read off the number
   ```
2. Edit the top of `scripts/create-lbm-sciml-board.sh`: set `OWNER`, `REPO`, and `PROJECT_NUMBER` (leave `PROJECT_NUMBER` empty to skip adding issues to a board).
3. Preview without writing anything:
   ```bash
   DRY_RUN=1 ./scripts/create-lbm-sciml-board.sh
   ```
4. Create for real:
   ```bash
   ./scripts/create-lbm-sciml-board.sh
   ```

Run it from the repository root so the default `CHECKLISTS_FILE=docs/issue-checklists.md` resolves. If your checklist file lives elsewhere, pass `CHECKLISTS_FILE=path ./scripts/create-lbm-sciml-board.sh`.

## Notes

- The script is re-runnable: it skips labels, milestones, and issues that already exist (issues matched by their `[E?.?]` key).
- If `docs/issue-checklists.md` is missing, issues are still created, just without the appended sub-tasks (the script warns once).
- Dependencies are written into issue bodies as text, not native "blocked by" relations; add those in the Project UI or via GraphQL if you want the hard links.
- Effort and epic sit in each issue body; priority is a label. To promote them to Projects v2 custom fields instead, that needs a GraphQL step.
- Compute conventions (single 16 GB GPU, standalone nodes, no multi-node training) are in `CLAUDE.md`.

## Next (yours to add)

`pyproject.toml`, the `fluxlb/` package skeleton, CI, and optionally a pre-commit config and PR template.
