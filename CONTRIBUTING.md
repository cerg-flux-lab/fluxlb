# Contributing to fluxlb

Thanks for your interest in contributing. This guide covers how to set up, the rules that keep the codebase differentiable and quantum-ready, and how to get a change merged.

## Getting started

```bash
git clone https://github.com/cerg-flux-lab/fluxlb.git fluxlb
cd fluxlb
pip install -e ".[dev]"
pytest -m smoke      # confirm your environment
```

## The one rule that matters most

**Streaming and collision stay strictly separate. Collision is a swappable module behind `fluxlb/core/collision/base.py`.** That seam is what keeps the solver differentiable and lets ML and quantum collision operators drop in. Do not fuse streaming into collision, do not bypass the `CollisionOperator` interface, and do not change its signature without opening an issue to discuss it first. New collision operators (classical, learned, or quantum) implement that interface.

## Development workflow

1. Open an issue (or comment on an existing one) before large work, so effort is not duplicated.
2. Branch from `main`: `feature/<short-name>` or `fix/<short-name>`.
3. Make focused commits with clear messages.
4. Ensure the checks below pass locally.
5. Open a pull request describing what changed and why. Link the issue.

## Checks that must pass

```bash
pytest                          # tests, including regression vs analytical solutions
ruff check . && ruff format .   # lint and format
mypy fluxlb                   # type check
```

Add tests with your change. In particular:

- New collision operator: include a mass and momentum conservation test.
- New differentiable path: include a gradient check against finite differences.
- New physics: add or extend a regression test against an analytical solution (e.g. Poiseuille, Taylor-Green).
- New ML model family: add a tiny-overfit smoke test (`@pytest.mark.smoke`).

## Code style

- Type hints on public functions.
- UK English in docstrings, comments and prose.
- Docstrings explain the physics or intent, not just the signature.
- Keep tensor operations autograd-safe: avoid in-place ops on tensors that require grad, and do not call `.item()`, `.numpy()`, or `.detach()` inside a differentiable path.
- Respect a single device/dtype context; register lattice constants as buffers.
- Confine Qiskit to `fluxlb/quantum/`. Do not add heavy dependencies without discussion.

## Compute notes

The reference cluster uses single, memory-limited GPUs (16 GB) on standalone nodes. Do not assume multi-node distributed training. Prefer gradient checkpointing, mixed precision, and streaming data over raising resource requirements.

## Scientific contributions and AI assistance

This lab follows an Intellectual Ownership Framework: AI is an amplifier, never the origin, and the governing test is whether you can independently defend your contribution. Practically:

- Pull requests that introduce scientific logic (loss and residual formulations, boundary conditions, architecture rationale, physical reasoning) must explain the reasoning, not just the code. Reviewers will ask.
- If a contribution is AI-assisted in a material way, say so in the PR, and make sure you understand and can defend it.
- Cite sources for methods and choices where relevant.

## Reporting issues

Use the issue templates. For bugs, include a minimal reproduction, the expected vs actual behaviour, and your environment (OS, Python, PyTorch, GPU).

## Licence

Contributions are made under the repository licence (APACHE-2.0). By contributing you agree your work will be released under that licence.
