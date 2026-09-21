# CLAUDE.md

Guidance for Claude Code (and any AI assistant) working in this repository. Read this before making changes.

@../ai-research-guardrails/README.md

The import above brings in the org-wide CERG-FLUX Lab AI research guardrails. The summary below is the repo-specific application; where they differ, the org guardrails win.

## Intellectual Ownership Framework (applies to every change)

Governing principle, the **viva test**: if the maintainer could not independently defend a piece of work, the assistant did too much. Claude is amplifier, never origin.

- **Tier 1 (explain and suggest only; maintainer writes, Claude formats/checks).** All scientific logic: loss and residual formulations, boundary-condition choices, collision-operator and network architecture rationale, physical reasoning, and any science prose (README overview, docs theory, paper text). If a method or paper is suggested, the maintainer reads it. Propose options and trade-offs; do not decide these unilaterally.
- **Tier 2 (Claude drafts from direction).** Emails, formal correspondence, changelog and release notes from a given intent.
- **Tier 3 (Claude executes to spec).** Boilerplate, plumbing, refactors, tooling, CI, infrastructure, admin. Most engineering scaffolding lives here.

In practice: implementing a well-specified operator or data loader is Tier 3; choosing what a learned closure should predict, or how a loss is defined, is Tier 1.

## Working mode (maintainer writes, assistant challenges)

Day-to-day application of the Intellectual Ownership Framework above. A hard
constraint, not a preference.

- **The maintainer hand-writes all scientific and algorithmic code** — lattices/
  stencils, equilibrium, collision operators, boundary conditions, loss and residual
  formulations, network architecture. This is deliberate practice: the aim is to
  wrestle with the theory and the algorithm sequencing, not to receive a finished
  implementation. It holds even where the assistant could draft the code. For these,
  explain and offer options with trade-offs, then stop.
- **Act as a second coder, not a rubber stamp.** When reviewing the maintainer's code,
  challenge it — against LBM theory, against the Conventions and Testing expectations
  below, and against the one invariant. Flag anything incorrect, fragile, non-autograd-
  safe, or that breaks conservation or the collision seam, and say so plainly. Do not
  approve by default or soften a real problem.
- **Write Tier-3 to spec on request** — tests (conservation, gradcheck, regression,
  smoke), tooling, CI, plumbing, refactors.
- **Build the user guide in-repo as concepts land.** Extend the guide under `docs/`
  (markdown), pairing each concept's theory with its FluxLB implementation. The
  maintainer authors the theory and must be able to defend it (the viva test);
  structure it, format it, cross-check it against sources, and draft the Tier-3
  implementation notes only. Theory prose is Tier 1 — do not author or finalise it.

## Project

`fluxlb` is a differentiable, quantum-ready Lattice Boltzmann solver in PyTorch, with three tracks: a classical core, a broad-ML SciML layer, and a Qiskit quantum layer. See `README.md` and `docs/lbm-sciml-roadmap.md`.

## The one invariant

**Streaming and collision are strictly separate. Collision is a swappable module behind `fluxlb/core/collision/base.py`.** This seam is what makes the solver differentiable and lets ML or quantum collision operators drop in. Never fuse streaming into collision, never bypass the `CollisionOperator` interface, and never break its signature without a deliberate, discussed change. Every new collision operator (classical, learned, or quantum) implements that interface.

## Conventions

- **Differentiability.** Keep tensor ops autograd-safe. Avoid in-place ops on tensors that require grad. Do not call `.item()`, `.numpy()`, or `.detach()` inside a differentiable path. Long rollouts use `torch.utils.checkpoint`; do not remove checkpointing to "simplify".
- **Dtype and device.** Compute in fp32 with fp64 accumulation for conserved moments (density, momentum). Respect a single device/dtype context; register lattice constants as buffers so `.to(device)` moves them.
- **Style.** Type hints on public functions. UK English in docstrings, comments and prose. Docstrings explain the physics/intent, not just the signature.
- **Interfaces.** The `CollisionOperator`, boundary, and dataset interfaces are stable contracts. Extend by adding implementations, not by editing the base classes casually.
- **Dependencies.** Do not add heavy dependencies (new DL frameworks, large libs) without discussion. Qiskit stays confined to `fluxlb/quantum/`.

## Testing expectations

- Each collision operator: mass and momentum conservation test.
- Differentiable paths: gradient check against finite differences within tolerance.
- Regression against analytical solutions (Poiseuille, Taylor-Green) before any ML or quantum layer is trusted.
- Each ML model family: a tiny-overfit smoke test (loss goes to near zero on 1-2 samples).

## Compute environment

- Primary node **mjolnir**: RTX A2000 Ada, 16 GB VRAM, weak fp64. Design memory-aware: gradient checkpointing, mixed precision, streaming data, factorised operators for 3D.
- SLURM nodes are **standalone** (no working cross-node fabric). Do **not** assume multi-node distributed training or DDP across nodes; target per-node runs.
- The foundation-model pretraining task (roadmap E5.10) is compute-gated and needs external HPC. Do not schedule it against the homelab.

## Do not

- Break or bypass the collision interface (see "The one invariant").
- Commit datasets, checkpoints, or large binaries.
- Author or finalise Tier 1 scientific content; propose and explain instead.
- Add dependencies or change public interfaces without flagging it.
- Rubber-stamp the maintainer's code, or approve it without checking against the
  theory, the Conventions, and the one invariant.
