# fluxlb

**Differentiable, quantum-ready Lattice Boltzmann in PyTorch.**

<!-- Badges (fill in once CI, PyPI, docs and licence are live) -->
<!-- ![CI](...) ![PyPI](...) ![Docs](...) ![Licence](...) -->

<!--
TIER 1 (yours to write / verify): the Overview below is a neutral starting scaffold, not final science.
Rewrite it in your own words and own the scientific claims (novelty, why differentiability matters,
what "quantum-ready" does and does not promise). Claude is amplifier, not origin.
-->
## Overview

`fluxlb` is a Lattice Boltzmann solver built on PyTorch. Streaming and collision are kept strictly separate, so the collision operator is a swappable module. That single design choice does three things at once: it keeps the classical solver fully differentiable, it lets machine-learning closures drop in at the collision step, and it exposes the exact seam where a quantum collision operator can be substituted.

> _Suggested points to expand in your own words:_ the scientific motivation; where a differentiable solver changes what is possible (inverse problems, learned closures); the honest scope of the quantum track (simulator-scale now, hardware later); how this complements, rather than replaces, production tools such as Palabos.

## Three tracks

- **Classical core.** Differentiable LBM: velocity sets (D2Q9, D3Q19, D3Q27), collision operators (BGK, TRT, MRT, regularised), standard boundary conditions, moments and time loop.
- **SciML (broad ML).** Neural operators, reduced-order models, learned closures trained in the loop, generative and graph models, uncertainty quantification, inverse problems and data assimilation, reinforcement learning and equation discovery. See the roadmap.
- **Quantum (Qiskit).** Streaming as a permutation unitary, Carleman linearisation of the collision feeding a quantum linear-algebra / Hamiltonian-simulation backend, and hybrid variational closures. Aligned with the lab's Fokker-Planck / Carleman quantum-CFD programme.

## Status

Early and under active development. Interfaces are not yet stable. The quickstart below reflects the **target** API.

## Installation

Requires Python 3.11+ and a recent PyTorch build.

```bash
git clone https://github.com/cerg-flux-lab/fluxlb.git fluxlb
cd fluxlb
pip install -e ".[dev]"

# optional extras
pip install -e ".[sciml]"     # ML model families
pip install -e ".[quantum]"   # Qiskit backend
```

## Quickstart (target API)

```python
import torch
import fluxlb as flb

# lid-driven cavity, D2Q9, BGK
solver = flb.LBMSolver(
    lattice=flb.lattices.D2Q9(),
    collision=flb.collision.BGK(tau=0.6),
    boundaries=flb.boundaries.LidDrivenCavity(nx=256, ny=256, u_lid=0.1),
    differentiable=False,
)

f = solver.initialise(device="cuda", dtype=torch.float32)
f = solver.run(f, n_steps=20_000)
rho, u = solver.moments(f)
```

Switch `differentiable=True` to backpropagate through the solve (see `examples/inverse_viscosity.py`).

## Repository structure

```
fluxlb/
  core/
    lattices.py            # velocity sets, weights, sound speed
    equilibrium.py         # f_eq
    streaming.py           # streaming (permutation) operator
    collision/
      base.py              # CollisionOperator interface (the classical-quantum seam)
      bgk.py trt.py mrt.py
    boundaries/            # bounce-back, Zou-He, periodic, ...
    solver.py              # LBMSolver(nn.Module): collide-stream-BC time loop
  sciml/                   # operators, rom, closures, generative, graph, uq, inverse, rl, discovery
  quantum/                 # streaming-as-permutation, carleman, hybrid QML closures (Qiskit)
  data/                    # dataset generation, IO, loaders
  eval/                    # metrics, benchmarks, leaderboard
tests/                     # unit, regression (analytical), gradient checks, smoke
docs/                      # theory guide, API reference, tutorials
examples/                  # runnable notebooks and scripts
scripts/                   # SLURM templates, board automation
```

## Roadmap

The full plan lives in [`docs/lbm-sciml-roadmap.md`](docs/lbm-sciml-roadmap.md) (SciML track) with per-issue sub-task checklists in [`docs/issue-checklists.md`](docs/issue-checklists.md). The GitHub Projects board is created by [`scripts/create-lbm-sciml-board.sh`](scripts/create-lbm-sciml-board.sh).

## Documentation

Built with Sphinx / MkDocs and published to `https://cerg-flux-lab.github.io/` (theory guide, API reference, worked tutorials). Contributions to docs are expected alongside code, not after.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). One rule dominates: **do not fuse streaming and collision, and do not bypass the `CollisionOperator` interface.** That seam is what keeps the solver differentiable and quantum-ready.

## Citing

<!-- TIER 1: confirm authorship, ordering and metadata before release. -->
A `CITATION.cff` will accompany the first release. Provisional entry:

```bibtex
@software{fluxlb,
  author  = {Bhamjee, Muaaz and {CERG-FLUX Lab}},
  title   = {fluxlb: Differentiable, quantum-ready Lattice Boltzmann in PyTorch},
  year    = {2026},
  url     = {https://github.com/cerg-flux-lab/fluxlb},
  orcid   = {0000-0002-2697-4589}
}
```

## Licence

Copyright 2026 Muaaz Bhamjee. Released under the [Apache License 2.0](LICENSE). See [`NOTICE`](NOTICE) for attribution.

## Acknowledgements

Developed in the **CERG-FLUX Lab (Fluids, Learning and Uncertainty in compleX systems)**, University of Pretoria. GitHub: [github.com/cerg-flux-lab](https://github.com/cerg-flux-lab). Lab site: [cerg-flux-lab.github.io](https://cerg-flux-lab.github.io/).

<!-- Add funding acknowledgements (grant numbers) before release. -->
