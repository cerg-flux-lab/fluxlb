# fluxlb

**Differentiable, quantum-ready Lattice Boltzmann in PyTorch.**

<!-- Badges (fill in once CI, PyPI, docs and licence are live) -->
<!-- ![CI](...) ![PyPI](...) ![Docs](...) ![Licence](...) -->

## Overview

`fluxlb` is a Lattice Boltzmann solver built on PyTorch. Streaming and collision are kept strictly separate, so the collision operator is a swappable module. That single design choice does three things at once: it keeps the classical solver fully differentiable, it lets machine-learning closures drop in at the collision step, and it exposes the exact seam where a quantum collision operator can be substituted.

**Why another LBM solver.** The solver exists to couple three areas that rarely meet in one code: the Lattice Boltzmann method, scientific machine learning, and quantum computing and quantum machine learning. Most LBM solvers are written in C++, whilst the frameworks for the other two, PyTorch and Qiskit, are Python native. Building the solver in PyTorch removes that boundary: the core LBM inherits GPU acceleration and automatic differentiation, and integrating a machine-learning component becomes a matter of swapping a module rather than crossing a language barrier.

**What differentiability buys.** Because the whole collide-stream-boundary loop is traceable by autograd, gradients flow from any output back to any input: the relaxation time, boundary values, initial conditions, or the weights of a learned closure. Inverse problems, closures trained through the solver, and data assimilation therefore become direct uses of the solver rather than separate tooling.

**Quantum scope.** The same Python-native design lets the quantum track develop inside the solver rather than beside it: streaming as a permutation unitary, Carleman linearisation of the collision step, and hybrid variational closures, all entering through the collision interface. "Quantum-ready" means the seam and simulator-scale implementations exist; it does not mean hardware-scale flow simulation today. 

<!-- Later, once there is something to compare: a 'Where it sits' paragraph positioning fluxlb against lettuce (PyTorch), XLB (JAX) and Palabos. -->

This is not a trivial pursuit, but one born out of a drive to push the frontier of computational fluid dynamics. Contributions are welcome; see [Contributing](#contributing).

## Three tracks

- **Classical core.** Differentiable LBM: velocity sets (D2Q9, D3Q19, D3Q27), collision operators (BGK, TRT, MRT, regularised), standard boundary conditions, moments and time loop.
- **SciML (broad ML).** Neural operators, reduced-order models, learned closures trained in the loop, generative and graph models, uncertainty quantification, inverse problems and data assimilation, reinforcement learning and equation discovery. See the roadmap.
- **Quantum (Qiskit).** Streaming as a permutation unitary, Carleman linearisation of the collision feeding a quantum linear-algebra / Hamiltonian-simulation backend, and hybrid variational closures. Aligned with the lab's Fokker-Planck / Carleman quantum-CFD programme.

## Status

Early and under active development. Interfaces are not yet stable. The quickstart below reflects the **target** API.

## Installation

Requires Python 3.12+ and a recent PyTorch build.

```bash
git clone https://github.com/cerg-flux-lab/fluxlb.git fluxlb
cd fluxlb
pip install -e ".[dev]"

# optional extras
pip install -e ".[sciml]"      # ML model families
pip install -e ".[quantum]"    # Qiskit backend
pip install -e ".[quantum-hw]" # For running on Quantum hardware
pip install -e ".[docs]"       # For building the documentation
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

Built with Sphinx and published to `https://cerg-flux-lab.github.io/fluxlb/` (theory guide, API reference, worked tutorials). Contributions to docs are expected alongside code, not after.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). One rule dominates: **do not fuse streaming and collision, and do not bypass the `CollisionOperator` interface.** That seam is what keeps the solver differentiable and quantum-ready.

## Citing

```bibtex
@software{fluxlb,
  author  = {Bhamjee, Muaaz and {{CERG-FLUX Lab}}},
  title   = {fluxlb: Differentiable, quantum-ready Lattice Boltzmann in PyTorch},
  year    = {2026},
  url     = {https://github.com/cerg-flux-lab/fluxlb},
}
```

## Licence

Copyright 2026 Muaaz Bhamjee. Released under the [Apache License 2.0](LICENSE). See [`NOTICE`](NOTICE) for attribution.

## Acknowledgements

Developed in the **CERG-FLUX Lab (Fluids, Learning and Uncertainty in compleX systems)**, University of Pretoria. GitHub: [github.com/cerg-flux-lab](https://github.com/cerg-flux-lab). Lab site: [cerg-flux-lab.github.io](https://cerg-flux-lab.github.io/).

## Funding

1. This work was funded by the South African Quantum Technology Initiative (SA QuTI) through the Department of Science, Technology and Innovation (DSTI) of South Africa via the University of Pretoria Quantum Science and Technology (UPQuST).
1. This work was supported by the University of Pretoria through the Research Development Programme (RDP). Grant Title: Advancing Computational Techniques for Multiphase Flow: Lattice Boltzmann Method, Deep Learning and Quantum Computing Approaches.

## Attribution

All core numerical methods and program logic have been independently developed by **Muaaz Bhamjee** for this project. No code from third-party solvers has been incorporated; the project depends only on the open-source libraries declared in `pyproject.toml` (PyTorch and NumPy, with Qiskit and the SciML stack as optional extras). The project leverages standard numerical and computational methods as described in the literature, but all implementation is original.

During development, the following AI-assisted tools were used to support productivity and code clarity:

- **Claude AI (Anthropic)** — debugging, code review, documentation formatting, code completion and inline suggestions within Visual Studio Code, and conceptual explanations

These AI tools provided assistance only. All code and research outputs are authored solely by **Muaaz Bhamjee**. AI tools assisted with implementation; `Co-Authored-By` trailers are not used because AI tools are not authors and hold no IP; all authorship remains with the project maintainers. Whilst AI use is encouraged to improve quality, understanding should not be delegated to AI.

This statement is provided to clarify licensing, attribution, and the role of AI in the development of this project.

