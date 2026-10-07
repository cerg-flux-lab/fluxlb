# API reference

Generated from the docstrings. Docstrings in `fluxlb` explain the physics and the intent, not
only the signature, so this reference is meant to be read alongside the user guide.

```{toctree}
:maxdepth: 2

core.gpu
core.lattices
core.equilibrium
core.streaming
```

## Package layout

| Package | Contents |
|---|---|
| `fluxlb.core` | classical solver: lattices, equilibrium, streaming, collision, boundaries, solver |
| `fluxlb.sciml` | machine-learning model families |
| `fluxlb.quantum` | Qiskit backend and Carleman / hybrid closures |
| `fluxlb.data` | dataset generation, IO, loaders |
| `fluxlb.eval` | metrics, benchmarks, leaderboard |

Only modules that are implemented and tested appear in the reference. Pages are added as
modules land.
