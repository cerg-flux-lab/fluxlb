# API reference

Generated from the docstrings. Docstrings in `fluxlb` explain the physics and the intent, not
only the signature, so this reference is meant to be read alongside the user guide.

```{toctree}
:maxdepth: 2
:glob:

*
```

## Package layout

| Package | Contents |
|---|---|
| `fluxlb.core` | classical solver: lattices, equilibrium, streaming, collision, boundaries, solver |
| `fluxlb.sciml` | machine-learning model families |
| `fluxlb.quantum` | Qiskit backend and Carleman / hybrid closures |
| `fluxlb.data` | dataset generation, IO, loaders |
| `fluxlb.eval` | metrics, benchmarks, leaderboard |

Only modules that are implemented and tested appear in the reference. A page is generated
at build time for every `fluxlb` module imported by a test under `tests/unit`, so a module
enters the reference when its unit test lands. Nothing is listed by hand.
