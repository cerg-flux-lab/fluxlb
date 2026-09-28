# fluxlb

**Differentiable, quantum-ready Lattice Boltzmann in PyTorch.**

`fluxlb` keeps streaming and collision strictly separate, so the collision operator is a
swappable module. That one seam keeps the classical solver differentiable, lets
machine-learning closures drop in at the collision step, and marks where a quantum collision
operator can be substituted.

This site is the **API reference**, generated from the docstrings, together with the
roadmap. The **user guide**, which carries the theory and the implementation narrative, is a
LaTeX document published here as a PDF; see {doc}`guide`.

```{toctree}
:maxdepth: 2
:caption: Contents

guide
api/index
lbm-sciml-roadmap
issue-checklists
```

## Source

- Repository: <https://github.com/cerg-flux-lab/fluxlb>
- User guide source: <https://github.com/cerg-flux-lab/fluxlb_guide> (mirrored from `docs/guide`)
- Lab: <https://cerg-flux-lab.github.io/>
