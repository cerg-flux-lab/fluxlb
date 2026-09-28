# User guide

The narrative user guide pairs each concept's theory with its `fluxlb` implementation. It is
written in LaTeX, lives in `docs/guide/` of the repository, and is mirrored to the
[`fluxlb_guide`](https://github.com/cerg-flux-lab/fluxlb_guide) repository for editing in
Overleaf.

- **PDF (latest build):** <https://cerg-flux-lab.github.io/fluxlb/guide/fluxlb-guide.pdf>
- **Source:** [`docs/guide/fluxlb-guide.tex`](https://github.com/cerg-flux-lab/fluxlb/tree/main/docs/guide)

Part I covers the device and precision utilities, the velocity sets, and the tests that
verify them. Later parts will follow the solver as it is built: streaming, collision
operators, boundary conditions, the differentiable time loop, and the SciML and quantum
tracks.

Theory prose in the guide is written by the maintainer. Passages still to be written are
marked in the PDF. This site does not duplicate the guide's theory; for the meaning of a
symbol or a design choice, read the guide, and for a function's signature and behaviour,
read the {doc}`api/index`.

## Building the PDF locally

```bash
cd docs/guide
latexmk -pdf fluxlb-guide.tex      # needs a TeX Live with biber
```
