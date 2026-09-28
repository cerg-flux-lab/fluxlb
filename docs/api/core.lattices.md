# `fluxlb.core.lattices`

Velocity sets, weights and sound speed. A lattice is an `nn.Module` holding constants as
non-persistent buffers so that one `.to(device, dtype)` moves everything with the solver.
For the moment identities each lattice satisfies and how they are tested, see the user guide.

```{eval-rst}
.. automodule:: fluxlb.core.lattices
```
