# `fluxlb.core.equilibrium`

Conserved moments and the discrete equilibrium. `moments()` projects a population field onto
density and momentum with fp64 accumulation; `equilibrium()` rebuilds the discrete
Maxwell-Boltzmann distribution from them as a Hermite expansion truncated at the order the
lattice's quadrature supports. For the derivation and the admissible orders per lattice, see
the user guide.

```{eval-rst}
.. automodule:: fluxlb.core.equilibrium
   :members:
```
