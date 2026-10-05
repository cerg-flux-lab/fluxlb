# Copyright 2026 Muaaz Bhamjee
# SPDX-License-Identifier: Apache-2.0
"""Conserved moments and the discrete equilibrium distribution.

:func:`moments` computes the collision invariants, density and momentum, from a
population field. :func:`equilibrium` computes the discrete Maxwell-Boltzmann distribution
with those invariants, as a Hermite expansion truncated at the order the lattice's
quadrature supports. Every collision operator relaxes ``f`` towards that equilibrium.

The expansion is isothermal. The lattice temperature is the reference temperature, ``cs2``
is ``k_B T / m`` in lattice units, and no temperature field enters. Thermal and
compressible models require additional terms in the temperature deviation, which are not
implemented.

Population fields are stored directions-first, ``f[Q, *spatial]``, so the direction axis
is ``dim=0`` for every reduction and ``lattice.c`` multiplies ``f`` directly. Velocity
fields are components-first, ``u[D, *spatial]``, to match.
"""

from __future__ import annotations

import torch

from fluxlb.core.gpu import accumulation_dtype
from fluxlb.core.lattices import Lattice

__all__ = ["equilibrium", "moments"]


def moments(lattice: Lattice, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Density and velocity of a population field.

    The density is the zeroth velocity moment of ``f``, the sum over directions, and the
    momentum is the first moment, the sum of ``f`` weighted by the discrete velocities.
    The velocity is momentum divided by density. These two moments are the collision
    invariants: BGK, TRT and MRT leave them unchanged, so ``f`` and the equilibrium built
    from these values have the same density and momentum, and mass and momentum
    conservation of the solver reduce to this function returning the same values before and
    after collision.

    Reductions over the direction axis accumulate in :func:`accumulation_dtype` and the
    results are returned in the compute dtype of ``f``. Rounding error in the invariants
    enters the equilibrium at the next step and is not corrected by the relaxation, so the
    fp32-compute / fp64-accumulation policy is applied here and only here; collision
    operators and solver diagnostics call this function rather than reducing ``f``
    themselves. The division by density is performed in the accumulation dtype before the
    cast back.

    Parameters
    ----------
    lattice
        Velocity set supplying ``c`` and ``cs2``.
    f
        Populations, shape ``[Q, *spatial]``.

    Returns
    -------
    rho : torch.Tensor
        Density, shape ``[*spatial]``, dtype of ``f``.
    u : torch.Tensor
        Velocity, shape ``[D, *spatial]``, dtype of ``f``.
    """
    dtype = f.dtype
    acc = accumulation_dtype(f.device)
    f_acc = f.to(acc)
    c_acc = lattice.c.to(acc)
    rho = torch.sum(f_acc, dim=0)
    u = torch.einsum("qa,q...->a...", c_acc, f_acc) / rho
    return rho.to(dtype), u.to(dtype)


def equilibrium(
    lattice: Lattice,
    rho: torch.Tensor,
    u: torch.Tensor,
    order: int | None = None,
) -> torch.Tensor:
    r"""Discrete equilibrium populations for a given density and velocity.

    The continuous Maxwell-Boltzmann distribution at the lattice reference temperature is
    expanded in tensor Hermite polynomials of the scaled velocity ``c / cs``, and the
    expansion is evaluated at the discrete velocities with the lattice weights as the
    Gauss-Hermite quadrature weights. Truncating at order ``N`` gives

    .. math::

        f^{\mathrm{eq}}_q = w_q \, \rho \sum_{n=0}^{N} \frac{1}{n!}
        \, \mathbf{a}^{(n)} : \mathbf{H}^{(n)}(\mathbf{c}_q / c_s),

    with Hermite coefficients ``a^(n) = (u / cs)^{⊗n}`` for the isothermal distribution.
    Contracting each Hermite tensor with ``u`` repeated ``n`` times collapses the terms to
    polynomials in ``c·u`` and ``u·u`` alone, which is the form implemented:

    ========  ======================================================================
    order     term added to ``1 + c·u / cs2``
    ========  ======================================================================
    2         ``((c·u)^2 - cs2 u·u) / (2 cs2^2)``
    3         ``((c·u)^3 - 3 cs2 (c·u) u·u) / (6 cs2^3)``
    4         ``((c·u)^4 - 6 cs2 (c·u)^2 u·u + 3 cs2^2 (u·u)^2) / (24 cs2^4)``
    ========  ======================================================================

    Order ``N`` includes every term below it.

    A truncation at order ``N`` reproduces the velocity moments of the Maxwell-Boltzmann
    distribution exactly up to order ``N``, provided the quadrature integrates polynomials
    of degree ``2N`` exactly. ``lattice.equilibrium_order`` stores the highest ``N`` for
    which this holds. A higher truncation adds terms with incorrect moments, so requesting
    one raises ``ValueError``. Order 1 is the linear equilibrium for scalar transport on
    D2Q5. Order 2 is the isothermal Navier-Stokes equilibrium on D2Q9, D3Q19 and D3Q27: it
    reproduces the momentum flux ``rho (cs2 I + u u)``. Orders 3 and 4 on D2Q37 recover the
    third- and fourth-order moments, which give the energy flux and remove the cubic
    velocity error of the order-2 form. The truncation error in the velocity is of order
    ``Ma^(N+1)``, so a low Mach number is assumed.

    Parameters
    ----------
    lattice
        Velocity set supplying ``c``, ``w``, ``cs2`` and ``equilibrium_order``.
    rho
        Density, shape ``[*spatial]``.
    u
        Velocity, shape ``[D, *spatial]``.
    order
        Truncation order of the Hermite expansion, ``1 <= order <= lattice.equilibrium_order``.
        ``None`` selects ``lattice.equilibrium_order``. A lower order than the lattice
        supports is allowed, for comparison against the standard order-2 form.

    Returns
    -------
    torch.Tensor
        Equilibrium populations, shape ``[Q, *spatial]``, dtype of ``rho``.

    Raises
    ------
    ValueError
        If ``order`` is below 1 or above ``lattice.equilibrium_order``.

    Notes
    -----
    The result is computed in the compute dtype of ``rho`` and ``u`` with no accumulation
    cast: the equilibrium is a pointwise reconstruction, not a reduction over directions,
    so its rounding error is not fed back through the invariants. The expression is a
    single broadcast over ``c`` and ``w`` with no in-place operations, so gradients flow to
    both ``rho`` and ``u`` and through ``lattice.cs2`` if it is ever made learnable.
    """
    if order is None:
        order = lattice.equilibrium_order
    if not 1 <= order <= lattice.equilibrium_order:
        raise ValueError(
            f"equilibrium: order {order} is outside 1..{lattice.equilibrium_order} for "
            f"{type(lattice).__name__}"
        )

    # Shapes: w -> [Q, 1, ..., 1] so it broadcasts against [Q, *spatial]; cu -> [Q, *spatial];
    # uu -> [*spatial]. All in the compute dtype of rho and u; no accumulation cast, since
    # f_eq is not a conserved-moment reduction.
    w = lattice.w.reshape(-1, *([1] * lattice.D))
    cs2 = lattice.cs2
    cu = torch.einsum("qa,a...->q...", lattice.c, u)
    uu = torch.sum(u * u, dim=0)

    series = 1.0 + cu / cs2
    if order >= 2:
        series = series + (cu**2 - cs2 * uu) / (2 * cs2**2)
    if order >= 3:
        series = series + (cu**3 - 3 * cs2 * cu * uu) / (6 * cs2**3)
    if order >= 4:
        series = series + (cu**4 - 6 * cs2 * cu**2 * uu + 3 * cs2**2 * uu**2) / (24 * cs2**4)
    return w * rho * series
