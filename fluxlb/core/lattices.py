# Copyright 2026 Muaaz Bhamjee
# SPDX-License-Identifier: Apache-2.0

"""Velocity sets, weights and sound speed for the classical solver.

A lattice is an :class:`torch.nn.Module` holding constants only. Registering them as buffers
means a single ``lattice.to(device, dtype)`` moves and casts everything together with the
solver, which keeps one device/dtype context as the Conventions require. The buffers are
non-persistent: they are derived data, so they stay out of checkpoints and can never be loaded
back with a stale dtype.

The distribution tensor ``f`` has shape ``[Q, *spatial]``; index 0 of every buffer runs over
the ``Q`` discrete velocities in the same order.
"""

import torch
from torch import nn

# ---------------------------------------------------------------------------
# Base class for a DdQq velocity set. Subclasses set Q and D and register the buffers
# listed below in __init__. Solver, equilibrium and boundary code should depend on this
# class, not on a concrete lattice, so that D3Q19 and D3Q27 drop in without signature changes.
# The buffers are non-persistent: they are derived data, so they stay out of checkpoints
# and can never be loaded back with a stale dtype.
# ---------------------------------------------------------------------------


class Lattice(nn.Module):
    """Base class for a DdQq velocity set.

    Subclasses set ``Q`` and ``D`` and register the buffers listed below in ``__init__``.
    Solver, equilibrium and boundary code should depend on this class, not on a concrete
    lattice, so that D3Q19 and D3Q27 drop in without signature changes.

    Attributes
    ----------
    Q : int
        Number of discrete velocities.
    D : int
        Spatial dimension.
    isotropy_order : int
        Highest even order at which the weighted velocity moments are isotropic. Order 2 is
        enough for scalar transport (advection-diffusion); order 4 is required to recover the
        isothermal Navier-Stokes equations; order 6 for thermal and compressible models.
        The tests check exactly the orders a lattice claims.
    c : torch.Tensor
        Discrete velocities, shape ``[Q, D]``, in the compute dtype so they can multiply
        ``f`` directly in the momentum sum.
    w : torch.Tensor
        Quadrature weights, shape ``[Q]``, in the compute dtype.
    opp : torch.Tensor
        Index of the opposite direction for each velocity, shape ``[Q]``, ``int64``. Used
        by bounce-back; ``f[opp]`` requires it on the same device as ``f``.
    cs2 : torch.Tensor
        Squared lattice sound speed, 0-d, in the compute dtype.
    shifts : list[tuple[int, ...]]
        The velocities as plain Python integers for ``torch.roll``. Kept off the device
        so the streaming step never synchronises with the host.
    """

    Q: int
    D: int
    isotropy_order: int

    c: torch.Tensor
    w: torch.Tensor
    opp: torch.Tensor
    cs2: torch.Tensor

    def __init__(
        self,
        c: torch.Tensor,
        w: torch.Tensor,
        opp: torch.Tensor,
        cs2: float,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Register the lattice constants as non-persistent buffers in ``dtype``.

        ``dtype`` is the compute dtype: fp32 by default per the Conventions (mjolnir has weak
        fp64 and MPS has none), fp64 for CPU regression tests. Casting to ``dtype`` happens
        here from exact Python values, so each dtype gets correctly rounded constants rather
        than an up-cast of fp32 ones.
        """
        super().__init__()
        if c.shape != (self.Q, self.D):
            raise ValueError(
                f"{type(self).__name__}: c has shape {tuple(c.shape)}, expected {(self.Q, self.D)}"
            )
        if w.shape != (self.Q,) or opp.shape != (self.Q,):
            raise ValueError(f"{type(self).__name__}: w and opp must have shape ({self.Q},)")
        if not torch.equal(c, c.round()):
            # Streaming shifts are taken from c with int(); a non-integer stencil would be
            # truncated silently and stream the wrong way.
            raise ValueError(
                f"{type(self).__name__}: velocities must be integer-valued for torch.roll"
            )
        self.register_buffer("c", c.to(dtype), persistent=False)
        self.register_buffer("w", w.to(dtype), persistent=False)
        self.register_buffer("opp", opp.to(torch.int64), persistent=False)
        self.register_buffer("cs2", torch.tensor(cs2, dtype=dtype), persistent=False)
        self.shifts: list[tuple[int, ...]] = [tuple(int(x) for x in v) for v in c.tolist()]


# ---------------------------------------------------------------------------
# 2D lattices. D2Q9 is the hydrodynamic stencil for the classical solver. D2Q5 is
# isotropic only to second order, so it serves scalar transport (advection-diffusion,
# passive temperature) and cannot carry the flow. D2Q37 is for thermal and compressible
# solvers.
# ---------------------------------------------------------------------------


class D2Q5(Lattice):
    """Two-dimensional, five-velocity lattice.

    For scalar transport only. With no diagonal velocities the fourth-order moment is not
    isotropic, so this lattice cannot recover Navier-Stokes; it is used for an
    advection-diffusion or temperature equation coupled to a D2Q9 flow field.

    Velocity ordering: rest particle first, then the four axis-aligned directions
    anticlockwise from ``+x``. The ``opp`` table is valid only for this ordering.

    Weights are ``1/3`` (rest) and ``1/6`` (axis); ``cs2 = 1/3``.
    """

    Q = 5
    D = 2
    isotropy_order = 2

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        """Build the D2Q5 constants in ``dtype`` (see :class:`Lattice`)."""
        # fmt: off
        c = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], dtype=torch.int64)
        # fmt: on
        w = torch.tensor([1 / 3] + [1 / 6] * 4, dtype=torch.float64)
        opp = torch.tensor([0, 3, 4, 1, 2], dtype=torch.int64)
        super().__init__(c=c, w=w, opp=opp, cs2=1 / 3, dtype=dtype)


class D2Q9(Lattice):
    """Two-dimensional, nine-velocity lattice.

    Velocity ordering: rest particle first, then the four axis-aligned directions
    anticlockwise from ``+x``, then the four diagonals anticlockwise from ``(+1, +1)``.
    The ``opp`` table is valid only for this ordering.

    Weights are ``4/9`` (rest), ``1/9`` (axis) and ``1/36`` (diagonal); ``cs2 = 1/3``.
    """

    Q = 9
    D = 2
    isotropy_order = 4

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        """Build the D2Q9 constants in ``dtype`` (see :class:`Lattice`)."""
        # fmt: off
        c = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                          [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=torch.int64)
        # fmt: on
        w = torch.tensor([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=torch.float64)
        opp = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=torch.int64)
        super().__init__(c=c, w=w, opp=opp, cs2=1 / 3, dtype=dtype)


class D2Q37(Lattice):
    """Two-dimensional, thirty-seven-velocity lattice for thermal and compressible flow.

    Space-filling form of the D2V37 quadrature of Philippi et al. (2006): integer velocities
    on eight shells, so it streams with ``torch.roll`` like D2Q9, with the reference
    temperature carried by the weights. The weighted moments are isotropic to eighth order
    (checked in the tests), which is what lets the equilibrium carry the energy equation.

    Velocity ordering: shell by shell with increasing ``|c|``; within a shell, opposite
    pairs are adjacent (``+x, -x, +y, -y`` and so on). This differs from D2Q9's anticlockwise
    order. The ``opp`` table is derived from the velocities, so it is valid for this
    ordering by construction.

    Shells and multiplicities: ``(0,0)`` x1, ``(1,0)`` x4, ``(1,1)`` x4, ``(2,0)`` x4,
    ``(2,1)`` x8, ``(2,2)`` x4, ``(3,0)`` x4, ``(3,1)`` x8. One weight per shell.

    ``cs2 = 1 / r**2`` with ``r = 1.19697977039307435897239`` the Hermite scaling factor,
    giving ``cs2 = 0.697953322...``. Weights as published (Philippi et al. 2006;
    Sbragaglia et al. 2009); they satisfy the moment constraints to orders 0-8 to 1e-14.
    TODO(maintainer): verify ``r`` and the eight weights against the source before relying
    on this lattice.
    """

    Q = 37
    D = 2
    isotropy_order = 8

    R = 1.19697977039307435897239
    """Hermite scaling factor of the ninth-order quadrature; ``cs2 = 1 / R**2``."""

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        """Build the D2Q37 constants in ``dtype`` (see :class:`Lattice`)."""
        # fmt: off
        shells: list[tuple[float, list[tuple[int, int]]]] = [
            (0.23315066913235250228650, [(0, 0)]),
            (0.10730609154221900241246, [(1, 0), (-1, 0), (0, 1), (0, -1)]),
            (0.05766785988879488203006, [(1, 1), (-1, 1), (1, -1), (-1, -1)]),
            (0.01420821615845075026469, [(2, 0), (-2, 0), (0, 2), (0, -2)]),
            (0.00535304900051377523273, [(2, 1), (-2, 1), (2, -1), (-2, -1),
                                         (1, 2), (-1, 2), (1, -2), (-1, -2)]),
            (0.00101193759267357547541, [(2, 2), (-2, 2), (2, -2), (-2, -2)]),
            (0.00024530102775771734547, [(3, 0), (-3, 0), (0, 3), (0, -3)]),
            (0.00028341425299419821740, [(3, 1), (-3, 1), (3, -1), (-3, -1),
                                         (1, 3), (-1, 3), (1, -3), (-1, -3)]),
        ]
        # fmt: on
        rows = [v for _, vs in shells for v in vs]
        weights = [wk for wk, vs in shells for _ in vs]
        c = torch.tensor(rows, dtype=torch.int64)
        w = torch.tensor(weights, dtype=torch.float64)
        opp = torch.tensor([rows.index((-x, -y)) for x, y in rows], dtype=torch.int64)
        super().__init__(c=c, w=w, opp=opp, cs2=1.0 / self.R**2, dtype=dtype)


# ---------------------------------------------------------------------------
# 3D lattices. D3Q19 is the hydrodynamic stencil for the classical solver. D3Q27 is for
# thermal and compressible solvers.
# ---------------------------------------------------------------------------


class D3Q19(Lattice):
    """Three-dimensional, nineteen-velocity lattice.

    Velocity ordering: TODO(maintainer) state the ordering here (rest first, then the six
    axis-aligned directions, then the twelve edge diagonals) and keep ``opp`` consistent
    with it. The ordering is a contract: boundary code will index ``f`` by it.

    Weights: TODO(maintainer) three distinct values for rest, axis and edge directions,
    with ``cs2 = 1/3``. Check them against the moment identities in
    ``tests/unit/test_lattices.py`` by adding this class to ``LATTICES`` there.
    """

    Q = 19
    D = 3
    isotropy_order = 4

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        """Build the D3Q19 constants in ``dtype`` (see :class:`Lattice`)."""
        # c = torch.tensor([[]]
        # TODO(maintainer): define c (int64, shape [19, 3]), w (float64, shape [19]) and
        # opp (int64, shape [19]) here, then replace the raise with
        #     super().__init__(c=c, w=w, opp=opp, cs2=1 / 3, dtype=dtype)
        raise NotImplementedError("D3Q19 stencil not yet defined")


class D3Q27(Lattice):
    """Three-dimensional, twenty-seven-velocity lattice.

    Velocity ordering: TODO(maintainer) state the ordering here (rest first, then the six
    axis-aligned directions, then the twelve edge diagonals, then the eight corner
    diagonals) and keep ``opp`` consistent with it.

    Weights: TODO(maintainer) four distinct values for rest, axis, edge and corner
    directions, with ``cs2 = 1/3``. D3Q27 is the full 3x3x3 stencil, so ``c`` can be
    generated from a product over ``{-1, 0, 1}`` rather than typed out, provided the
    rest velocity is moved to index 0 and ``opp`` is derived from ``c`` rather than
    hand-written. Check against the moment identities as for D3Q19.
    """

    Q = 27
    D = 3
    isotropy_order = 4

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        """Build the D3Q27 constants in ``dtype`` (see :class:`Lattice`)."""
        # TODO(maintainer): define c (int64, shape [27, 3]), w (float64, shape [27]) and
        # opp (int64, shape [27]) here, then replace the raise with
        #     super().__init__(c=c, w=w, opp=opp, cs2=1 / 3, dtype=dtype)
        raise NotImplementedError("D3Q27 stencil not yet defined")
