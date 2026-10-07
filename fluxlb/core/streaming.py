# Copyright 2026 Muaaz Bhamjee
# SPDX-License-Identifier: Apache-2.0
r"""Streaming step of the lattice Boltzmann update.

Streaming is a pure shift: every post-collision population :math:`f_i^*(\mathbf{x}, t)`
is moved one lattice step along its own velocity,

.. math::

    f_i(\mathbf{x} + \mathbf{c}_i \Delta t,\; t + \Delta t) = f_i^*(\mathbf{x}, t).

It is exact, linear, non-local and touches no values; it only relocates them. In PyTorch
that is a ``torch.roll`` per direction (periodic by construction), with boundaries then
overriding the populations that crossed a wall. Because it is a permutation of the
population array, it is also the operation that becomes a permutation unitary on the
quantum track.

:math:`f^{\mathrm{eq}}` never appears in streaming. It lives entirely in collision, which
is local: at each node you take moments of the current populations
(:math:`\rho = \sum_i f_i`, :math:`\rho \mathbf{u} = \sum_i \mathbf{c}_i f_i`), build
:math:`f^{\mathrm{eq}}(\rho, \mathbf{u})` from them, and relax :math:`f_i` toward it,
:math:`f_i^* = f_i - \tfrac{1}{\tau}(f_i - f_i^{\mathrm{eq}})`. The two tie together
through the loop:

1. Collision pulls populations toward the local equilibrium (:math:`f^{\mathrm{eq}}`
   shares :math:`\rho` and :math:`\rho \mathbf{u}` with :math:`f`, so mass and momentum
   are conserved locally).
2. Streaming moves those relaxed populations to neighbouring nodes, so each node now
   holds a new mix and therefore new moments.
3. Collision recomputes :math:`f^{\mathrm{eq}}` from those new moments.
   :math:`f^{\mathrm{eq}}` is never stored across steps; it is regenerated from the
   streamed state every time.

The physics of the coupling: :math:`f^{\mathrm{eq}}` carries the inviscid transport (its
second moment is the Euler pressure tensor, which is why the truncation order of
:math:`f^{\mathrm{eq}}` in :math:`\mathbf{u}` matters). The non-equilibrium part
:math:`f - f^{\mathrm{eq}}`, which streaming carries to the neighbours before the next
relaxation, is what produces viscous stress. That is where
:math:`\nu = c_s^2 (\tau - \tfrac{1}{2}) \Delta t` comes from, and the :math:`\tfrac{1}{2}`
is purely a consequence of streaming being a discrete one-step shift.

That is also why the seam in fluxlb is drawn where it is: streaming is a fixed, trivially
differentiable permutation; everything that depends on :math:`f^{\mathrm{eq}}`, and hence
everything a learned or quantum operator would replace, sits inside the collision module.

Implementation notes
--------------------
Convention: push. The shift applied to ``f[q]`` is ``+lattice.shifts[q]`` along the
spatial axes, so the stored field after the call holds, at every node, the populations
that arrived there.

Population fields are stored directions-first, ``f[Q, *spatial]``, matching
:mod:`fluxlb.core.equilibrium`; spatial axis ``i`` of ``f`` is ``dim=i + 1`` and pairs
with component ``i`` of ``lattice.c``. This function never knows about walls; a
non-periodic boundary is applied after it and overwrites the wrapped values it owns.
"""

from __future__ import annotations

import torch

from fluxlb.core.lattices import Lattice

__all__ = ["stream"]


def stream(lattice: Lattice, f: torch.Tensor) -> torch.Tensor:
    r"""Push every population one lattice spacing along its own velocity.

    Streaming is a pure shift: every post-collision population
    :math:`f_i^*(\mathbf{x}, t)` is moved one lattice step along its own velocity,
    :math:`f_i(\mathbf{x} + \mathbf{c}_i \Delta t, t + \Delta t) = f_i^*(\mathbf{x}, t)`.
    It is exact, linear, non-local and touches no values; it only relocates them, so it
    conserves mass and momentum exactly and has no tunable parameter. It is a permutation
    of the population array: applying it with every shift negated is its inverse. A
    ``torch.roll`` per direction is periodic by construction; boundaries then override the
    populations that crossed a wall.

    The result is a new tensor, with no in-place write, so the step is autograd-safe by
    construction.

    Parameters
    ----------
    lattice
        Velocity set supplying ``shifts``, the integer displacement of each direction,
        kept on the host so no device synchronisation occurs here.
    f
        Post-collision populations, shape ``[Q, *spatial]``, with ``len(spatial) == lattice.D``.

    Returns
    -------
    torch.Tensor
        Streamed populations, same shape, dtype and device as ``f``. A new tensor;
        ``f`` is not modified.

    Raises
    ------
    ValueError
        If ``f.shape[0] != lattice.Q`` or ``f.ndim != lattice.D + 1``.
    """
    if f.shape[0] != lattice.Q:
        raise ValueError(f"f.shape[0] ({f.shape[0]}) != lattice.Q ({lattice.Q})")
    if f.ndim != lattice.D + 1:
        raise ValueError(f"f.ndim ({f.ndim}) != lattice.D + 1 ({lattice.D + 1})")
    # f[q] has already dropped the direction axis, so its spatial axes are 0..D-1.
    # Each roll allocates a new tensor and stack assembles them: no write into a
    # tensor that is in the autograd graph.
    dims = tuple(range(lattice.D))
    return torch.stack(
        [torch.roll(f[q], shifts=lattice.shifts[q], dims=dims) for q in range(lattice.Q)]
    )
