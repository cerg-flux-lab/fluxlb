# Copyright 2026 Muaaz Bhamjee
# SPDX-License-Identifier: Apache-2.0
"""Streaming: a push-convention, periodic permutation of the population array.

The acceptance criteria follow the contract in :mod:`fluxlb.core.streaming`. Streaming must
be a pure relocation of values: the multiset of entries is unchanged, every population lands
exactly one lattice spacing along its own velocity with periodic wrap, applying the negated
shifts undoes it, and the conserved moments are unchanged. Because the operation is a
permutation, the conservation checks compare in fp64 and sorted order rather than to a
tolerance; a non-zero difference there is a bug, not rounding.
"""

import pytest
import torch

from fluxlb.core import lattices
from fluxlb.core.equilibrium import moments
from fluxlb.core.streaming import stream

LATTICES = [lattices.D2Q5, lattices.D2Q9, lattices.D2Q37, lattices.D3Q19, lattices.D3Q27]
DTYPES = [torch.float32, torch.float64]
# Grids are larger than the longest shift on every lattice (|c| <= 3 on D2Q37) so a wrap
# is a genuine wrap and not a shift by a multiple of the grid.
GRID = {2: (5, 7), 3: (4, 5, 6)}


@pytest.fixture(params=LATTICES, ids=lambda cls: cls.__name__)
def lattice_cls(request):
    return request.param


@pytest.fixture(params=DTYPES, ids=lambda dt: str(dt).removeprefix("torch."))
def dtype(request):
    return request.param


@pytest.fixture
def lattice(lattice_cls, dtype):
    return lattice_cls(dtype=dtype)


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


@pytest.fixture
def f(lattice, dtype):
    """A positive population field with no repeated values, so a misplaced entry is visible."""
    spatial = GRID[lattice.D]
    return 0.5 + torch.rand((lattice.Q, *spatial), generator=_gen(), dtype=dtype)


def _unstream(lattice, g):
    """Reference inverse: roll every direction back by its negated shift."""
    dims = tuple(range(lattice.D))
    return torch.stack(
        [torch.roll(g[q], [-s for s in lattice.shifts[q]], dims) for q in range(lattice.Q)]
    )


# --- shape, dtype, aliasing ---------------------------------------------------------


@pytest.mark.smoke
def test_shape_and_dtype(lattice, f, dtype):
    g = stream(lattice, f)
    assert g.shape == f.shape
    assert g.dtype == dtype
    assert g.device == f.device


def test_returns_new_tensor_and_leaves_input_unchanged(lattice, f):
    before = f.clone()
    g = stream(lattice, f)
    assert g.data_ptr() != f.data_ptr()
    assert torch.equal(f, before)


# --- the permutation ------------------------------------------------------------------


def test_push_lands_one_step_along_own_velocity(lattice, f):
    """Push convention: f[q] at x arrives at x + c_q, modulo the grid."""
    g = stream(lattice, f)
    spatial = GRID[lattice.D]
    origin = (0,) * lattice.D
    for q in range(lattice.Q):
        target = tuple((s % n) for s, n in zip(lattice.shifts[q], spatial, strict=True))
        assert torch.equal(g[(q, *target)], f[(q, *origin)]), f"direction {q}"


def test_every_direction_is_a_roll_by_its_shift(lattice, f):
    """The whole slab, not just one node: the roll is exactly ``torch.roll`` by ``+shifts[q]``."""
    g = stream(lattice, f)
    dims = tuple(range(lattice.D))
    for q in range(lattice.Q):
        assert torch.equal(g[q], torch.roll(f[q], lattice.shifts[q], dims)), f"direction {q}"


def test_rest_population_does_not_move(lattice, f):
    g = stream(lattice, f)
    rest = [q for q in range(lattice.Q) if all(s == 0 for s in lattice.shifts[q])]
    assert len(rest) == 1
    assert torch.equal(g[rest[0]], f[rest[0]])


def test_periodic_wrap(lattice, f):
    """A population leaving the last node of an axis re-enters at the first node."""
    g = stream(lattice, f)
    spatial = GRID[lattice.D]
    for q in range(lattice.Q):
        for axis, s in enumerate(lattice.shifts[q]):
            if s <= 0:
                continue
            # source on the last node of this axis, zero elsewhere
            src = [0] * lattice.D
            src[axis] = spatial[axis] - 1
            dst = [(src[a] + sh) % spatial[a] for a, sh in enumerate(lattice.shifts[q])]
            assert dst[axis] == s - 1, "wrap arithmetic"
            assert torch.equal(g[(q, *dst)], f[(q, *src)]), f"direction {q}, axis {axis}"


def test_is_a_permutation_of_the_entries(lattice, f):
    g = stream(lattice, f)
    assert torch.equal(g.flatten().sort().values, f.flatten().sort().values)


def test_negated_shifts_invert(lattice, f):
    g = stream(lattice, f)
    assert torch.equal(_unstream(lattice, g), f)


def test_opposite_direction_streams_back(lattice, f):
    """Streaming direction opp[q] by +c_opp[q] equals streaming q by -c_q: the opp table and
    the shift list agree."""
    g = stream(lattice, f)
    dims = tuple(range(lattice.D))
    for q in range(lattice.Q):
        qb = int(lattice.opp[q])
        back = torch.roll(g[q], lattice.shifts[qb], dims)
        assert torch.equal(back, f[q]), f"direction {q}, opposite {qb}"


# --- conservation -------------------------------------------------------------------


@pytest.mark.conservation
def test_global_mass_and_momentum_conserved(lattice_cls):
    """Global invariants are exactly unchanged by a permutation.

    Summation order differs between the two fields, so the comparison is made in fp64 on a
    sorted copy of the per-direction contributions; a last-bit difference in fp32 summation
    is not a streaming error.
    """
    lattice = lattice_cls(dtype=torch.float64)
    spatial = GRID[lattice.D]
    f = 0.5 + torch.rand((lattice.Q, *spatial), generator=_gen(), dtype=torch.float64)
    g = stream(lattice, f)
    flat = lambda t: t.reshape(lattice.Q, -1)  # noqa: E731
    # mass: per direction the slab is a permutation, so direction sums match exactly
    assert torch.equal(flat(f).sort(dim=1).values, flat(g).sort(dim=1).values)
    # momentum: each direction's slab sum is unchanged, hence sum_q c_q * sum_x f_q is too
    mass_f = flat(f).sort(dim=1).values.sum(dim=1)
    mass_g = flat(g).sort(dim=1).values.sum(dim=1)
    assert torch.equal(mass_f, mass_g)
    mom_f = torch.einsum("qa,q->a", lattice.c, mass_f)
    mom_g = torch.einsum("qa,q->a", lattice.c, mass_g)
    assert torch.equal(mom_f, mom_g)


@pytest.mark.conservation
def test_moments_of_uniform_field_unchanged(lattice):
    """On a spatially uniform field streaming is the identity, so local moments are unchanged."""
    spatial = GRID[lattice.D]
    fq = 0.5 + torch.rand(lattice.Q, generator=_gen(), dtype=lattice.w.dtype)
    f = fq.reshape(-1, *([1] * lattice.D)).expand(lattice.Q, *spatial).clone()
    g = stream(lattice, f)
    assert torch.equal(g, f)
    rho_f, u_f = moments(lattice, f)
    rho_g, u_g = moments(lattice, g)
    assert torch.equal(rho_f, rho_g) and torch.equal(u_f, u_g)


# --- errors -----------------------------------------------------------------------------


def test_wrong_q_raises(lattice_cls):
    lattice = lattice_cls()
    spatial = GRID[lattice.D]
    with pytest.raises(ValueError, match=r"lattice\.Q"):
        stream(lattice, torch.rand((lattice.Q + 1, *spatial)))


def test_wrong_ndim_raises(lattice_cls):
    lattice = lattice_cls()
    spatial = GRID[lattice.D]
    with pytest.raises(ValueError, match=r"lattice\.D \+ 1"):
        stream(lattice, torch.rand((lattice.Q, *spatial[:-1])))
    with pytest.raises(ValueError, match=r"lattice\.D \+ 1"):
        stream(lattice, torch.rand((lattice.Q, *spatial, 2)))


# --- device -------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_output_on_input_device(lattice_cls):
    lattice = lattice_cls().to("cuda")
    f = torch.rand((lattice.Q, *GRID[lattice.D]), device="cuda")
    g = stream(lattice, f)
    assert g.device == f.device
    assert torch.equal(g.cpu(), stream(lattice_cls(), f.cpu()))
