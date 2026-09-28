# Copyright 2026 Muaaz Bhamjee
# SPDX-License-Identifier: Apache-2.0
"""Lattice constants: moment identities, opposite-direction table, buffer and dtype behaviour.

Every test is parametrised over ``LATTICES``; add D3Q19 and D3Q27 there when they land.
The moment identities are the conditions on the weights and velocities for the discrete
equilibrium to reproduce the isothermal Navier-Stokes equations: normalisation, zero odd
moments, and second- and fourth-order isotropy with the given ``cs2``.
"""

import pytest
import torch

from fluxlb.core import gpu, lattices

LATTICES = [lattices.D2Q5, lattices.D2Q9, lattices.D2Q37, lattices.D3Q19, lattices.D3Q27]
DTYPES = [torch.float32, torch.float64]
TOL = {torch.float32: 1e-6, torch.float64: 1e-12}


def _eye(lattice, dtype):
    return torch.eye(lattice.D, dtype=dtype)


def _isotropic_tensor(order, delta):
    """Sum over all pairings of `order` indices of products of Kronecker deltas."""

    def pairings(idx):
        if not idx:
            yield []
            return
        first, rest = idx[0], idx[1:]
        for k, other in enumerate(rest):
            for tail in pairings(rest[:k] + rest[k + 1 :]):
                yield [(first, other), *tail]

    d = delta.shape[0]
    out = torch.zeros((d,) * order, dtype=delta.dtype)
    for pairing in pairings(list(range(order))):
        term = torch.ones((d,) * order, dtype=delta.dtype)
        for i, j in pairing:
            shape = [1] * order
            shape[i] = shape[j] = d
            term = term * delta.reshape(shape)
        out += term
    return out


def _moment(lattice, order):
    c = lattice.c
    letters = "abcdefgh"[:order]
    spec = "q," + ",".join(f"q{x}" for x in letters) + "->" + letters
    return torch.einsum(spec, lattice.w, *([c] * order))


@pytest.fixture(params=LATTICES, ids=lambda cls: cls.__name__)
def lattice_cls(request):
    return request.param


@pytest.fixture(params=DTYPES, ids=lambda dt: str(dt).removeprefix("torch."))
def dtype(request):
    return request.param


@pytest.fixture
def lattice(lattice_cls, dtype):
    return lattice_cls(dtype=dtype)


# --- shapes and metadata --------------------------------------------------------


@pytest.mark.smoke
def test_shapes_match_q_and_d(lattice):
    assert lattice.c.shape == (lattice.Q, lattice.D)
    assert lattice.w.shape == (lattice.Q,)
    assert lattice.opp.shape == (lattice.Q,)
    assert lattice.cs2.ndim == 0
    assert len(lattice.shifts) == lattice.Q
    assert all(len(s) == lattice.D for s in lattice.shifts)


def test_velocities_are_distinct_integers(lattice):
    assert torch.equal(lattice.c, lattice.c.round())
    assert len({tuple(v) for v in lattice.c.tolist()}) == lattice.Q  # no duplicates
    assert torch.all(lattice.c[0] == 0)  # rest velocity first


def test_shifts_are_python_ints_equal_to_c(lattice):
    assert isinstance(lattice.shifts, list)
    assert all(isinstance(x, int) for s in lattice.shifts for x in s)
    assert torch.equal(torch.tensor(lattice.shifts, dtype=lattice.c.dtype), lattice.c)


# --- moment identities ------------------------------------------------------------


def test_weights_are_positive_and_normalised(lattice, dtype):
    assert torch.all(lattice.w > 0)
    assert torch.isclose(lattice.w.sum(), torch.tensor(1.0, dtype=dtype), atol=TOL[dtype])


def test_first_moment_vanishes(lattice, dtype):
    m1 = torch.einsum("q,qa->a", lattice.w, lattice.c)
    assert torch.allclose(m1, torch.zeros(lattice.D, dtype=dtype), atol=TOL[dtype])


def test_second_moment_is_isotropic(lattice, dtype):
    m2 = torch.einsum("q,qa,qb->ab", lattice.w, lattice.c, lattice.c)
    assert torch.allclose(m2, lattice.cs2 * _eye(lattice, dtype), atol=TOL[dtype])


def test_third_moment_vanishes(lattice, dtype):
    m3 = torch.einsum("q,qa,qb,qc->abc", lattice.w, lattice.c, lattice.c, lattice.c)
    assert torch.allclose(m3, torch.zeros_like(m3), atol=TOL[dtype])


def test_isotropy_order_is_declared(lattice):
    assert isinstance(lattice.isotropy_order, int)
    assert lattice.isotropy_order >= 2 and lattice.isotropy_order % 2 == 0


def test_fourth_moment_is_isotropic(lattice, dtype):
    if lattice.isotropy_order < 4:
        pytest.skip(
            f"{type(lattice).__name__} claims isotropy only to order {lattice.isotropy_order}"
        )
    expected = lattice.cs2**2 * _isotropic_tensor(4, _eye(lattice, dtype))
    assert torch.allclose(_moment(lattice, 4), expected, atol=10 * TOL[dtype])


def test_fifth_moment_vanishes(lattice, dtype):
    if lattice.isotropy_order < 6:
        pytest.skip("odd moments above third are only required for sixth-order lattices")
    m5 = _moment(lattice, 5)
    assert torch.allclose(m5, torch.zeros_like(m5), atol=10 * TOL[dtype])


def test_sixth_moment_is_isotropic(lattice, dtype):
    if lattice.isotropy_order < 6:
        pytest.skip(
            f"{type(lattice).__name__} claims isotropy only to order {lattice.isotropy_order}"
        )
    expected = lattice.cs2**3 * _isotropic_tensor(6, _eye(lattice, dtype))
    assert torch.allclose(_moment(lattice, 6), expected, atol=100 * TOL[dtype])


def test_seventh_moment_vanishes(lattice, dtype):
    if lattice.isotropy_order < 8:
        pytest.skip("odd moments above fifth are only required for eighth-order lattices")
    m7 = _moment(lattice, 7)
    assert torch.allclose(m7, torch.zeros_like(m7), atol=1e3 * TOL[dtype])


def test_eighth_moment_is_isotropic(lattice, dtype):
    if lattice.isotropy_order < 8:
        pytest.skip(
            f"{type(lattice).__name__} claims isotropy only to order {lattice.isotropy_order}"
        )
    expected = lattice.cs2**4 * _isotropic_tensor(8, _eye(lattice, dtype))
    assert torch.allclose(_moment(lattice, 8), expected, atol=1e3 * TOL[dtype])


def test_d2q37_sound_speed_matches_weights():
    lat = lattices.D2Q37(dtype=torch.float64)
    m2 = torch.einsum("q,qa,qb->ab", lat.w, lat.c, lat.c)
    assert torch.allclose(m2, lat.cs2 * torch.eye(2, dtype=torch.float64), atol=1e-14)
    assert torch.isclose(lat.cs2, torch.tensor(1 / lattices.D2Q37.R**2, dtype=torch.float64))


def test_d2q5_is_not_fourth_order_isotropic():
    """Documents why D2Q5 is scalar-transport only: the claimed order is honest."""
    lat = lattices.D2Q5(dtype=torch.float64)
    expected = lat.cs2**2 * _isotropic_tensor(4, _eye(lat, torch.float64))
    assert not torch.allclose(_moment(lat, 4), expected, atol=1e-3)


# --- opposite-direction table ---------------------------------------------------------


def test_opp_is_an_involution(lattice):
    assert lattice.opp.dtype == torch.int64
    assert torch.equal(lattice.opp[lattice.opp], torch.arange(lattice.Q))
    assert lattice.opp[0] == 0


def test_opp_reverses_velocity(lattice):
    assert torch.equal(lattice.c[lattice.opp], -lattice.c)


def test_opp_preserves_weight(lattice):
    assert torch.equal(lattice.w[lattice.opp], lattice.w)


# --- buffers, dtype and device --------------------------------------------------------


def test_constants_are_built_in_requested_dtype(lattice, dtype):
    assert lattice.c.dtype == dtype
    assert lattice.w.dtype == dtype
    assert lattice.cs2.dtype == dtype
    assert lattice.opp.dtype == torch.int64


def test_fp64_constants_are_not_upcast_fp32(lattice_cls):
    """Building in fp64 must round the exact fractions, not up-cast fp32 values."""
    w32 = lattice_cls(dtype=torch.float32).w.double()
    w64 = lattice_cls(dtype=torch.float64).w
    assert not torch.equal(w32, w64)
    assert torch.allclose(w32, w64, atol=1e-7)


def test_module_to_dtype_casts_float_buffers_only(lattice_cls):
    lat = lattice_cls().to(torch.float64)
    assert lat.w.dtype == lat.c.dtype == lat.cs2.dtype == torch.float64
    assert lat.opp.dtype == torch.int64


def test_buffers_are_non_persistent_and_not_parameters(lattice_cls):
    lat = lattice_cls()
    assert list(lat.parameters()) == []
    assert lat.state_dict() == {}
    assert {n for n, _ in lat.named_buffers()} == {"c", "w", "opp", "cs2"}


def test_module_moves_to_selected_device(lattice_cls):
    device = gpu.get_device()
    lat = lattice_cls().to(device)
    for name, buf in lat.named_buffers():
        assert buf.device.type == device.type, name
    assert torch.equal(lat.c[lat.opp].cpu(), -lat.c.cpu())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_module_moves_to_cuda(lattice_cls):
    lat = lattice_cls().to("cuda")
    assert all(b.is_cuda for _, b in lat.named_buffers())
    f = torch.rand(lat.Q, 4, 4, device="cuda")
    assert f[lat.opp].shape == f.shape  # opp indexes f on the same device


# --- construction-time validation -------------------------------------------------


class _Bad(lattices.Lattice):
    Q = 3
    D = 1
    isotropy_order = 2


def _bad(c, w=None, opp=None):
    w = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64) if w is None else w
    opp = torch.tensor([0, 2, 1]) if opp is None else opp
    return _Bad(c=c, w=w, opp=opp, cs2=0.5)


def test_rejects_non_integer_velocities():
    with pytest.raises(ValueError, match="integer-valued"):
        _bad(torch.tensor([[0.0], [0.5], [-0.5]], dtype=torch.float64))


def test_rejects_wrong_shapes():
    with pytest.raises(ValueError, match="shape"):
        _bad(torch.tensor([[0, 0], [1, 0], [-1, 0]]))  # D mismatch
    with pytest.raises(ValueError, match="shape"):
        _bad(torch.tensor([[0], [1], [-1]]), w=torch.tensor([1.0, 0.0]))  # Q mismatch
