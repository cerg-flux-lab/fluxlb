# Copyright 2026 Muaaz Bhamjee
# SPDX-License-Identifier: Apache-2.0
"""Moments and equilibrium: shapes, dtype policy, rest state, and moment recovery.

The moment-recovery tests are the acceptance criteria for the second-order equilibrium:
the zeroth and first moments must be recovered on every lattice, and the second moment
``rho (cs2 I + u u)`` on every lattice that claims fourth-order isotropy. D2Q5 is only
second-order isotropic, so the second-moment test is skipped there by design.
"""

import pytest
import torch

from fluxlb.core import lattices
from fluxlb.core.equilibrium import equilibrium, moments

LATTICES = [lattices.D2Q5, lattices.D2Q9, lattices.D2Q37, lattices.D3Q19, lattices.D3Q27]
DTYPES = [torch.float32, torch.float64]
TOL = {torch.float32: 1e-5, torch.float64: 1e-12}
GRID = {2: (4, 5), 3: (3, 4, 5)}
U_MAX = 0.05  # small Mach so the truncated expansion is well inside its validity


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
def fields(lattice, dtype):
    """A smooth, positive density and a small velocity field on a tiny grid."""
    gen = _gen()
    spatial = GRID[lattice.D]
    rho = 1.0 + 0.1 * torch.rand(spatial, generator=gen, dtype=dtype)
    u = U_MAX * (2 * torch.rand((lattice.D, *spatial), generator=gen, dtype=dtype) - 1)
    return rho, u


def _eye(lattice, dtype):
    return torch.eye(lattice.D, dtype=dtype)


# --- moments ------------------------------------------------------------------------


@pytest.mark.smoke
def test_moments_shapes_and_dtype(lattice, dtype):
    spatial = GRID[lattice.D]
    f = torch.rand((lattice.Q, *spatial), generator=_gen(), dtype=dtype)
    rho, u = moments(lattice, f)
    assert rho.shape == spatial
    assert u.shape == (lattice.D, *spatial)
    assert rho.dtype == dtype and u.dtype == dtype


def test_moments_of_rest_populations(lattice, dtype):
    spatial = GRID[lattice.D]
    rho0 = 1.0 + 0.1 * torch.rand(spatial, generator=_gen(), dtype=dtype)
    f = lattice.w.reshape(-1, *([1] * lattice.D)) * rho0
    rho, u = moments(lattice, f)
    assert torch.allclose(rho, rho0, atol=TOL[dtype])
    assert torch.allclose(u, torch.zeros_like(u), atol=TOL[dtype])


def test_moments_match_fp64_reference(lattice_cls):
    """fp32 moments must agree with an fp64 reduction to fp32 rounding, not fp32 summation."""
    lat32, lat64 = lattice_cls(dtype=torch.float32), lattice_cls(dtype=torch.float64)
    spatial = GRID[lat32.D]
    f64 = torch.rand((lat32.Q, *spatial), generator=_gen(), dtype=torch.float64)
    rho_ref, u_ref = moments(lat64, f64)
    rho, u = moments(lat32, f64.float())
    assert torch.allclose(rho.double(), rho_ref, rtol=2e-7, atol=0)
    assert torch.allclose(u.double(), u_ref, rtol=1e-6, atol=1e-7)


# --- equilibrium --------------------------------------------------------------------


@pytest.mark.smoke
def test_equilibrium_shape_and_dtype(lattice, fields, dtype):
    rho, u = fields
    feq = equilibrium(lattice, rho, u)
    assert feq.shape == (lattice.Q, *rho.shape)
    assert feq.dtype == dtype


def test_equilibrium_at_rest_is_weighted_density(lattice, fields, dtype):
    rho, u = fields
    feq = equilibrium(lattice, rho, torch.zeros_like(u))
    expected = lattice.w.reshape(-1, *([1] * lattice.D)) * rho
    assert torch.allclose(feq, expected, atol=TOL[dtype])


def test_equilibrium_recovers_density(lattice, fields, dtype):
    rho, u = fields
    feq = equilibrium(lattice, rho, u)
    assert torch.allclose(feq.sum(dim=0), rho, atol=TOL[dtype])


def test_equilibrium_recovers_momentum(lattice, fields, dtype):
    rho, u = fields
    feq = equilibrium(lattice, rho, u)
    j = torch.einsum("qa,q...->a...", lattice.c, feq)
    assert torch.allclose(j, rho * u, atol=TOL[dtype])


def test_equilibrium_second_moment(lattice, fields, dtype):
    if lattice.isotropy_order < 4:
        pytest.skip("second moment of f_eq needs fourth-order isotropy")
    rho, u = fields
    feq = equilibrium(lattice, rho, u)
    pi = torch.einsum("qa,qb,q...->ab...", lattice.c, lattice.c, feq)
    eye = _eye(lattice, dtype).reshape(lattice.D, lattice.D, *([1] * lattice.D))
    expected = rho * (lattice.cs2 * eye + u.unsqueeze(0) * u.unsqueeze(1))
    assert torch.allclose(pi, expected, atol=TOL[dtype])


def test_equilibrium_roundtrip_through_moments(lattice, fields, dtype):
    rho, u = fields
    rho2, u2 = moments(lattice, equilibrium(lattice, rho, u))
    assert torch.allclose(rho2, rho, atol=TOL[dtype])
    assert torch.allclose(u2, u, atol=TOL[dtype])


def test_equilibrium_is_positive_at_small_mach(lattice, fields):
    rho, u = fields
    assert torch.all(equilibrium(lattice, rho, u) > 0)


def test_equilibrium_default_order_is_the_lattice_order(lattice, fields):
    rho, u = fields
    default = equilibrium(lattice, rho, u)
    explicit = equilibrium(lattice, rho, u, order=lattice.equilibrium_order)
    assert torch.equal(default, explicit)


def test_equilibrium_order_above_lattice_raises(lattice, fields):
    rho, u = fields
    with pytest.raises(ValueError, match="outside"):
        equilibrium(lattice, rho, u, order=lattice.equilibrium_order + 1)


def test_equilibrium_order_zero_raises(lattice, fields):
    rho, u = fields
    with pytest.raises(ValueError, match="outside"):
        equilibrium(lattice, rho, u, order=0)


def _spatial_ones(lattice):
    return [1] * lattice.D


def test_equilibrium_third_moment(lattice, fields, dtype):
    """Order >= 3 must reproduce the Maxwell-Boltzmann third moment at unit temperature.

    TODO(maintainer): verify the expected tensor against the source you use for the
    Hermite coefficients.
    """
    if lattice.equilibrium_order < 3:
        pytest.skip("third moment of f_eq needs an order-3 truncation")
    rho, u = fields
    D, ones = lattice.D, _spatial_ones(lattice)
    feq = equilibrium(lattice, rho, u)
    pi3 = torch.einsum("qa,qb,qc,q...->abc...", lattice.c, lattice.c, lattice.c, feq)
    d = _eye(lattice, dtype)
    ua, ub, uc = u[:, None, None], u[None, :, None], u[None, None, :]
    dab = d.reshape(D, D, 1, *ones)
    dac = d.reshape(D, 1, D, *ones)
    dbc = d.reshape(1, D, D, *ones)
    expected = rho * (ua * ub * uc + lattice.cs2 * (ua * dbc + ub * dac + uc * dab))
    assert torch.allclose(pi3, expected, atol=TOL[dtype])


def test_equilibrium_fourth_moment(lattice, fields, dtype):
    """Order >= 4 must reproduce the Maxwell-Boltzmann fourth moment at unit temperature.

    TODO(maintainer): verify the expected tensor against the source you use for the
    Hermite coefficients.
    """
    if lattice.equilibrium_order < 4:
        pytest.skip("fourth moment of f_eq needs an order-4 truncation")
    rho, u = fields
    D, ones = lattice.D, _spatial_ones(lattice)
    feq = equilibrium(lattice, rho, u)
    pi4 = torch.einsum("qa,qb,qc,qd,q...->abcd...", lattice.c, lattice.c, lattice.c, lattice.c, feq)
    d = _eye(lattice, dtype)
    ua = u[:, None, None, None]
    ub = u[None, :, None, None]
    uc = u[None, None, :, None]
    ud = u[None, None, None, :]
    dab = d.reshape(D, D, 1, 1, *ones)
    dac = d.reshape(D, 1, D, 1, *ones)
    dad = d.reshape(D, 1, 1, D, *ones)
    dbc = d.reshape(1, D, D, 1, *ones)
    dbd = d.reshape(1, D, 1, D, *ones)
    dcd = d.reshape(1, 1, D, D, *ones)
    cs2, cs4 = lattice.cs2, lattice.cs2**2
    expected = rho * (
        ua * ub * uc * ud
        + cs2
        * (
            ua * ub * dcd
            + ua * uc * dbd
            + ua * ud * dbc
            + ub * uc * dad
            + ub * ud * dac
            + uc * ud * dab
        )
        + cs4 * (dab * dcd + dac * dbd + dad * dbc)
    )
    assert torch.allclose(pi4, expected, atol=TOL[dtype])


def test_equilibrium_follows_device_of_inputs(lattice, fields):
    """CPU-only check that no constant is materialised on a fixed device."""
    rho, u = fields
    feq = equilibrium(lattice, rho, u)
    assert feq.device == rho.device
