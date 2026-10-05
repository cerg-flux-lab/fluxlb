# Copyright 2026 Muaaz Bhamjee
# SPDX-License-Identifier: Apache-2.0
"""Finite-difference gradient checks for moments() and equilibrium().

Run in fp64 on a tiny grid. Failure here means an in-place op, a ``.detach()``, or a
non-differentiable cast has crept into the path; it does not test the physics.
"""

import pytest
import torch

from fluxlb.core import lattices
from fluxlb.core.equilibrium import equilibrium, moments

LATTICES = [lattices.D2Q9, lattices.D2Q37, lattices.D3Q19]
GRID = {2: (3, 4), 3: (2, 3, 3)}

pytestmark = pytest.mark.gradcheck


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture(params=LATTICES, ids=lambda cls: cls.__name__)
def lattice(request):
    return request.param(dtype=torch.float64)


def test_moments_gradcheck(lattice):
    f = torch.rand((lattice.Q, *GRID[lattice.D]), dtype=torch.float64) + 0.5
    f.requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x: moments(lattice, x), (f,))


def test_equilibrium_gradcheck(lattice):
    spatial = GRID[lattice.D]
    rho = (1.0 + 0.1 * torch.rand(spatial, dtype=torch.float64)).requires_grad_(True)
    u = (0.05 * torch.randn((lattice.D, *spatial), dtype=torch.float64)).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda r, v: equilibrium(lattice, r, v), (rho, u))


def test_equilibrium_grad_flows_to_both_inputs(lattice):
    spatial = GRID[lattice.D]
    rho = torch.ones(spatial, dtype=torch.float64, requires_grad=True)
    u = (0.02 * torch.ones((lattice.D, *spatial), dtype=torch.float64)).requires_grad_(True)
    (equilibrium(lattice, rho, u) ** 2).sum().backward()
    assert rho.grad is not None and torch.all(torch.isfinite(rho.grad))
    assert u.grad is not None and torch.all(torch.isfinite(u.grad))
