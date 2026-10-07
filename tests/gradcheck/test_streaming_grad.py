# Copyright 2026 Muaaz Bhamjee
# SPDX-License-Identifier: Apache-2.0
"""Finite-difference gradient checks for stream().

Streaming is a permutation, so its Jacobian is a permutation matrix and the adjoint is the
same permutation applied backwards. Run in fp64 on a tiny grid. Failure here means an
in-place write or a non-differentiable op has entered the path; it does not test the physics.
"""

import pytest
import torch

from fluxlb.core import lattices
from fluxlb.core.streaming import stream

LATTICES = [lattices.D2Q9, lattices.D2Q37, lattices.D3Q19]
GRID = {2: (3, 4), 3: (2, 3, 3)}

pytestmark = pytest.mark.gradcheck


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture(params=LATTICES, ids=lambda cls: cls.__name__)
def lattice(request):
    return request.param(dtype=torch.float64)


def test_stream_gradcheck(lattice):
    f = torch.rand((lattice.Q, *GRID[lattice.D]), dtype=torch.float64) + 0.5
    f.requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x: stream(lattice, x), (f,))


def test_adjoint_is_the_inverse_permutation(lattice):
    """The vector-Jacobian product of a permutation is the inverse permutation: streaming the
    upstream gradient with every shift negated must reproduce ``f.grad``."""
    f = torch.rand((lattice.Q, *GRID[lattice.D]), dtype=torch.float64, requires_grad=True)
    v = torch.rand_like(f)
    (stream(lattice, f) * v).sum().backward()
    dims = tuple(range(lattice.D))
    expected = torch.stack(
        [torch.roll(v[q], [-s for s in lattice.shifts[q]], dims) for q in range(lattice.Q)]
    )
    assert torch.equal(f.grad, expected)


def test_grad_survives_a_rollout(lattice):
    """Several streams composed stay differentiable and the gradient is finite and non-trivial."""
    f = torch.rand((lattice.Q, *GRID[lattice.D]), dtype=torch.float64, requires_grad=True)
    g = f
    for _ in range(5):
        g = stream(lattice, g)
    (g**2).sum().backward()
    assert f.grad is not None and torch.all(torch.isfinite(f.grad))
    assert torch.allclose(f.grad, 2 * f.detach())
