# Copyright 2026 Muaaz Bhamjee
# SPDX-License-Identifier: Apache-2.0
"""Packaging smoke test: the installed distribution imports and reports a version."""

import importlib.metadata

import pytest

import fluxlb


@pytest.mark.smoke
def test_package_imports_and_has_version() -> None:
    """Editable install exposes the package and setuptools-scm supplies a version."""
    assert fluxlb is not None
    assert importlib.metadata.version("fluxlb")
