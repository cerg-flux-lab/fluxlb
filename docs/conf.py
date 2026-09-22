# Copyright 2026 Muaaz Bhamjee
# SPDX-License-Identifier: Apache-2.0
"""Sphinx configuration for the fluxlb API reference site.

Division of labour (see CLAUDE.md, "Working mode"): this site is the API reference, built
from the docstrings, plus the roadmap and tutorials. The narrative user guide, including all
theory, lives in LaTeX under ``docs/guide`` and is published here as a PDF only. Do not add
theory prose to these pages.
"""

from __future__ import annotations

import importlib.metadata

project = "fluxlb"
author = "Muaaz Bhamjee"
copyright = "2026, Muaaz Bhamjee"
try:
    release = importlib.metadata.version("fluxlb")
except importlib.metadata.PackageNotFoundError:  # building without an install
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "guide/**", "Thumbs.db", ".DS_Store"]

# Docstrings follow the numpy convention (pyproject: ruff pydocstyle convention = numpy).
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
always_document_param_types = True
typehints_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

myst_enable_extensions = ["dollarmath", "colon_fence", "deflist"]
suppress_warnings = ["myst.header"]  # the roadmap uses H4 under H2 deliberately
myst_heading_anchors = 3

html_theme = "furo"
html_title = "fluxlb"
html_static_path: list[str] = []
html_theme_options = {
    "source_repository": "https://github.com/cerg-flux-lab/fluxlb",
    "source_branch": "main",
    "source_directory": "docs/",
}
