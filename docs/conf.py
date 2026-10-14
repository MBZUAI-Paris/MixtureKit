# Sphinx configuration of the MixtureKit documentation.
#   pip install -e ".[doc]" && sphinx-build -W docs docs/_build/html

import os
import sys
from importlib import metadata

sys.path.insert(0, os.path.abspath(".."))

project = "MixtureKit"
author = "Ahmad Chamma, Omar El Herraoui, Guokan Shang"
copyright = "2025, MBZUAI"
try:
    release = metadata.version("MixtureKit")
except metadata.PackageNotFoundError:
    release = "1.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3
exclude_patterns = ["_build"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

html_theme = "furo"
html_title = "MixtureKit"
html_logo = "../logo.png"
html_theme_options = {
    "source_repository": "https://github.com/MBZUAI-Paris/MixtureKit",
    "source_branch": "main",
    "source_directory": "docs/",
}
