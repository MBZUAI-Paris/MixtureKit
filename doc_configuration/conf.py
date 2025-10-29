# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sphinx_bootstrap_theme
import sphinx_gallery

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MixtureKit"
copyright = "2025, MBZUAI"
author = "Ahmad Chamma"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.bibtex",
    "sphinx_gallery.gen_gallery",
    "numpydoc",
]

# The master toctree document.
master_doc = "index"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# generate autosummary even if no references
autosummary_generate = True

# sphinxcontrib-bibtex
bibtex_bibfiles = ["./references.bib"]
bibtex_style = "unsrt"
bibtex_reference_style = "author_year"
bibtex_footbibliography_header = ""

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "bootstrap"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "navbar_sidebarrel": False,
    "navbar_pagenav": False,
    "source_link_position": "",
    "navbar_links": [
        ("Examples", "auto_examples/index"),
        ("API", "api"),
        ("GitHub", "https://github.com/MBZUAI-Paris/MixtureKit", True),
    ],
    "bootswatch_theme": "flatly",
    "bootstrap_version": "3",
}

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

examples_dirs = ["../examples"]
gallery_dirs = ["auto_examples"]

sphinx_gallery_conf = {
    "doc_module": "groupmne",
    "reference_url": dict(groupmne=None),
    "examples_dirs": examples_dirs,
    "gallery_dirs": gallery_dirs,
    "plot_gallery": "True",
    "thumbnail_size": (160, 112),
    "min_reported_time": 1.0,
    "backreferences_dir": os.path.join("generated"),
    "abort_on_example_error": False,
    "show_memory": False,
    "download_all_examples": False,
    "show_signature": False,
    "notebook_extensions": {},
}


def setup(app):
    app.add_css_file("style.css")
