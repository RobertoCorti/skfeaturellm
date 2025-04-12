import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "skfeaturellm"
copyright = "2024, Roberto Corti, Stefeano Polo"
author = "Roberto Corti, Stefeano Polo"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Add logo configuration
html_logo = "_static/logo.png"
html_theme_options = {
    "logo_only": True,
    "display_version": True,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
}
