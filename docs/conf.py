import doctest
from pathlib import Path

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Vortex"
copyright = "2024, Spiral"
author = "Spiral"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "hawkmoth",  # C API
    "myst_parser",  # Markdown support
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_inline_tabs",
    "sphinxcontrib.bibtex",
    "sphinxext.opengraph",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pyarrow": ("https://arrow.apache.org/docs", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "polars": ("https://docs.pola.rs/api/python/stable", None),
}

git_root = Path(__file__).parent.parent

nitpicky = True  # ensures all :class:, :obj:, etc. links are valid
nitpick_ignore = []

doctest_global_setup = "import pyarrow; import vortex"
doctest_default_flags = (
    doctest.ELLIPSIS | doctest.IGNORE_EXCEPTION_DETAIL | doctest.DONT_ACCEPT_TRUE_FOR_1 | doctest.NORMALIZE_WHITESPACE
)

# -- Options for MyST Parser -------------------------------------------------

myst_enable_extensions = [
    "colon_fence",  # Use ::: for Sphinx directives
]
myst_heading_anchors = 3

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["style.css"]  # relative to _static/

# -- Options for Furo Theme ------------------------------------------------

html_theme_options = {
    "light_logo": "vortex_spiral_logo.svg",
    "dark_logo": "vortex_spiral_logo_dark_theme.svg",
}

# -- Options for OpenGraph ---------------------------------------------------

ogp_site_url = "https://docs.vortex.dev"
ogp_image = "https://docs.vortex.dev/_static/vortex_spiral_logo.svg"

# -- Options for Sphinx BibTEX -------------------------------------------

bibtex_bibfiles = ["references.bib"]

# -- Options for hawkmoth C API gen ----------------------------

hawkmoth_root = str(git_root / "vortex-ffi/cinclude")

# C types that aren't keywords are not found, so we need to ignore them.
nitpick_ignore += [
    ("c:identifier", "bool"),
    ("c:identifier", "usize_t"),
    ("c:identifier", "size_t"),
    ("c:identifier", "uint64_t"),
    ("c:identifier", "int64_t"),
    ("c:identifier", "uint32_t"),
    ("c:identifier", "int32_t"),
    ("c:identifier", "uint16_t"),
    ("c:identifier", "int16_t"),
    ("c:identifier", "uint8_t"),
    ("c:identifier", "int8_t"),
]
