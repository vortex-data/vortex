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
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pyarrow": ("https://arrow.apache.org/docs", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "polars": ("https://docs.pola.rs/api/python/stable", None),
}

nitpicky = True  # ensures all :class:, :obj:, etc. links are valid

doctest_global_setup = "import pyarrow; import vortex"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["style.css"]  # relative to _static/

# -- Options for PyData Theme ------------------------------------------------
html_theme_options = {
    "show_toc_level": 2,
    "logo": {
        "alt_text": "The Vortex logo.",
        "text": "Vortex",
        "image_light": "_static/vortex_spiral_logo.svg",
        "image_dark": "_static/vortex_spiral_logo_dark_theme.svg",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/spiraldb/vortex",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/vortex-array",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ],
    "header_links_before_dropdown": 3,
}
html_sidebars = {
    # hide the primary (left-hand) sidebar on pages without sub-pages
    "quickstart": [],
    "guide": [],
    "file_format": [],
}
