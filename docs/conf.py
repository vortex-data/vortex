import doctest
import re
from pathlib import Path

import hawkmoth.docstring
from sphinx.util import logging

log = logging.getLogger("vortex.docs.conf")

# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Vortex"
copyright = "The Vortex contributors"
author = "Vortex contributors"

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

hawkmoth_transform_default = "c_to_rust"

# Track the hawkmoth references so we can warn if they are not all registered!
C_DOCS: set[str] | None = None


def _replace_rust_references(app, lines, transform, options):
    """Replace Rust references with C equivalents in hawkmoth docstrings.

    See: https://hawkmoth.readthedocs.io/en/stable/extending.html#event-hawkmoth-process-docstring
    """
    if transform != "c_to_rust":
        # Not for us!
        return

    import sys

    # This is one of my finest hacks...
    # Hawkmoth doesn't expose type information to us. So we grab it from the caller's stack frame locals.
    stack_frame = sys._getframe(6)
    docs: hawkmoth.docstring.RootDocstring = stack_frame.f_locals["root"]

    global C_DOCS
    if C_DOCS is None:
        C_DOCS = set(
            d._name
            for d in docs.walk(
                recurse=False,  # Ignore e.g. enum members
                filter_types=(
                    hawkmoth.docstring.FunctionDocstring,
                    hawkmoth.docstring.EnumDocstring,
                    hawkmoth.docstring.UnionDocstring,
                    hawkmoth.docstring.StructDocstring,
                ),
            )
        )

    # Remove the current docstring from the set of C docs
    slf = stack_frame.f_locals["self"]
    C_DOCS.discard(slf.arguments[0])

    # Pattern to match [`crate::path::to::function`]
    pattern = r"\[`([^:]+::)*?(vx_[^`]+)`\]"

    def replace_match(match):
        # Extract the function name (already starts with vx_)
        # TODO(ngates): detect if the reference is a function or a type
        func_name = match.group(2)

        refs = list(docs.walk(filter_names=[func_name]))
        if not refs:
            # If we can't find the function, return the original match without a reference
            return func_name
        ref = refs[0]
        if isinstance(ref, hawkmoth.docstring.FunctionDocstring):
            # If it's a function, return the C identifier
            return f":c:func:`{func_name}`"
        elif isinstance(ref, hawkmoth.docstring.EnumDocstring):
            # If it's an enum, return the C identifier
            return f":c:type:`{func_name}`"
        elif isinstance(ref, hawkmoth.docstring.TypedefDocstring):
            # If it's a typedef, return the C identifier
            return f":c:type:`{func_name}`"
        elif isinstance(ref, hawkmoth.docstring.StructDocstring):
            # If it's a typedef, return the C identifier
            return f":c:type:`{func_name}`"
        else:
            return func_name

    for i, line in enumerate(lines):
        lines[i] = re.sub(pattern, replace_match, line)


def _post_process(app, builder):
    """Post-process the documentation after writing."""
    global C_DOCS
    if C_DOCS:
        # TODO(ngates): enable this one we've cleaned up the entire C API.
        # log.warning("Some C references were not found: %s", ", ".join(sorted(C_DOCS)))
        C_DOCS = None  # Reset for next build


def setup(app):
    app.connect("hawkmoth-process-docstring", _replace_rust_references)
    app.connect("write-started", _post_process)
