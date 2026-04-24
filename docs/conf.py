import doctest
import os
import re
import shutil
import subprocess
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
    "breathe",  # C++ API (Doxygen -> Sphinx bridge)
    "hawkmoth",  # C API
    "myst_parser",  # Markdown support
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_inline_tabs",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.mermaid",
    "sphinxext.opengraph",
]

templates_path = ["_templates"]
html_show_sourcelink = False
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pyarrow": ("https://arrow.apache.org/docs", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/version/2.3/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "polars": ("https://docs.pola.rs/api/python/stable", "polars.objects.inv"),
}

git_root = Path(__file__).parent.parent

nitpicky = True  # ensures all :class:, :obj:, etc. links are valid
nitpick_ignore = []

doctest_global_setup = "import pyarrow; import vortex; import vortex as vx; import random; random.seed(a=0)"
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

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["style.css"]
html_favicon = "_static/vortex_logo.svg"  # relative to _static/

# -- Options for PyData Sphinx Theme ----------------------------------------

html_theme_options = {
    "logo": {
        "image_light": "_static/vortex_logo.svg",
        "image_dark": "_static/vortex_logo_dark_theme.svg",
    },
    "github_url": "https://github.com/vortex-data/vortex",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/vortex-data",
            "icon": "fa-brands fa-python",
        },
        {
            "name": "Crates.io",
            "url": "https://crates.io/crates/vortex",
            "icon": "fa-brands fa-rust",
        },
    ],
    "header_links_before_dropdown": 7,
    "navbar_align": "left",
    "show_nav_level": 2,
    "navigation_depth": 3,
    "show_toc_level": 2,
}

# -- Options for OpenGraph ---------------------------------------------------

ogp_site_url = "https://docs.vortex.dev"
ogp_image = "https://docs.vortex.dev/_static/vortex_logo.svg"

# -- Options for Sphinx BibTEX -------------------------------------------

bibtex_bibfiles = ["references.bib"]

# -- Options for Breathe C++ API gen ------------------------------------

_doxygen_xml_dir = str(Path(__file__).parent / "_build" / "doxygen-cpp" / "xml")

os.makedirs(os.path.dirname(_doxygen_xml_dir), exist_ok=True)

if not shutil.which("doxygen"):
    raise RuntimeError("doxygen is required to build the docs but was not found on PATH")
subprocess.run(["doxygen", "Doxyfile.cpp"], cwd=Path(__file__).parent, check=True)

breathe_projects = {"vortex-cpp": _doxygen_xml_dir}
breathe_default_project = "vortex-cpp"

# C++ types from cxx bridge and standard library that Sphinx cannot resolve.
nitpick_ignore += [
    ("cpp:identifier", t)
    for t in [
        "vortex",
        "rust",
        "ffi",
        "uint8_t",
        "uint16_t",
        "uint32_t",
        "uint64_t",
        "int8_t",
        "int16_t",
        "int32_t",
        "int64_t",
        "size_t",
        "std::size_t",
    ]
]
nitpick_ignore_regex = [
    # cxx bridge internals that will never be resolvable in Sphinx.
    (r"cpp:identifier", r"rust::.*"),
    (r"cpp:identifier", r"ffi::.*"),
    # Doxygen file-level labels (e.g. "dtype_8hpp") that we don't generate pages for.
    (r"ref", r".*_8hpp"),
]

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


# Most tools change their table formatting based on the perceived number of columns. Most will
# obey the COLUMNS environment variable (because they use `shutil.get_terminal_size()`), but
# some COUGH polars COUGH do not.
os.environ["COLUMNS"] = "80"
# https://github.com/pola-rs/polars/blob/8a55acce8bb822c549861c371b6d48dee6c3379f/crates/polars-core/src/fmt.rs#L720
os.environ["POLARS_TABLE_WIDTH"] = "80"


def _convert_python_fenced_blocks_from_rust_to_valid_reST_blocks(app, what, name, obj, options, lines: list[str]):
    """Remove Markdown-style code fences from Python docs written in Rust.

    We would like `cargo test` to Just Work (TM). Unfortunately, by default, it executes any
    code-block in any docstring even though we intend those docs to be *Python* doc tests.

    For example, the following is interpreted by Rust as Rust code (which it will try to doctest):

        /// >>> 1 + 1
        /// 3
        fn foo() {
        }

    What syntax can we use to communicate to Rust "This is not Rust code" but communicate to Python
    "This is Python code"? The following appears as executable code to both, so it does not work:

        /// .. code-block:: python
        ///
        ///     >>> 1 + 1
        ///     3
        fn foo() {
        }

    This does not appear to work unless we wrap all the code in braces or a function, which makes it
    not valid Python:

        /// #[no_run]
        /// >>> 1 + 1
        /// 3

    The following is executed by neither language and does not render properly (because it is not
    valid reStructured Text):

        /// ```python
        /// >>> 1 + 1
        /// 3
        /// ```

    Okay, so, our solution is to just adopt the last option and explicitly remove the code fences
    when we parse docstrings in Sphinx.

    """
    in_block = False
    for i, line in enumerate(lines):
        if line == "```python":
            lines[i] = ""
            in_block = True
        elif in_block and line == "```":
            lines[i] = ""
            in_block = False


def _resolve_breathe_cpp_references(app, env, node, contnode):
    """Resolve relative C++ references emitted by Breathe.

    Breathe emits cross-namespace parameter types with relative qualifiers (e.g. ``scalar::Scalar``
    instead of ``vortex::scalar::Scalar``). This handler intercepts unresolved references and
    re-resolves them under the ``vortex::`` namespace.
    """
    if node.get("refdomain") != "cpp" or node.get("reftype") != "identifier":
        return None

    target = node.get("reftarget", "")
    if not target or target.startswith("vortex::"):
        return None

    cpp_domain = env.get_domain("cpp")
    # Try resolving with the vortex:: prefix.
    node = node.deepcopy()
    node["reftarget"] = f"vortex::{target}"
    return cpp_domain.resolve_xref(
        env, node.get("refdoc", ""), app.builder, "identifier", node["reftarget"], node, contnode
    )


def setup(app):
    app.connect("hawkmoth-process-docstring", _replace_rust_references)
    app.connect("write-started", _post_process)
    app.connect("autodoc-process-docstring", _convert_python_fenced_blocks_from_rust_to_valid_reST_blocks)
    app.connect("missing-reference", _resolve_breathe_cpp_references)
