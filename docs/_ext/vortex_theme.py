"""
Build hooks for our theme.
1. Add pygment styles to vortex.css
2. Load only one CSS and no JS per page except search page
3. Rewrite TOC into detail/summary with regex so it loads without JS
4. Minify CSS
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import rcssmin
from pygments.formatters.html import HtmlFormatter

if TYPE_CHECKING:
    from typing import Any

    from sphinx.application import Sphinx

_LIGHT_STYLE = "a11y-high-contrast-light"
_DARK_STYLE = "a11y-high-contrast-dark"

_DOCS_DIR = Path(__file__).parent.parent
_CSS_SOURCE = _DOCS_DIR / "_theme" / "vortex" / "vortex.css"
_GEN_STATIC = _DOCS_DIR / "_build" / "_gen_static"

_PAGE_CSS = {"_static/vortex.css"}
_PAGE_JS: set[str] = set()
_SEARCH_JS = {
    "_static/documentation_options.js",
    "_static/doctools.js",
    "_static/sphinx_highlight.js",
}


def _css(app: Sphinx) -> None:
    if app.builder.format != "html":
        return
    light = HtmlFormatter(style=_LIGHT_STYLE).get_style_defs(".highlight")
    dark = HtmlFormatter(style=_DARK_STYLE).get_style_defs(".highlight")
    css = rcssmin.cssmin(f"{_CSS_SOURCE.read_text()}\n{light}\n@media (prefers-color-scheme: dark) {{\n{dark}\n}}\n")
    out = _GEN_STATIC / "vortex.css"
    if not out.exists() or out.read_text() != css:
        out.write_text(css)


def _filter_assets(app: Sphinx, pagename: str, templatename: str, context: dict[str, Any], doctree: Any) -> None:
    allowed_js = _SEARCH_JS if pagename == "search" else _PAGE_JS
    context["script_files"] = [
        js for js in context.get("script_files", ()) if str(getattr(js, "filename", js)) in allowed_js
    ]
    context["css_files"] = [
        css for css in context.get("css_files", ()) if str(getattr(css, "filename", css)) in _PAGE_CSS
    ]


_SUMMARY = '<summary aria-label="Toggle section"></summary>'
_TOC_BRANCH = re.compile(r"</a>\s*<ul([^>]*)>")
_TOC_BRANCH_END = re.compile(r"</ul>\s*</li>")
_TOC_CURRENT_BRANCH = re.compile(rf"<details>({re.escape(_SUMMARY)}<ul[^>]*\bcurrent\b)")
_TOC_CURRENT_PARENT = re.compile(r'(<a aria-current="page" (?:(?!</a>).)*</a>)<details>', re.S)
_TOC_CURRENT_LINK = re.compile(r'<a class="([^"]*\bcurrent\b[^"]*)"')


def _toc(app: Sphinx, pagename: str, templatename: str, context: dict[str, Any], doctree: Any) -> None:
    toctree = context.get("toctree")
    if toctree is None:
        context["vortex_globaltoc"] = ""
        return
    html = toctree(maxdepth=3, collapse=False, includehidden=True, titles_only=True) or ""
    html = _TOC_CURRENT_LINK.sub(r'<a aria-current="page" class="\1"', html)
    html = _TOC_BRANCH.sub(rf"</a><details>{_SUMMARY}<ul\1>", html)
    html = _TOC_CURRENT_BRANCH.sub(r"<details open>\1", html)
    html = _TOC_CURRENT_PARENT.sub(r"\1<details open>", html)
    html = _TOC_BRANCH_END.sub("</ul></details></li>", html)
    context["vortex_globaltoc"] = html


def setup(app: Sphinx) -> dict[str, Any]:
    _GEN_STATIC.mkdir(parents=True, exist_ok=True)
    app.connect("builder-inited", _css)
    app.connect("html-page-context", _filter_assets)
    app.connect("html-page-context", _toc)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
