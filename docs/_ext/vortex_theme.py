"""
Build hooks for our theme.
1. Join the Vortex design system's tokens and docs rules with the theme's own CSS into one vortex.css,
   and fail the build if a copy of the system was edited by hand
2. Load only one CSS and no JS per page except search page
3. Rewrite TOC into detail/summary with regex so it loads without JS
4. Make the 404 page's relative URLs root-relative, since it is served at any address
5. Minify CSS
6. Read the repository's star count for the header's GitHub button, as vortex.dev does
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import rcssmin
from sphinx.errors import ExtensionError

if TYPE_CHECKING:
    from typing import Any

    from sphinx.application import Sphinx

_DOCS_DIR = Path(__file__).parent.parent
_THEME_DIR = _DOCS_DIR / "_theme" / "vortex"
# Copies of the design system, written by its tools/site-styles.mjs. Each starts with a header
# whose sha256 covers the rest of the file.
_SYSTEM_CSS = (_THEME_DIR / "system" / "vortex-tokens.css", _THEME_DIR / "system" / "vortex.css")
_CSS_SOURCE = _THEME_DIR / "vortex.css"
_SYSTEM_HEADER = re.compile(r"\A/\*.*?sha256 ([0-9a-f]{64}) \*/\n", re.S)
_GEN_STATIC = _DOCS_DIR / "_build" / "_gen_static"

_PAGE_CSS = {"_static/vortex.css"}
_PAGE_JS: set[str] = set()
_SEARCH_JS = {
    "_static/documentation_options.js",
    "_static/doctools.js",
    "_static/sphinx_highlight.js",
}


def _system_css(path: Path) -> str:
    text = path.read_text()
    header = _SYSTEM_HEADER.match(text)
    body = text[header.end() :] if header else ""
    if not header or hashlib.sha256(body.encode()).hexdigest() != header.group(1):
        raise ExtensionError(f"{path} does not match its sha256 header: edit the design system and copy it again")
    return body


def _css(app: Sphinx) -> None:
    if app.builder.format != "html":
        return
    css = rcssmin.cssmin("\n".join([*map(_system_css, _SYSTEM_CSS), _CSS_SOURCE.read_text()]))
    out = _GEN_STATIC / "vortex.css"
    if not out.exists() or out.read_text() != css:
        out.write_text(css)


def _filter_assets(app: Sphinx, pagename: str, templatename: str, context: dict[str, Any], doctree: object) -> None:
    allowed_js = _SEARCH_JS if pagename == "search" else _PAGE_JS
    context["script_files"] = [
        js for js in context.get("script_files", ()) if str(getattr(js, "filename", js)) in allowed_js
    ]
    context["css_files"] = [
        css for css in context.get("css_files", ()) if str(getattr(css, "filename", css)) in _PAGE_CSS
    ]


_TOC_BRANCH = re.compile(r">((?:(?!</a>).)*)</a>\s*<ul([^>]*)>", re.S)
_TOC_BRANCH_END = re.compile(r"</ul>\s*</li>")
_TOC_CURRENT_BRANCH = re.compile(r"<details>(<summary[^>]*></summary><ul[^>]*\bcurrent\b)")
_TOC_CURRENT_PARENT = re.compile(r'(<a aria-current="page" (?:(?!</a>).)*</a>)<details>', re.S)
_TOC_CURRENT_LINK = re.compile(r'<a class="([^"]*\bcurrent\b[^"]*)"')
_TAGS = re.compile(r"<[^>]+>")
# The page's own contents: one entry for the page title, holding the sections. The rail shows the
# sections alone, since the title is already the page's h1.
_PAGE_TOC = re.compile(r'\s*<ul>\s*<li><a [^>]*href="#">(?:(?!</a>).)*</a>(<ul>.*</ul>)\s*</li>\s*</ul>\s*', re.S)


def _branch(m: re.Match[str]) -> str:
    # Each toggle is named after its section, so a screen reader can tell them apart.
    label = _TAGS.sub("", m.group(1)).strip()
    return f'>{m.group(1)}</a><details><summary aria-label="{label} pages"></summary><ul{m.group(2)}>'


def _toc(app: Sphinx, pagename: str, templatename: str, context: dict[str, Any], doctree: object) -> None:
    page_toc = _PAGE_TOC.fullmatch(context.get("toc") or "")
    if page_toc:
        context["toc"] = page_toc.group(1)
    toctree = context.get("toctree")
    if toctree is None:
        context["vortex_globaltoc"] = ""
        return
    html = toctree(maxdepth=3, collapse=False, includehidden=True, titles_only=True) or ""
    html = _TOC_CURRENT_LINK.sub(r'<a aria-current="page" class="\1"', html)
    html = _TOC_BRANCH.sub(_branch, html)
    html = _TOC_CURRENT_BRANCH.sub(r"<details open>\1", html)
    html = _TOC_CURRENT_PARENT.sub(r"\1<details open>", html)
    html = _TOC_BRANCH_END.sub("</ul></details></li>", html)
    context["vortex_globaltoc"] = html


# The 404 page is served at whatever address missed, so its relative links and assets would resolve
# against that address. Every relative URL in it becomes root-relative: links, assets, and the
# header's search form.
_RELATIVE_URL = re.compile(r'((?:href|src|action)=")(?![a-z][a-z0-9+.-]*:|/|#)')


def _root_relative_404(app: Sphinx, exception: Exception | None) -> None:
    if exception is not None or app.builder.format != "html":
        return
    page = Path(app.outdir) / "404.html"
    if page.exists():
        page.write_text(_RELATIVE_URL.sub(r"\1/", page.read_text()))


# The header's GitHub button shows the star count, as on vortex.dev, so every surface draws the same
# button. It is read once per build. Any failure (offline, a timeout, a rate limit) leaves it out,
# and the button shows the mark alone.
_STARS_API = "https://api.github.com/repos/vortex-data/vortex"


def _format_stars(n: int) -> str:
    """vortex.dev's formatStars: "950", "3.3k", "12k"."""
    if n < 1000:
        return str(n)
    k = int(n / 100 + 0.5) / 10 if n < 9950 else int(n / 1000 + 0.5)
    return f"{k:g}k"


def _github_stars(app: Sphinx) -> None:
    stars = None
    try:
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        # A token, where the build has one, lifts the API's limit of 60 requests an hour.
        if token := os.environ.get("GITHUB_TOKEN"):
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(_STARS_API, headers=headers)
        with urllib.request.urlopen(request, timeout=5) as response:
            count = json.load(response).get("stargazers_count")
        if isinstance(count, int) and count >= 0:
            stars = _format_stars(count)
    except Exception:  # noqa: BLE001 - the count is optional
        stars = None
    app.config.html_context["github_stars"] = stars


def setup(app: Sphinx) -> dict[str, Any]:
    _GEN_STATIC.mkdir(parents=True, exist_ok=True)
    app.connect("builder-inited", _css)
    app.connect("builder-inited", _github_stars)
    app.connect("html-page-context", _filter_assets)
    app.connect("html-page-context", _toc)
    app.connect("build-finished", _root_relative_404)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
