"""
Build hooks for our theme.
1. Join the Vortex design system's tokens and docs rules with the theme's own CSS into one vortex.css,
   and fail the build if a copy of the system was edited by hand
2. Load only one CSS and no JS per page except search page
3. Rewrite TOC into detail/summary with regex so it loads without JS
4. Minify CSS, and after the build drop the rules and custom properties that no page uses
"""

from __future__ import annotations

import hashlib
import re
import zlib
from html.parser import HTMLParser
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


# The design system's CSS holds rules for every surface. After the build, keep a rule only if each
# class, id, element, and attribute in one of its selectors is in some page or in a script that a
# page loads. Pseudo-classes and attribute values are not checked, so :hover, :focus-visible,
# :checked, and [data-theme="dark"] stay. Rules on the open, checked, and aria-* attributes stay even
# when no built page has them, since the reader or a script sets them at run time.
_STATE_ATTRS = re.compile(r"open|checked|aria-[\w-]+")
_SCRIPT_WORD = re.compile(r"[A-Za-z_][\w-]*")
_SCRIPT_SRC = re.compile(r'<script[^>]*\bsrc="([^"?]+)')
_CSS_LINK = re.compile(r"(_static/vortex\.css\?v=)[0-9a-f]{8}")
_CSS_TOKEN = re.compile(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|[(){}\[\],;]|[^\"'(){}\[\],;]+")
_SIMPLE = re.compile(
    r"(?P<fn>::?[\w-]+)\("
    r"|(?P<pseudo>::?[\w-]+)"
    r"|\.(?P<cls>[\w-]+)"
    r"|#(?P<id>[\w-]+)"
    r"|\[\s*(?P<attr>[\w-]+)(?:\"[^\"]*\"|'[^']*'|[^\]])*\]"
    r"|(?P<type>[A-Za-z][\w-]*|\*)"
    r"|(?P<comb>[\s>+~]+)"
)
_RELATIVE = re.compile(r"([>+~]?\s*)(.*)", re.S)
_LEGACY_ELEMENTS = {":before", ":after", ":first-line", ":first-letter"}
_NESTED = ("@media", "@supports")
_VAR = re.compile(r"var\(\s*(--[\w-]+)")


class _Names(HTMLParser):
    """The elements, attributes, classes, and ids in the built pages, and the scripts they load."""

    def __init__(self) -> None:
        super().__init__()
        self.names: dict[str, set[str]] = {"type": {"*"}, "attr": set(), "cls": set(), "id": set()}
        self.scripts: set[Path] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.names["type"].add(tag)
        for name, value in attrs:
            self.names["attr"].add(name)
            if name == "class" and value:
                self.names["cls"].update(value.split())
            elif name == "id" and value:
                self.names["id"].add(value)


def _split(text: str, sep: str) -> list[str]:
    """Split CSS text at each top-level `sep`: one outside strings, brackets, and parentheses."""
    parts, depth, start, pos = [], 0, 0, 0
    for token in _CSS_TOKEN.findall(text):
        if token in "([":
            depth += 1
        elif token in ")]":
            depth -= 1
        elif token == sep and depth == 0:
            parts.append(text[start:pos])
            start = pos + 1
        pos += len(token)
    parts.append(text[start:])
    return parts


def _close(text: str, start: int) -> int:
    """The index of the bracket that closes the one just before `start`."""
    depth, pos = 1, start
    for token in _CSS_TOKEN.findall(text[start:]):
        if token in "({[":
            depth += 1
        elif token in ")}]":
            depth -= 1
            if depth == 0:
                return pos
        pos += len(token)
    raise ExtensionError(f"unbalanced CSS: {text[start - 1 : start + 80]}")


def _selector(sel: str, names: dict[str, set[str]]) -> tuple[bool, str, tuple[int, int, int]]:
    """Whether a complex selector can match, the selector with its unused :is() and :where()
    arguments removed, and its specificity, which the removal keeps."""
    ok, out, spec, pos = True, [], [0, 0, 0], 0
    while pos < len(sel):
        m = _SIMPLE.match(sel, pos)
        if not m:
            raise ExtensionError(f"cannot read the CSS selector {sel!r}")
        kind, pos = m.lastgroup, m.end()
        if kind == "fn":
            end = _close(sel, pos)
            name, inner = m.group("fn"), sel[pos:end]
            if name in (":is", ":where", ":has", ":not"):
                # An argument of :has() can start with a combinator.
                parts = [_RELATIVE.match(p.strip()).groups() for p in _split(inner, ",")]
                args = [_selector(rest, names) for _, rest in parts]
                top = max(a[2] for a in args)
                if name != ":where":
                    spec = [s + t for s, t in zip(spec, top)]
                if name != ":not":
                    # Keep the arguments that can match. :is() and :has() take the specificity of
                    # their most specific argument, so keep that one too.
                    kept = [a[0] for a in args]
                    ok = ok and any(kept)
                    if any(kept) and name != ":where" and max(a[2] for a, k in zip(args, kept) if k) < top:
                        kept[[a[2] for a in args].index(top)] = True
                    inner = ",".join(
                        comb + (a[1] if a[0] else rest) for (comb, rest), a, k in zip(parts, args, kept) if k
                    )
            else:
                spec[2 if name.startswith("::") else 1] += 1
            out.append(f"{name}({inner})")
            pos = end + 1
            continue
        text = m.group(0)
        if kind == "pseudo":
            spec[2 if text.startswith("::") or text in _LEGACY_ELEMENTS else 1] += 1
        elif kind in ("cls", "attr"):
            spec[1] += 1
            value = m.group(kind)
            ok = ok and (value in names[kind] or (kind == "attr" and bool(_STATE_ATTRS.fullmatch(value))))
        elif kind == "id":
            spec[0] += 1
            ok = ok and m.group("id") in names["id"]
        elif kind == "type":
            spec[2] += text != "*"
            ok = ok and text.lower() in names["type"]
        out.append(text)
    return ok, "".join(out), (spec[0], spec[1], spec[2])


def _rules(css: str, names: dict[str, set[str]]) -> list[tuple[str, list[Any]]]:
    """Parse minified CSS into (prelude, declarations) and (@media prelude, rules), with the
    selectors that cannot match removed."""
    rules: list[tuple[str, list[Any]]] = []
    pos = 0
    while pos < len(css):
        brace = css.index("{", pos)
        end = _close(css, brace + 1)
        prelude, body = css[pos:brace].strip(), css[brace + 1 : end]
        pos = end + 1
        if prelude.startswith(_NESTED):
            rules.append((prelude, _rules(body, names)))
        elif prelude.startswith("@"):
            rules.append((prelude, _split(body, ";")))
        else:
            selectors = [s for ok, s, _ in (_selector(p.strip(), names) for p in _split(prelude, ",")) if ok]
            rules.append((",".join(selectors), _split(body, ";")))
    return rules


def _declarations(rules: list[tuple[str, list[Any]]]) -> list[str]:
    return [d for prelude, body in rules for d in (_declarations(body) if prelude.startswith(_NESTED) else body)]


def _drop(rules: list[tuple[str, list[Any]]], used: set[str]) -> list[tuple[str, list[Any]]]:
    """The rules without the custom properties outside `used`, and without empty rules."""
    out = []
    for prelude, body in rules:
        if prelude.startswith(_NESTED):
            body = _drop(body, used)
        else:
            body = [d for d in body if not d.strip().startswith("--") or d.split(":", 1)[0].strip() in used]
        if prelude and body:
            out.append((prelude, body))
    return out


def _write(rules: list[tuple[str, list[Any]]]) -> str:
    return "".join(
        f"{prelude}{{{_write(body) if prelude.startswith(_NESTED) else ';'.join(body)}}}" for prelude, body in rules
    )


def _prune(app: Sphinx, exception: Exception | None) -> None:
    if exception or app.builder.format != "html":
        return
    outdir = Path(app.outdir)
    pages = sorted(outdir.rglob("*.html"))
    names = _Names()
    for page in pages:
        text = page.read_text()
        names.feed(text)
        names.close()
        names.reset()
        names.scripts.update(
            (page.parent / src).resolve() for src in _SCRIPT_SRC.findall(text) if not src.endswith("searchindex.js")
        )
    words = set()
    for script in names.scripts:
        if script.is_file():
            words.update(_SCRIPT_WORD.findall(script.read_text()))
    for kind in names.names.values():
        kind.update(words)
    rules = _rules((_GEN_STATIC / "vortex.css").read_text(), names.names)
    # Drop custom properties that nothing reads, until none is left: a token can read another.
    while True:
        used = {m for d in _declarations(rules) for m in _VAR.findall(d)}
        pruned = _drop(rules, used)
        if pruned == rules:
            break
        rules = pruned
    css = _write(rules)
    (outdir / "_static" / "vortex.css").write_text(css)
    # Sphinx versions the link by a checksum of the CSS it copied, before the pages existed.
    version = f"{zlib.crc32(css.encode()):08x}"
    for page in pages:
        text = page.read_text()
        linked = _CSS_LINK.sub(rf"\g<1>{version}", text)
        if linked != text:
            page.write_text(linked)


def setup(app: Sphinx) -> dict[str, Any]:
    _GEN_STATIC.mkdir(parents=True, exist_ok=True)
    app.connect("builder-inited", _css)
    app.connect("html-page-context", _filter_assets)
    app.connect("html-page-context", _toc)
    app.connect("build-finished", _prune)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
