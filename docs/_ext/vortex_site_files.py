"""
Build hooks for the URLs the site serves and the files crawlers read.
1. Link to each page by the URL it is served at (`page`, `dir/`), not its `.html` file,
   which the host redirects
2. Write sitemap.xml with the URL of every page
3. Write llms.txt (https://llmstxt.org), an index of the docs for language models
4. Write llms-full.txt, the plain text of every page
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import quote
from xml.sax.saxutils import escape

from sphinx.builders.text import TextBuilder
from sphinx.writers.text import TextTranslator

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

    from sphinx.application import Sphinx

# Pages that Sphinx generates and that hold no documentation.
_GENERATED = {"genindex", "py-modindex", "search"}


def _target_uri(docname: str, typ: str | None = None) -> str:
    if docname == "index":
        return ""
    if docname.endswith("/index"):
        return quote(docname.removesuffix("index"))
    return quote(docname)


def _clean_links(app: Sphinx) -> None:
    if app.builder.name == "html":
        app.builder.get_target_uri = _target_uri


def _url(app: Sphinx, docname: str) -> str:
    return app.config.html_baseurl.rstrip("/") + "/" + _target_uri(docname)


def _toctree_order(app: Sphinx, docname: str = "index") -> Iterator[str]:
    """Yield `docname` and the documents its toctrees include, in reading order."""
    yield docname
    for child in app.env.toctree_includes.get(docname, ()):
        yield from _toctree_order(app, child)


def _pages(app: Sphinx) -> list[str]:
    ordered = list(dict.fromkeys(_toctree_order(app)))
    rest = sorted(app.env.found_docs - set(ordered))
    return [doc for doc in ordered + rest if doc not in _GENERATED]


def _title(app: Sphinx, docname: str) -> str:
    return app.env.titles[docname].astext()


def _sitemap(app: Sphinx, pages: list[str]) -> str:
    urls = "".join(f"  <url><loc>{escape(_url(app, doc))}</loc></url>\n" for doc in pages)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{urls}"
        "</urlset>\n"
    )


def _summary(app: Sphinx) -> str:
    from docutils import nodes

    doctree = app.env.get_doctree("index")
    first = next(iter(doctree.findall(nodes.paragraph)), None)
    return " ".join(first.astext().split()) if first else ""


def _llms(app: Sphinx) -> str:
    base = app.config.html_baseurl.rstrip("/")
    lines = [f"# {app.config.project}", "", f"> {_summary(app)}", ""]
    lines += [
        f"The documentation for {app.config.project}, by section. "
        f"[llms-full.txt]({base}/llms-full.txt) holds the text of every page in one file.",
        "",
    ]
    for section in app.env.toctree_includes.get("index", ()):
        lines += [f"## {_title(app, section)}", ""]
        lines += [f"- [{_title(app, doc)}]({_url(app, doc)})" for doc in dict.fromkeys(_toctree_order(app, section))]
        lines.append("")
    lines += [
        "## Benchmarks and project",
        "",
        "- [Benchmark summaries](https://bench.vortex.dev/llms.txt): every benchmark group's rankings and methods",
        "- [Vortex homepage](https://vortex.dev/llms.txt): what Vortex is, how to install it, and its results",
        "",
        "## Optional",
        "",
        f"- [Full text of the documentation]({base}/llms-full.txt)",
        f"- [Java API reference (Javadoc)]({base}/_static/vortex-jni/)",
        f"- [Spark connector reference (Javadoc)]({base}/_static/vortex-spark/)",
        "- [Source code](https://github.com/vortex-data/vortex)",
        "",
    ]
    return "\n".join(lines)


def _llms_full(app: Sphinx, pages: list[str]) -> str:
    text_builder = TextBuilder(app, app.env)
    text_builder.init()
    parts = []
    for doc in pages:
        doctree = app.env.get_and_resolve_doctree(doc, app.builder)
        translator = TextTranslator(doctree, text_builder)
        doctree.walkabout(translator)
        parts.append(f"Source: {_url(app, doc)}\n\n{translator.body.strip()}\n")
    return "\n\n".join(parts)


def _write(app: Sphinx, exception: Exception | None) -> None:
    if exception is not None or app.builder.name != "html":
        return
    out = app.builder.outdir
    pages = _pages(app)
    (out / "sitemap.xml").write_text(_sitemap(app, pages), encoding="utf-8")
    (out / "llms.txt").write_text(_llms(app), encoding="utf-8")
    (out / "llms-full.txt").write_text(_llms_full(app, pages), encoding="utf-8")


def setup(app: Sphinx) -> dict[str, Any]:
    app.connect("builder-inited", _clean_links)
    app.connect("build-finished", _write)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
