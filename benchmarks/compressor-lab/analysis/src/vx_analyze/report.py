# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Report sections and the self-contained HTML they render into."""

from __future__ import annotations

import html
import math
from dataclasses import dataclass, field

import plotly.graph_objects as go
import polars as pl
from jinja2 import Template


@dataclass
class Table:
    df: pl.DataFrame
    caption: str = ""
    #: Column name -> format: "pct", "pct1", "x", "gbps", "int", "mb", "f2", "ms".
    formats: dict[str, str] = field(default_factory=dict)


@dataclass
class Code:
    text: str


@dataclass
class Section:
    key: str
    title: str
    lead: str
    blocks: list[Table | go.Figure | Code | str] = field(default_factory=list)


def _fmt(v, kind: str | None) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    try:
        match kind:
            case "pct":
                return f"{v:+.1%}" if v < 0 else f"{v:.1%}"
            case "spct":
                return f"{v:+.1%}"
            case "x":
                return f"{v:.2f}×"
            case "gbps":
                return f"{v:.1f}"
            case "int":
                return f"{int(v):,}"
            case "mb":
                return f"{v / 1e6:,.1f}"
            case "f2":
                return f"{v:.2f}"
            case "ms":
                return f"{v / 1e6:,.2f}"
    except (TypeError, ValueError):
        pass
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def table_html(t: Table) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in t.df.columns)
    body = []
    for row in t.df.iter_rows():
        cells = []
        for c, v in zip(t.df.columns, row):
            kind = t.formats.get(c)
            cls = ' class="num"' if kind or isinstance(v, (int, float)) else ""
            cells.append(f"<td{cls}>{html.escape(_fmt(v, kind))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    cap = f"<caption>{html.escape(t.caption)}</caption>" if t.caption else ""
    return f'<div class="tbl"><table>{cap}<thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


TEMPLATE = Template(
    """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
<style>
:root { --ink:#0b0b0b; --ink2:#52514e; --rule:#e6e5e1; --bg:#fcfcfb; --accent:#2a78d6; }
body { font-family: Inter, system-ui, sans-serif; color: var(--ink); background: var(--bg);
       margin: 0 auto; max-width: 1180px; padding: 24px 32px 80px; line-height: 1.45; }
h1 { font-size: 24px; margin: 0 0 4px; } h2 { font-size: 19px; margin: 40px 0 6px;
  padding-top: 16px; border-top: 1px solid var(--rule); }
.meta { color: var(--ink2); font-size: 13px; margin-bottom: 16px; }
.lead { max-width: 80ch; } nav a { margin-right: 14px; font-size: 13px; color: var(--accent); }
.note { color: var(--ink2); font-size: 13px; max-width: 80ch; }
.tbl { overflow-x: auto; margin: 12px 0; }
table { border-collapse: collapse; font-size: 13px; }
caption { text-align: left; color: var(--ink2); padding-bottom: 4px; }
th, td { padding: 4px 10px; border-bottom: 1px solid var(--rule); text-align: left; white-space: nowrap; }
th { font-weight: 600; color: var(--ink2); } td.num { text-align: right; font-variant-numeric: tabular-nums; }
.fig { margin: 8px 0 16px; }
pre { font-size: 12px; background: #f4f3f0; padding: 10px 12px; overflow-x: auto; }
</style></head><body>
<h1>{{ title }}</h1>
<div class="meta">{{ meta }}</div>
<nav>{% for s in sections %}<a href="#{{ s.key }}">{{ loop.index }}. {{ s.title }}</a>{% endfor %}</nav>
{% for s in sections %}
<h2 id="{{ s.key }}">{{ loop.index }}. {{ s.title }}</h2>
<p class="lead">{{ s.lead }}</p>
{% for b in s.rendered %}{{ b }}{% endfor %}
{% endfor %}
</body></html>"""
)


def render(title: str, meta: str, sections: list[Section]) -> str:
    first = [True]

    def block(b) -> str:
        if isinstance(b, Table):
            return table_html(b)
        if isinstance(b, go.Figure):
            out = b.to_html(
                full_html=False,
                include_plotlyjs="inline" if first[0] else False,
                config={"displaylogo": False, "responsive": True},
            )
            first[0] = False
            return f'<div class="fig">{out}</div>'
        if isinstance(b, Code):
            return f"<pre>{html.escape(b.text)}</pre>"
        return f'<p class="note">{html.escape(b)}</p>'

    for s in sections:
        s.rendered = [block(b) for b in s.blocks]
    return TEMPLATE.render(title=title, meta=meta, sections=sections)
