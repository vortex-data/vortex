# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Shared chart styling: a fixed categorical order, a one-hue sequential ramp, recessive axes."""

import plotly.graph_objects as go
import plotly.io as pio

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL = [
    [0.0, "#f4f8fd"],
    [0.15, "#cde2fb"],
    [0.35, "#86b6ef"],
    [0.55, "#3987e5"],
    [0.75, "#1c5cab"],
    [1.0, "#0d366b"],
]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"

pio.templates["vx"] = go.layout.Template(
    layout=go.Layout(
        font={"family": "Inter, system-ui, sans-serif", "size": 13, "color": INK},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        colorway=SERIES,
        margin={"l": 60, "r": 20, "t": 80, "b": 50},
        xaxis={
            "automargin": True,
            "gridcolor": GRID,
            "zerolinecolor": GRID,
            "linecolor": GRID,
            "tickfont": {"color": INK_2},
        },
        yaxis={
            "automargin": True,
            "gridcolor": GRID,
            "zerolinecolor": GRID,
            "linecolor": GRID,
            "tickfont": {"color": INK_2},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "right",
            "x": 1,
            "font": {"color": INK_2},
        },
        hoverlabel={"bgcolor": "white", "font": {"color": INK}},
        title={"font": {"size": 15, "color": INK}, "x": 0, "xanchor": "left", "y": 0.98, "yanchor": "top"},
    )
)
pio.templates.default = "vx"


def color(i: int) -> str:
    """The i-th categorical hue. Past eight, callers fold series into Other or facet."""
    return SERIES[i % len(SERIES)]
