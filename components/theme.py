"""Design tokens shared by Streamlit theme (config.toml) and Plotly charts."""

from __future__ import annotations

import plotly.graph_objects as go

NAVY = "#0B2545"
TEAL = "#0E9F9A"
TEAL_LIGHT = "#7FD1CC"
ORANGE = "#F28C28"  # FIRE goal accents only
RED = "#D64545"  # losses, errors and risk only
GREY = "#8A94A6"
GRID = "rgba(138,148,166,0.20)"

ASSET_CLASS_COLORS = {
    "global_equity": "#0E9F9A",
    "us_equity": "#1D4E89",
    "europe_equity": "#5B8DD6",
    "em_equity": "#7A5EA8",
    "bonds": "#6FA37A",
    "gold": "#C9A227",
    "cash": "#9AA5B1",
}

SERIES = ["#0E9F9A", "#1D4E89", "#5B8DD6", "#7A5EA8", "#6FA37A", "#C9A227", "#9AA5B1", "#3FB8AF", "#2E3A59"]

ICONS = {
    "home": ":material/space_dashboard:",
    "planner": ":material/flag:",
    "portfolio": ":material/donut_large:",
    "strategy": ":material/science:",
    "risk": ":material/ssid_chart:",
    "post_fire": ":material/beach_access:",
    "scenarios": ":material/compare_arrows:",
    "methodology": ":material/menu_book:",
    "save": ":material/save:",
    "run": ":material/play_arrow:",
    "download": ":material/download:",
    "upload": ":material/upload_file:",
    "settings": ":material/tune:",
    "refresh": ":material/refresh:",
    "warning": ":material/warning:",
    "info": ":material/info:",
    "apply": ":material/done_all:",
    "add": ":material/add:",
    "copy": ":material/content_copy:",
    "edit": ":material/edit:",
    "delete": ":material/delete:",
    "data": ":material/database:",
    "celebrate": ":material/celebration:",
}


def style(fig: go.Figure, height: int = 380, legend_bottom: bool = True, **layout) -> go.Figure:
    """Apply consistent Polish-locale styling. Colours of traces are set explicitly by callers."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36 if layout.get("title") else 12, b=8),
        separators=", ",  # decimal comma, space as thousands separator
        hovermode=layout.pop("hovermode", "x unified"),
        font=dict(family="Inter, Source Sans Pro, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="top", y=-0.15, x=0) if legend_bottom else None,
        **layout,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def hex_to_rgba(color: str, alpha: float) -> str:
    c = color.lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
