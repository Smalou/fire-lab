"""Plotly figure factories (presentation only – data comes from core)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from components.theme import ASSET_CLASS_COLORS, GREY, NAVY, ORANGE, RED, SERIES, TEAL, hex_to_rgba, style
from services import catalog

PLN_AXIS = dict(tickformat=",.0f", ticksuffix=" zł")


def allocation_donut(assets: list[str], weights, classes: dict[str, str], height: int = 300) -> go.Figure:
    labels = [catalog.short_name(a) for a in assets]
    colors = [ASSET_CLASS_COLORS.get(classes.get(a, ""), GREY) for a in assets]
    fig = go.Figure(go.Pie(labels=labels, values=np.asarray(weights) * 100, hole=0.62, sort=False,
                           marker=dict(colors=colors, line=dict(width=1, color="rgba(255,255,255,0.6)")),
                           texttemplate="%{percent:.0%}", hovertemplate="%{label}<br>%{value:.1f}%<extra></extra>"))
    return style(fig, height, legend_bottom=False, showlegend=True, hovermode="closest")


def class_allocation(assets: list[str], weights, classes: dict[str, str]) -> pd.Series:
    labels = catalog.asset_class_labels()
    s = pd.Series(np.asarray(weights, dtype=float), index=[labels.get(classes.get(a, ""), "Inne") for a in assets])
    return s.groupby(level=0).sum().sort_values(ascending=False)


def accumulation_chart(table: pd.DataFrame, target: float, fire_age: float | None, nominal: bool = False,
                       fire_target_age: int | None = None, height: int = 380) -> go.Figure:
    col = "wealth_nominal" if nominal else "wealth_real"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=table["age"], y=table[col], name="Wartość portfela" + (" (nominalnie)" if nominal else " (realnie)"),
                             line=dict(color=TEAL, width=3), fill="tozeroy", fillcolor=hex_to_rgba(TEAL, 0.10),
                             hovertemplate="%{y:,.0f} zł"))
    if not nominal:
        fig.add_hline(y=target, line=dict(color=ORANGE, dash="dash", width=2), annotation_text="Cel FIRE",
                      annotation_position="top left", annotation_font_color=ORANGE)
    if fire_age is not None:
        fig.add_vline(x=fire_age, line=dict(color=ORANGE, width=1, dash="dot"))
    if fire_target_age is not None:
        fig.add_vrect(x0=table["age"].min(), x1=fire_target_age, fillcolor=hex_to_rgba(NAVY, 0.04), line_width=0,
                      annotation_text="Akumulacja", annotation_position="top left")
    fig.update_xaxes(title="Wiek")
    fig.update_yaxes(title="PLN (ceny stałe)" if not nominal else "PLN (nominalnie)", **PLN_AXIS)
    return style(fig, height)


def fan_chart(ages, pct: np.ndarray, target: float | None = None, fire_age: int | None = None,
              title: str | None = None, height: int = 420, color: str = TEAL) -> go.Figure:
    """pct: array (5, len(ages)) of P5, P25, P50, P75, P95."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ages, y=pct[4], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ages, y=pct[0], fill="tonexty", fillcolor=hex_to_rgba(color, 0.12), line=dict(width=0),
                             name="P5–P95", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ages, y=pct[3], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ages, y=pct[1], fill="tonexty", fillcolor=hex_to_rgba(color, 0.25), line=dict(width=0),
                             name="P25–P75", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ages, y=pct[2], line=dict(color=color, width=3), name="Mediana (P50)",
                             hovertemplate="Mediana: %{y:,.0f} zł"))
    fig.add_trace(go.Scatter(x=ages, y=pct[0], line=dict(color=RED, width=1, dash="dot"), name="P5",
                             hovertemplate="P5: %{y:,.0f} zł"))
    fig.add_trace(go.Scatter(x=ages, y=pct[4], line=dict(color=GREY, width=1, dash="dot"), name="P95",
                             hovertemplate="P95: %{y:,.0f} zł"))
    if target:
        fig.add_hline(y=target, line=dict(color=ORANGE, dash="dash", width=2), annotation_text="Cel FIRE",
                      annotation_font_color=ORANGE)
    if fire_age:
        fig.add_vline(x=fire_age, line=dict(color=ORANGE, dash="dot", width=1))
    fig.update_xaxes(title="Wiek")
    fig.update_yaxes(title="Majątek realny (PLN, ceny stałe)", **PLN_AXIS)
    return style(fig, height, title=title)


def histogram(values: np.ndarray, title_x: str, color: str = TEAL, nbins: int = 50, money: bool = True,
              vlines: dict[str, float] | None = None, height: int = 320) -> go.Figure:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    fig = go.Figure(go.Histogram(x=v, nbinsx=nbins, marker_color=color, opacity=0.85,
                                 hovertemplate="%{x}<br>Liczba ścieżek: %{y}<extra></extra>", histnorm=""))
    for name, x in (vlines or {}).items():
        fig.add_vline(x=x, line=dict(color=ORANGE if "cel" in name.lower() or "fire" in name.lower() else NAVY, dash="dash"),
                      annotation_text=name)
    fig.update_xaxes(title=title_x, **(PLN_AXIS if money else {}))
    fig.update_yaxes(title="Liczba ścieżek")
    return style(fig, height, hovermode="closest", bargap=0.05)


def cdf_chart(values: np.ndarray, title_x: str, money: bool = True, color: str = TEAL, height: int = 320,
              marker: float | None = None) -> go.Figure:
    v = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
    p = np.arange(1, v.size + 1) / v.size
    fig = go.Figure(go.Scatter(x=v, y=p, line=dict(color=color, width=2.5), name="Dystrybuanta empiryczna",
                               hovertemplate="%{x:,.0f}<br>P(X ≤ x) = %{y:.1%}<extra></extra>"))
    if marker is not None:
        fig.add_vline(x=marker, line=dict(color=ORANGE, dash="dash"), annotation_text="Cel FIRE")
    fig.update_xaxes(title=title_x, **(PLN_AXIS if money else {}))
    fig.update_yaxes(title="Prawdopodobieństwo skumulowane", tickformat=".0%")
    return style(fig, height, hovermode="closest")


def heatmap(z: np.ndarray, x_labels, y_labels, title_x: str, title_y: str, fmt: str = ".0%", height: int = 380,
            colorscale=None, zmid=None) -> go.Figure:
    colorscale = colorscale or [[0, RED], [0.5, "#F5E6C8"], [1, TEAL]]
    text = np.vectorize(lambda v: "—" if not np.isfinite(v) else f"{v:{fmt}}".replace(".", ","))(z)
    fig = go.Figure(go.Heatmap(z=z, x=x_labels, y=y_labels, colorscale=colorscale, zmin=0 if fmt.endswith("%") else None,
                               zmax=1 if fmt.endswith("%") else None, zmid=zmid, text=text, texttemplate="%{text}",
                               hovertemplate=f"{title_x}: %{{x}}<br>{title_y}: %{{y}}<br>Wartość: %{{text}}<extra></extra>",
                               colorbar=dict(tickformat=fmt)))
    fig.update_xaxes(title=title_x, type="category")
    fig.update_yaxes(title=title_y, type="category")
    return style(fig, height, hovermode="closest")


def matrix_heatmap(m: np.ndarray, labels: list[str], fmt: str = ".2f", diverging: bool = True, height: int = 380) -> go.Figure:
    scale = [[0, RED], [0.5, "#FFFFFF"], [1, TEAL]] if diverging else [[0, "#FFFFFF"], [1, NAVY]]
    text = np.vectorize(lambda v: f"{v:{fmt}}".replace(".", ","))(m)
    fig = go.Figure(go.Heatmap(z=m, x=labels, y=labels, colorscale=scale, zmid=0 if diverging else None,
                               zmin=-1 if diverging else None, zmax=1 if diverging else None,
                               text=text, texttemplate="%{text}", hovertemplate="%{y} / %{x}: %{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return style(fig, height, hovermode="closest")


def lines(df: pd.DataFrame, y_title: str, money: bool = False, pct: bool = False, height: int = 360,
          colors: list[str] | None = None) -> go.Figure:
    fig = go.Figure()
    colors = colors or SERIES
    for i, c in enumerate(df.columns):
        fig.add_trace(go.Scatter(x=df.index, y=df[c], name=str(c), line=dict(color=colors[i % len(colors)], width=2),
                                 hovertemplate="%{y:,.2f}" if not pct else "%{y:.2%}"))
    fig.update_yaxes(title=y_title, **(PLN_AXIS if money else {}), **({"tickformat": ".0%"} if pct else {}))
    fig.update_xaxes(tickformat="%m.%Y")
    return style(fig, height)
