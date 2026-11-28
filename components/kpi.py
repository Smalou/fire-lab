"""KPI cards built on native ``st.metric`` (border, delta and sparkline)."""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st


def kpi(label: str, value: str, delta: str | None = None, help: str | None = None, chart: Sequence[float] | None = None,
        chart_type: str = "area", delta_color: str = "normal", icon: str | None = None, delta_description: str | None = None) -> None:
    st.metric(
        label,
        value,
        delta=delta,
        delta_color=delta_color,
        help=help,
        border=True,
        chart_data=list(chart) if chart is not None else None,
        chart_type=chart_type,
        icon=icon,
        delta_description=delta_description,
        delta_arrow="off" if delta_color == "off" else "auto",
    )


def kpi_row(items: list[dict], columns: int | None = None) -> None:
    cols = st.columns(columns or len(items))
    for i, item in enumerate(items):
        with cols[i % len(cols)]:
            kpi(**item)
