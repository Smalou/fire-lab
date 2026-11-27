"""Reusable page building blocks."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from components.formatting import fmt_date, fmt_datetime
from components.theme import ICONS
from services import state
from services.analytics import Context


def page_header(title: str, subtitle: str, icon: str | None = None) -> None:
    st.title(title, anchor=False)
    st.caption(subtitle)


def data_badge(ctx: Context) -> None:
    """Data source, retrieval date and synthetic-data warning (always visible)."""
    if ctx.md is None:
        return
    md = ctx.md
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        if md.is_synthetic:
            st.badge("DANE SYNTETYCZNE", icon=":material/science:", color="orange",
                     help="Tryb demonstracyjny: dane wygenerowane losowo ze stałym ziarnem, nie są notowaniami rynkowymi.")
        else:
            st.badge(md.source_label.split(" – ")[0], icon=ICONS["data"], color="blue")
        text = f"Źródło: {md.source_label} · aktualizacja: {fmt_datetime(md.retrieved_at)}"
        if ctx.check and ctx.check.start is not None:
            text += f" · okres analizy: {fmt_date(ctx.check.start)} – {fmt_date(ctx.check.end)} ({ctx.check.n_months} mies.)"
        st.caption(text)


def stale_notice(computed_at: datetime | None, fresh: bool) -> None:
    if computed_at is None:
        return
    if fresh:
        st.caption(f":material/check_circle: Wyniki aktualne · obliczono {fmt_datetime(computed_at)}")
    else:
        st.warning(
            f"Wyniki z {fmt_datetime(computed_at)} są **nieaktualne** – od tego czasu zmieniono parametry scenariusza. "
            "Uruchom obliczenia ponownie.",
            icon=ICONS["warning"],
        )


def context_guard(ctx: Context) -> bool:
    """Show data problems with actionable fixes. Returns True if analyses can proceed."""
    if ctx.ok:
        for w in (ctx.md.warnings if ctx.md else []):
            st.info(w, icon=ICONS["info"])
        if ctx.moments and ctx.moments.psd_repaired:
            st.warning("Macierz kowariancji nie była dodatnio półokreślona – zastosowano rzutowanie na najbliższą macierz PSD.",
                       icon=ICONS["warning"])
        return True
    st.error(ctx.error or "Brak danych.", icon=":material/error:")
    s = ctx.scenario
    chk = ctx.check
    with st.container(horizontal=True):
        if chk and chk.suggested_start is not None and s.market.start_date is not None:
            if st.button(f"Skróć okres – start od {fmt_date(chk.suggested_start)}", icon=":material/content_cut:"):
                new = s.model_copy(deep=True)
                new.market.start_date = chk.suggested_start.date()
                state.save(new, "Skrócono okres analizy do wspólnej historii aktywów.")
                st.rerun()
        if chk and chk.substitutes:
            for orig, sub in chk.substitutes.items():
                if sub in s.portfolio.assets:
                    continue
                if st.button(f"Zastąp {orig} → {sub} (zamiennik)", icon=":material/swap_horiz:"):
                    new = s.model_copy(deep=True)
                    new.portfolio.assets = [sub if a == orig else a for a in new.portfolio.assets]
                    new.description = (new.description + f" [Zamiennik: {sub} zamiast {orig}]").strip()
                    state.save(new, f"Zastąpiono {orig} jawnie oznaczonym zamiennikiem {sub}.")
                    st.rerun()
        if s.market.data_source != "demo" and st.button("Przełącz na tryb demonstracyjny", icon=":material/science:"):
            from services.scenarios import remap_portfolio_to_source

            new = s.model_copy(deep=True)
            new.portfolio = remap_portfolio_to_source(new, "demo")
            new.market.data_source = "demo"
            new.market.benchmark = "SYN_GLOBAL"
            new.market.start_date = None
            state.save(new, "Przełączono na dane demonstracyjne (syntetyczne).")
            st.rerun()
    st.page_link("views/portfolio_builder.py", label="Zmień skład portfela", icon=ICONS["portfolio"])
    return False


def not_computed(message: str, page: str, label: str, icon: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{message}**")
        st.page_link(page, label=label, icon=icon)
