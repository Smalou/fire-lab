"""Summary report (Markdown and standalone HTML) with assumptions, data source and parameters."""

from __future__ import annotations

import html
from datetime import datetime

from components.formatting import fmt_age, fmt_datetime, fmt_money, fmt_num, fmt_pct
from core.backtest import STRATEGY_LABELS
from core.fire import MODE_LABELS
from core.models import Scenario
from core.withdrawal import STRATEGY_LABELS as W_LABELS
from services import catalog


def build_markdown(s: Scenario, ctx, proj, mc: dict | None, mc_fresh: bool, bt: dict | None, bt_fresh: bool,
                   dec: dict | None, dec_fresh: bool) -> str:
    p, f, m, sim = s.profile, s.fire, s.market, s.simulation
    now = datetime.now()
    lines = [
        f"# FIRE Lab – raport scenariusza „{s.name}”",
        "",
        f"*Wygenerowano: {fmt_datetime(now)}*  ",
        f"*Źródło danych: {ctx.md.source_label if ctx.md else '—'} (aktualizacja: {fmt_datetime(ctx.md.retrieved_at) if ctx.md else '—'})*",
        "",
    ]
    if ctx.md is not None and ctx.md.is_synthetic:
        lines += ["> **Uwaga:** wyniki obliczono na danych syntetycznych (tryb demonstracyjny). Nie opisują one rzeczywistych rynków.", ""]
    lines += [
        "## Założenia",
        "",
        "| Parametr | Wartość |",
        "|---|---|",
        f"| Wiek obecny / FIRE / horyzont | {p.current_age} / {p.fire_age} / {p.life_horizon_age} |",
        f"| Kapitał początkowy | {fmt_money(p.initial_capital)} |",
        f"| Wpłata miesięczna (realnie) | {fmt_money(p.monthly_contribution)} |",
        f"| Wydatki po FIRE (mies., realnie) | {fmt_money(p.retirement_monthly_expenses)} |",
        f"| Inflacja | {fmt_pct(p.inflation, 2)} |",
        f"| Wariant FIRE | {MODE_LABELS[f.mode]} |",
        f"| SWR | {fmt_pct(f.swr, 2)} |",
        f"| Koszty portfela | {fmt_pct(s.portfolio.management_fee, 2)} rocznie |",
        f"| Stopa wolna od ryzyka | {fmt_pct(m.risk_free_rate, 2)} |",
        f"| Okres analizy / estymacji | {ctx.check.start:%m.%Y} – {ctx.check.end:%m.%Y} / ostatnie {m.estimation_years} lat |" if ctx.check and ctx.check.start is not None else "| Okres analizy | — |",
        f"| Strategia wypłat | {W_LABELS[s.withdrawal.strategy]} |",
        f"| Podatek od wypłat (uproszczony) | {fmt_pct(s.withdrawal.withdrawal_tax_rate, 0)} |",
        "",
        "## Portfel",
        "",
        "| Instrument | Waga |",
        "|---|---|",
    ]
    lines += [f"| {catalog.display_name(a)} | {fmt_pct(w, 1)} |" for a, w in zip(s.portfolio.assets, s.portfolio.weights)]
    if ctx.ok:
        st_ = ctx.stats
        lines += ["", f"Oczekiwana stopa (arytm.): {fmt_pct(st_['expected_arithmetic'], 2)}, geometryczna: "
                      f"{fmt_pct(st_['geometric_nominal'], 2)}, zmienność: {fmt_pct(st_['volatility'], 2)}, "
                      f"realna netto: {fmt_pct(st_['geometric_real_net'], 2)}."]
    if proj is not None:
        lines += ["", "## Plan deterministyczny", "",
                  f"- Wymagany kapitał FIRE: **{fmt_money(proj.target.capital)}**",
                  f"- Prognozowany wiek FIRE: **{fmt_age(proj.fire_age) if proj.fire_age else 'poza horyzontem pracy'}**"]
    lines += ["", "## Monte Carlo", ""]
    if mc is None:
        lines.append("Nie obliczono.")
    else:
        sm = mc["summary"]
        lines += [
            "" if mc_fresh else "> Wyniki nieaktualne względem bieżących parametrów.",
            f"- Model: {'block bootstrap' if mc['model'] == 'bootstrap' else 'parametryczny'}, ścieżek: {fmt_num(mc['n_paths'])}, ziarno: {mc['seed']}"
            f", grube ogony: {'tak' if sim.fat_tails else 'nie'}",
            f"- P(FIRE w docelowym wieku): {fmt_pct(sm['p_fire_by_target_age'], 1)}",
            f"- P(utrzymania kapitału): {fmt_pct(sm['p_not_depleted'], 1)}",
            f"- P(pokrycia wydatków): {fmt_pct(sm['p_spending_covered'], 1)}",
            f"- P(sukcesu planu): {fmt_pct(sm['p_success'], 1)}",
            f"- VaR {fmt_pct(mc['var'].confidence, 1)} ({mc['var'].horizon_years} r.): {fmt_pct(mc['var'].var, 1)}; "
            f"ES: {fmt_pct(mc['var'].es, 1)}",
            f"- Mediana majątku końcowego (realnie): {fmt_money(sm['median_terminal_wealth'])}",
        ]
    lines += ["", "## Backtest", ""]
    if bt is None:
        lines.append("Nie obliczono.")
    else:
        if not bt_fresh:
            lines.append("> Wyniki nieaktualne względem bieżących parametrów.")
        lines += ["| Strategia | CAGR | Zmienność | Sharpe | Max DD | XIRR |", "|---|---|---|---|---|---|"]
        for k, r in bt["results"].items():
            mm = r.metrics
            lines.append(f"| {STRATEGY_LABELS[k]} | {fmt_pct(mm['cagr'], 2)} | {fmt_pct(mm['volatility'], 2)} | "
                         f"{fmt_num(mm['sharpe'], 2)} | {fmt_pct(mm['max_drawdown'], 1)} | {fmt_pct(mm['xirr'], 2)} |")
    lines += ["", "## Dekumulacja", ""]
    if dec is None:
        lines.append("Nie obliczono.")
    else:
        if not dec_fresh:
            lines.append("> Wyniki nieaktualne względem bieżących parametrów.")
        lines += [f"Kapitał startowy: {fmt_money(dec['start_capital'])}", "",
                  "| Strategia | P(niewyczerpania) | P(pokrycia wydatków) | Mediana majątku końcowego |", "|---|---|---|---|"]
        for label, v in dec["variants"].items():
            sm = v["summary"]
            lines.append(f"| {label} | {fmt_pct(sm['p_not_depleted'], 1)} | {fmt_pct(sm['p_spending_covered'], 1)} | "
                         f"{fmt_money(sm['median_terminal_wealth'])} |")
    lines += ["", "## Ograniczenia", "",
              "- Historyczne stopy zwrotu nie gwarantują przyszłych wyników; parametry obarczone są błędem estymacji.",
              "- Model bez pełnego odwzorowania polskiego systemu podatkowego (jawny, uproszczony podatek od wypłat).",
              "- Stała inflacja i stopa wolna od ryzyka; brak modelowania długowieczności poza przyjętym horyzontem.",
              "- Katalog instrumentów obejmuje istniejące dziś fundusze (survivorship bias)."]
    return "\n".join(lines)


def markdown_to_html(md: str, title: str) -> str:
    """Minimal, dependency-free Markdown → HTML for the report structure used above."""
    out, in_table, in_list = [], False, False
    for raw in md.splitlines():
        line = raw.rstrip()
        esc = html.escape(line)
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= {"-"} for c in cells):
                continue
            if not in_table:
                out.append("<table>")
                in_table = True
                out.append("<tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in cells) + "</tr>")
            else:
                out.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>")
            in_table = False
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline(html.escape(line[2:]))}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if line.startswith("# "):
            out.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("> "):
            out.append(f"<blockquote>{_inline(html.escape(line[2:]))}</blockquote>")
        elif line:
            out.append(f"<p>{_inline(esc)}</p>")
    if in_table:
        out.append("</table>")
    if in_list:
        out.append("</ul>")
    css = ("body{font-family:Inter,Segoe UI,sans-serif;max-width:900px;margin:2rem auto;color:#0B2545;padding:0 1rem}"
           "h1{color:#0B2545}h2{color:#0E9F9A;border-bottom:1px solid #dde3ea;padding-bottom:4px}"
           "table{border-collapse:collapse;width:100%;margin:1rem 0}td,th{border:1px solid #dde3ea;padding:6px 8px;text-align:left}"
           "th{background:#F3F6F9}blockquote{border-left:4px solid #F28C28;margin:1rem 0;padding:.5rem 1rem;background:#FFF6EC}")
    return f"<!doctype html><html lang='pl'><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{css}</style></head><body>{''.join(out)}</body></html>"


def _inline(text: str) -> str:
    import re

    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
