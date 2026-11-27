import numpy as np
import pandas as pd
import pytest

from services import catalog
from services.market_data import DataUnavailableError, _flatten_download, check_coverage, load_demo, parse_price_csv, prepare_monthly


def test_catalog_covers_required_asset_classes():
    classes = {i.asset_class for i in catalog.instruments("yfinance")}
    assert {"global_equity", "us_equity", "europe_equity", "em_equity", "bonds", "gold", "cash"} <= classes
    for i in catalog.instruments():
        assert i.currency in {"USD", "EUR", "PLN"} and i.listed_since
        if i.substitute:
            assert catalog.get(i.substitute) is not None


def test_demo_data_is_flagged_synthetic():
    md = load_demo(["SYN_GLOBAL", "SYN_BONDS"])
    assert md.is_synthetic and "syntetyczne" in md.source_label.lower()
    assert md.returns.shape[1] == 2 and len(md.returns) > 200
    y = md.income_yield(["SYN_GLOBAL", "SYN_BONDS"])
    assert 0 < y["SYN_BONDS"] < 0.1


def _fake_daily():
    idx = pd.bdate_range("2019-01-01", "2021-12-31")
    n = len(idx)
    usd = pd.Series(np.linspace(100, 150, n), index=idx)
    eur = pd.Series(np.linspace(50, 60, n), index=idx)
    adj = pd.DataFrame({"SPY": usd, "EUNL.DE": eur})
    close = adj * 1.0
    div = pd.DataFrame(0.0, index=idx, columns=adj.columns)
    div.loc[idx[idx.month == 6][0], "SPY"] = 2.0
    fx = pd.DataFrame({"USDPLN=X": np.full(n, 4.0), "EURPLN=X": np.full(n, 4.5)}, index=idx)
    return adj, close, div, fx


def test_fx_conversion_to_base_currency():
    adj, close, div, fx = _fake_daily()
    monthly, income, warns = prepare_monthly(adj, close, div, fx, {"SPY": "USD", "EUNL.DE": "EUR"}, "PLN")
    assert monthly["SPY"].iloc[-1] == pytest.approx(150 * 4.0)
    assert monthly["EUNL.DE"].iloc[-1] == pytest.approx(60 * 4.5)
    assert not warns
    # dividends converted with FX, used only for income (not added to returns)
    assert income["SPY"].sum() == pytest.approx(2.0 * 4.0)
    r = monthly.pct_change().dropna()
    assert r["SPY"].iloc[0] == pytest.approx(monthly["SPY"].iloc[1] / monthly["SPY"].iloc[0] - 1)


def test_fx_changes_returns():
    adj, close, div, fx = _fake_daily()
    adj["SPY"] = 100.0
    fx["USDPLN=X"] = np.linspace(4.0, 3.0, len(fx))  # PLN strengthens
    monthly, _, _ = prepare_monthly(adj, close, div, fx, {"SPY": "USD", "EUNL.DE": "EUR"}, "PLN")
    assert monthly["SPY"].iloc[-1] < monthly["SPY"].iloc[0]


def test_missing_fx_is_reported():
    adj, close, div, fx = _fake_daily()
    monthly, _, warns = prepare_monthly(adj, close, div, fx.drop(columns=["EURPLN=X"]), {"SPY": "USD", "EUNL.DE": "EUR"}, "PLN")
    assert "EUNL.DE" not in monthly and any("EUR/PLN" in w for w in warns)


def test_flatten_yfinance_multiindex():
    idx = pd.date_range("2020-01-01", periods=3, tz="America/New_York")
    cols = pd.MultiIndex.from_product([["Adj Close", "Close", "Dividends"], ["A", "B"]])
    df = pd.DataFrame(np.arange(18).reshape(3, 6), index=idx, columns=cols)
    out = _flatten_download(df, "Adj Close", ["A", "B"])
    assert list(out.columns) == ["A", "B"] and out.index.tz is None


def test_csv_parsing_semicolon_decimal_comma():
    dates = pd.date_range("2015-01-31", periods=48, freq="ME")
    rows = [f"{d:%Y-%m-%d};{100 + i * 0.5:.2f};{50 + i * 0.1:.2f}" for i, d in enumerate(dates)]
    rows = [r.split(";", 1)[0] + ";" + r.split(";", 1)[1].replace(".", ",") for r in rows]  # decimal comma
    txt = "Data;Akcje;Obligacje\n" + "\n".join(rows)
    prices = parse_price_csv(txt.encode())
    assert list(prices.columns) == ["Akcje", "Obligacje"]
    assert prices["Akcje"].iloc[1] == pytest.approx(100.5)


def test_csv_rejects_bad_files():
    with pytest.raises(DataUnavailableError):
        parse_price_csv(b"a,b\nx,1\ny,2\n")
    short = "date,A\n" + "\n".join(f"2020-{m:02d}-28,{100 + m}" for m in range(1, 13))
    with pytest.raises(DataUnavailableError):
        parse_price_csv(short.encode())
    neg = "date,A\n" + "\n".join(f"{d:%Y-%m-%d},{-1}" for d in pd.date_range("2010-01-31", periods=50, freq="ME"))
    with pytest.raises(DataUnavailableError):
        parse_price_csv(neg.encode())


def test_coverage_check_suggests_shorter_period_and_substitute():
    md = load_demo(["SYN_GLOBAL", "SYN_BONDS"])
    md.prices.loc[: "2010-12-31", "SYN_BONDS"] = np.nan  # pretend bonds start later
    chk = check_coverage(md, ["SYN_GLOBAL", "SYN_BONDS"], start="2005-01-01")
    assert chk.suggested_start >= pd.Timestamp("2011-01-01")
    assert any("dopiero od" in m for m in chk.messages)
    tight = check_coverage(md, ["SYN_GLOBAL", "SYN_BONDS"], start="2024-06-01")
    assert not tight.ok
