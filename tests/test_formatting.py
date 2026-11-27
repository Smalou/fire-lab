from components.formatting import NBSP, fmt_money, fmt_num, fmt_pct, years_pl


def test_polish_number_format():
    assert fmt_num(1234567.891, 2) == f"1{NBSP}234{NBSP}567,89"
    assert fmt_money(2_250_000) == f"2{NBSP}250{NBSP}000{NBSP}zł"
    assert fmt_money(2_250_000, compact=True) == f"2,25{NBSP}mln{NBSP}zł"
    assert fmt_pct(0.0525, 2) == "5,25%"
    assert fmt_pct(float("nan")) == "—"


def test_polish_plural_years():
    assert years_pl(1) == f"1{NBSP}rok"
    assert years_pl(3) == f"3{NBSP}lata"
    assert years_pl(5) == f"5{NBSP}lat"
    assert years_pl(22) == f"22{NBSP}lata"
    assert years_pl(12) == f"12{NBSP}lat"
