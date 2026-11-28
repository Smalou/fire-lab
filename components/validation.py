"""Polish rendering of Pydantic validation errors."""

from __future__ import annotations

from pydantic import ValidationError

FIELD_LABELS = {
    "current_age": "Obecny wiek",
    "fire_age": "Docelowy wiek FIRE",
    "life_horizon_age": "Horyzont życia",
    "initial_capital": "Kapitał początkowy",
    "monthly_contribution": "Miesięczna wpłata",
    "swr": "Stopa wypłat (SWR)",
    "weights": "Wagi portfela",
    "amount": "Kwota",
    "age": "Wiek",
    "end_age": "Wiek końcowy",
    "name": "Nazwa",
    "max_work_age": "Maksymalny wiek pracy",
    "n_paths": "Liczba ścieżek",
}

TYPE_MESSAGES = {
    "greater_than_equal": "wartość musi być ≥ {ge}",
    "less_than_equal": "wartość musi być ≤ {le}",
    "greater_than": "wartość musi być > {gt}",
    "less_than": "wartość musi być < {lt}",
    "string_too_short": "pole nie może być puste",
    "string_too_long": "tekst jest za długi",
    "missing": "pole jest wymagane",
    "int_parsing": "wymagana liczba całkowita",
    "float_parsing": "wymagana liczba",
}


def polish_errors(exc: ValidationError) -> list[str]:
    out = []
    for e in exc.errors():
        loc = [str(p) for p in e.get("loc", ()) if not str(p).isdigit()]
        field = FIELD_LABELS.get(loc[-1], loc[-1]) if loc else ""
        typ = e.get("type", "")
        if typ == "value_error":
            msg = str(e.get("msg", "")).removeprefix("Value error, ")
        elif typ in TYPE_MESSAGES:
            msg = TYPE_MESSAGES[typ].format(**(e.get("ctx") or {}))
        else:
            msg = e.get("msg", "niepoprawna wartość")
        out.append(f"{field}: {msg}" if field else msg)
    return out
