# Dane demonstracyjne – DANE SYNTETYCZNE

Pliki w tym katalogu **nie są notowaniami rynkowymi**. Zostały wygenerowane skryptem
`scripts/generate_demo_data.py` (wielowymiarowy rozkład t-Studenta, stałe ziarno losowe)
wyłącznie po to, aby aplikacja działała bez dostępu do Internetu.

* `synthetic_monthly_prices.csv` – miesięczne indeksy cen całkowitych (z reinwestycją dochodu) w PLN, start = 100.
* `synthetic_monthly_income.csv` – syntetyczny dochód (dywidendy/kupony) na jednostkę – używany tylko przez strategię „tylko dochód”.

Parametry (roczne, arytmetyczne): zob. stałe `MU`, `VOL`, `CORR`, `YIELD` w skrypcie.
Wyniki uzyskane na tych danych ilustrują działanie metod, ale nie opisują rzeczywistych rynków.
