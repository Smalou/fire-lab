# FIRE Lab – plan realizacji

> Od akumulacji do niezależności finansowej.
> Pytanie badawcze: czy strategia inwestycyjna optymalna w fazie akumulacji kapitału pozostaje optymalna po osiągnięciu niezależności finansowej?

## 1. Stan wyjściowy repozytorium

Repozytorium `Smalou/fire-lab` zawierało wyłącznie jednolinijkowy `README.md`. Brak kodu do zachowania – aplikacja budowana od podstaw.

Ograniczenie środowiska deweloperskiego: środowisko chmurowe, w którym powstał kod, blokuje dostęp do Yahoo Finance. Dlatego:

* tryb demonstracyjny korzysta z **jawnie oznaczonych danych syntetycznych** (`data/demo/synthetic_monthly_prices.csv`, generowanych skryptem `scripts/generate_demo_data.py` ze stałym ziarnem),
* tryb yfinance działa na komputerze użytkownika z dostępem do Internetu,
* skrypt `scripts/fetch_snapshot.py` zapisuje migawkę rzeczywistych notowań do `data/snapshot/`, którą aplikacja może następnie wczytać offline (oznaczoną jako „Yahoo Finance – migawka z dnia …”).

## 2. Architektura

```
app.py                  entrypoint: konfiguracja strony, st.navigation, pasek boczny
core/                   czysta warstwa obliczeniowa (bez importu streamlit)
  models.py             Pydantic: Scenario i jego sekcje (jeden centralny model)
  returns.py            konwersje stóp, annualizacja, realne/nominalne
  fire.py               kapitał FIRE (5 trybów), ścieżka deterministyczna
  portfolio.py          estymacja μ/Σ, PSD, optymalizacja Markowitza, RP, CAPM
  metrics.py            CAGR, TWR, XIRR, Sharpe, Sortino, MDD, beta, alfa, TE, IR
  backtest.py           walk-forward backtest z wpłatami i kosztami
  simulation.py         generatory ścieżek: MVN / t-Student, block bootstrap
  lifecycle.py          wektorowy silnik cyklu życia (akumulacja + dekumulacja)
  withdrawal.py         strategie wypłat (stała realna / % portfela / guardrails / dochód)
  risk.py               VaR, ES, percentyle, scenariusze kryzysowe, wrażliwość
  sequence.py           eksperyment Sequence of Returns Risk
services/
  catalog.py            katalog instrumentów (data/instruments.json)
  market_data.py        yfinance, FX, CSV, demo, migawka; walidacja i braki danych
  scenarios.py          CRUD scenariuszy, JSON, scenariusze domyślne
  state.py              session_state: szkic / zatwierdzona konfiguracja / wyniki / UI
  analytics.py          cache'owane (st.cache_data) wywołania warstwy core
  reports.py            raport HTML/Markdown
components/             theme, formatowanie PL, karty KPI, wykresy, nagłówki
views/                  8 stron
data/                   katalog instrumentów, dane demo
tests/                  pytest + Streamlit AppTest
```

Zasada: widoki tylko zbierają parametry i rysują wyniki. Każda liczba pochodzi z `core/`.

## 3. Zarządzanie stanem

| Warstwa | Gdzie | Opis |
|---|---|---|
| Parametry edytowane | widżety w `st.form` | szkic, nie wpływa na wyniki do zatwierdzenia |
| Zatwierdzona konfiguracja | `st.session_state["scenarios"][id]` (Pydantic `Scenario`) | jedno źródło prawdy dla wszystkich stron |
| Wyniki | `st.session_state["results"][kind]` z odciskiem (`fingerprint`) części scenariusza, od której zależą | wynik z innym odciskiem = **nieaktualny** |
| Stan UI | klucze widżetów (`ui_*`), `st.query_params` | tryby widoków, zaznaczenia – nie unieważniają wyników |

Kosztowne obliczenia (Monte Carlo, backtest, granica efektywna) uruchamiane przyciskiem lub po zatwierdzeniu formularza i cache'owane `st.cache_data`. Sekcje interaktywne w `st.fragment`.

## 4. Decyzje metodologiczne

1. **Krok czasowy: miesiąc.** Ceny dzienne → ostatnia cena miesiąca → proste miesięczne stopy zwrotu.
2. **Dane skorygowane.** Stopy całkowite z `Adj Close` (dywidendy reinwestowane). Dywidendy surowe służą wyłącznie do strategii „tylko dochód” jako część stopy całkowitej wypłacana z portfela – nie są dodawane drugi raz.
3. **Waluta bazowa (domyślnie PLN).** Cena w PLN = cena lokalna × kurs FX z tego samego dnia; stopa w PLN zawiera ryzyko walutowe.
4. **Realne vs nominalne.** Użytkownik podaje kwoty w dzisiejszych złotych (realnych), a wzrosty dochodów i wpłat jako realne (ponad inflację). Silnik liczy nominalnie: przepływy indeksowane inflacją, stopy rynkowe nominalne; wyniki prezentowane realnie przez deflator \(I_t=(1+\pi)^{t}\). Inflacja naliczana dokładnie raz.
5. **Konwencja przepływów:** wpłaty, wypłaty i rebalancing na **początku miesiąca**, następnie stopa zwrotu miesiąca, na końcu opłata za zarządzanie. Ta sama konwencja w ścieżce deterministycznej, backteście i Monte Carlo.
6. **Annualizacja:** średnia arytmetyczna ×12, zmienność ×√12, CAGR geometryczny. Do projekcji deterministycznej stosowana stopa geometryczna (mediana), nie arytmetyczna.
7. **Kowariancja:** próbkowa lub Ledoit–Wolf (scikit-learn). Kontrola PSD przez wartości własne; w razie potrzeby rzut na najbliższą macierz PSD z ostrzeżeniem.
8. **Optymalizacja:** SLSQP (SciPy) z ograniczeniami wag i sumy; status zbieżności zawsze raportowany, błędy nie są ukrywane.
9. **Backtest walk-forward:** wagi na miesiąc t+1 wyznaczane wyłącznie z danych do końca miesiąca t. Okres rozgrzewki (okno estymacji) wspólny dla wszystkich strategii. Test jednostkowy sprawdza, że zmiana przyszłych danych nie zmienia wcześniejszych decyzji.
10. **Wyniki vs wpłaty:** TWR/CAGR/MDD liczone na indeksie stopy zwrotu (bez wpłat); XIRR i zysk inwestycyjny = wartość − suma wpłat.
11. **Monte Carlo:** (a) wielowymiarowy rozkład normalny lub t-Studenta stóp logarytmicznych aktywów z macierzą kowariancji, (b) kołowy block bootstrap wektorów historycznych stóp (zachowuje korelacje i autokorelację w blokach). Ziarno `numpy.random.default_rng(seed)`, generacja w paczkach – rzeczywisty postęp w `st.progress`.
12. **VaR/ES:** definiowane na rozkładzie stopy zwrotu portfela (bez przepływów) w horyzoncie H lat (domyślnie 1) i poziomie ufności α (domyślnie 95%); raportowane jako % i PLN dla obecnego kapitału.
13. **Sukces po FIRE ≠ dodatni kapitał.** Raportowane osobno: P(niewyczerpania), P(pokrycia wydatków ≥ progu w każdym roku), zmienność konsumpcji, największy wymagany spadek wydatków.
14. **Reguła 4%** = stała realna wypłata ze stopą początkową 4% (preset, nie osobny mechanizm).
15. **Podatki:** model bez podatku; opcjonalny, jawny, uproszczony efektywny podatek od wypłat (gross-up). Nie jest to model polskiego systemu podatkowego.
16. **Survivorship bias:** katalog zawiera wyłącznie istniejące dziś ETF-y – komunikat w aplikacji i metodologii.

### Tryby FIRE (odmienna logika)

| Tryb | Kapitał docelowy | Logika ścieżki |
|---|---|---|
| Traditional | E / SWR | wpłaty do FIRE, potem pełne wypłaty |
| Lean | E_podst / SWR | po FIRE finansowane tylko wydatki podstawowe |
| Fat | E·(1+m) / SWR + bufor gotówkowy k lat | wyższe wydatki + rezerwa płynnościowa |
| Coast | C_FIRE / (1+g)^(T−t) | wpłaty tylko do osiągnięcia kwoty Coast, potem portfel rośnie bez wpłat |
| Barista | (E − I_częśc.) / SWR | po FIRE dochód z pracy w niepełnym wymiarze do wieku X |

## 5. Etapy

1. Fundament aplikacji i design system.
2. Warstwa danych.
3. Silnik obliczeń finansowych.
4. Planer FIRE.
5. Konstruktor portfela.
6. Backtesting.
7. Monte Carlo.
8. Dekumulacja.
9. Porównywarka scenariuszy.
10. Integracja, testy, UX, trzy audyty.

## 6. Kryteria akceptacji

* `streamlit run app.py` uruchamia aplikację; wszystkie 8 stron działa (AppTest).
* Polski interfejs, polski format liczb i dat, PLN domyślnie.
* Tryb demo działa bez Internetu; dane syntetyczne oznaczone na każdej stronie.
* Wszystkie wyniki z rzeczywistych obliczeń; zmiana portfela unieważnia zależne wyniki.
* Brak nieobsłużonych wyjątków w typowych interakcjach, brak martwych przycisków.
* `pytest` przechodzi; README pozwala uruchomić projekt.
