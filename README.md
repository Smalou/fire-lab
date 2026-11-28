# FIRE Lab – Od akumulacji do niezależności finansowej

Akademicka aplikacja Streamlit na zajęcia **Zarządzanie portfelem** (studia magisterskie).

**Pytanie badawcze:** czy strategia inwestycyjna optymalna w fazie akumulacji kapitału pozostaje optymalna po osiągnięciu FIRE (fazie wypłat)?

Interfejs jest w całości po polsku (formaty liczb, dat i waluty, domyślnie PLN). Kod, nazwy zmiennych i testy są po angielsku.

## Uruchomienie

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Testy (jednostkowe i AppTest dla wszystkich stron):

```bash
pytest -q
```

Wymagany Python 3.11 lub nowszy. Aplikacja była testowana na Pythonie 3.13 i Streamlit 1.67.

## Baza danych i profile użytkowników

Scenariusze, ustawienia profilu (aktywny scenariusz, aktywny plik danych) i wczytane pliki CSV zapisują się automatycznie w bazie **SQLite** `data/fire_lab.db` (inna lokalizacja: zmienna `FIRE_LAB_DB`). Plik bazy nie trafia do repozytorium. Wyniki analiz nie są zapisywane, bo liczą się ponownie z zapisanych ustawień.

| Tabela | Zawartość |
|---|---|
| `users` | profil: dostawca (`google` / `local`), identyfikator (`sub` z Google albo nazwa profilu lokalnego), e-mail, nazwa, zdjęcie, daty |
| `user_settings` | ustawienia profilu jako para klucz → JSON |
| `scenarios` | scenariusz jako dokument JSON modelu `Scenario` + `schema_version`, nazwa, kolejność, daty |
| `datasets` | wczytane pliki CSV (oryginalne bajty + przypisanie klas aktywów), bez duplikatów (SHA-256) |

Nowe pola dodane do modelu scenariusza z wartościami domyślnymi nie wymagają migracji bazy: starsze dokumenty wczytają się z wartościami domyślnymi. Zmiana nazwy lub znaczenia pola wymaga podbicia `SCHEMA_VERSION` w `core/models.py` i dopisania funkcji w `services/persistence.py` (`UPGRADERS`). Scenariusz, którego nie da się wczytać, zostaje w bazie, a aplikacja pokazuje ostrzeżenie zamiast błędu.

**Tryby logowania:**

- **Bez konfiguracji (domyślnie, na własnym komputerze):** profile lokalne bez hasła, wybierane w panelu bocznym. Oddzielają dane, ale ich nie chronią.
- **Google:** skopiuj `.streamlit/secrets.example.toml` do `.streamlit/secrets.toml` i uzupełnij. W [Google Cloud Console](https://console.cloud.google.com/apis/credentials) utwórz *OAuth client ID* typu *Web application*, a jako *Authorized redirect URI* dodaj `http://localhost:8501/oauth2callback` (i adres wdrożonej aplikacji z `/oauth2callback`). Zalogowani użytkownicy mają własne dane, niezalogowani pracują jako goście (tylko sesja). Lista `allowed_emails` ogranicza, kto może zapisywać dane.

Strona **Profil i dane** pokazuje, co jest zapisane, pozwala pobrać wszystkie dane (JSON), usunąć zapisane pliki albo cały profil.

## Źródła danych

Źródło wybiera się w panelu bocznym (przycisk **Dane**). Każda strona pokazuje źródło, datę aktualizacji i okres analizy.

| Źródło | Opis |
|---|---|
| **Demo (dane syntetyczne)** | Działa offline. Ceny są wygenerowane losowo (`scripts/generate_demo_data.py`, stałe ziarno) i **nie są notowaniami rynkowymi**. Aplikacja zawsze oznacza je etykietą „DANE SYNTETYCZNE”. |
| **Yahoo Finance (na żywo)** | Notowania `yfinance` (ceny skorygowane i dywidendy), przeliczone na walutę bazową kursami FX z Yahoo i zapisane w pamięci podręcznej na 12 godzin. Przycisk **Odśwież** czyści tę pamięć. |
| **Migawka Yahoo Finance** | Rzeczywiste dane zapisane lokalnie przez `python scripts/fetch_snapshot.py`, które trzeba uruchomić na komputerze z dostępem do Internetu. Dzięki migawce wyniki są powtarzalne i nie wymagają sieci. |
| **Import CSV** | Własne ceny miesięczne lub dzienne: separator `;` albo `,`, przecinek dziesiętny, daty ISO lub DD.MM.RRRR. Klasę aktywów każdej kolumny przypisuje się w edytorze. |

Gdy historia danych jest za krótka, aplikacja to zgłasza, proponuje krótszy okres lub zamiennik z tej samej klasy aktywów (np. EUNL.DE → ACWI) i nie uruchamia obliczeń.

## Strony

1. **Centrum dowodzenia:** kluczowe wskaźniki, ścieżka akumulacji, struktura portfela, czynniki ryzyka i raport (Markdown/HTML).
2. **Planer FIRE:** 5 trybów (Traditional, Lean, Fat, Coast, Barista), z których każdy liczy cel inaczej, reguła przejścia na FIRE, zdarzenia finansowe i dodatkowe dochody.
3. **Konstruktor portfela:** Markowitz (min. wariancja, maks. Sharpe, docelowa stopa), Risk Parity, granica efektywna, CML, wkłady do ryzyka, CAPM/SML. Wybrany portfel można zaznaczyć na wykresie i zastosować w scenariuszu.
4. **Laboratorium strategii:** backtest walk-forward 8 strategii z wpłatami, kosztami i rebalancingiem. Liczy TWR, XIRR, CAGR, Sharpe, Sortino, maks. obsunięcie, betę, alfę, tracking error i IR.
5. **Laboratorium ryzyka:** Monte Carlo całego cyklu (100–10 000 ścieżek), model parametryczny (normalny lub t-Studenta, opcjonalnie Ledoit–Wolf) albo block bootstrap, VaR/ES, wykres wachlarzowy, rozkłady, mapa sukcesu, wrażliwość i scenariusze kryzysowe.
6. **Życie po FIRE:** strategie wypłat (stała realna z regułą 4% jako przypadkiem szczególnym, stały %, Guardrails, tylko dochód), podatek, dochody i eksperyment z ryzykiem sekwencji stóp zwrotu.
7. **Porównywarka scenariuszy:** tworzenie, klonowanie i usuwanie scenariuszy, import i eksport JSON, porównanie wyników.
8. **Metodologia:** wzory (`st.latex`), założenia i literatura.
9. **Profil i dane:** konto (Google lub profil lokalny), zapisane pliki, eksport i usunięcie danych.

## Struktura projektu

```
app.py                  nawigacja, ustawienia strony, panel boczny
views/                  9 stron Streamlit (tylko UI)
core/                   obliczenia, niezależne od Streamlit
  models.py             centralny model scenariusza (pydantic) + zależności wyników
  returns.py, metrics.py
  portfolio.py          estymacja momentów, optymalizacja, granica, CAPM
  fire.py, lifecycle.py wspólny silnik akumulacja → FIRE → wypłaty
  withdrawal.py, sequence.py, simulation.py, risk.py, backtest.py
services/               dane (yfinance/CSV/demo), stan sesji, cache, raporty
  db.py                 SQLite: schemat, migracje, repozytorium (bez Streamlit)
  persistence.py        zapis/odczyt scenariuszy z bazy, aktualizacja wersji schematu
  auth.py               logowanie Google (st.login) i profile lokalne
components/             formatowanie PL, motyw, KPI, wykresy, panel boczny
data/                   katalog instrumentów, dane demo, migawka
scripts/                generowanie danych demo, pobieranie migawki
tests/                  pytest + streamlit.testing (AppTest)
.streamlit/config.toml  motyw jasny i ciemny
```

Szczegółowe decyzje projektowe i metodologiczne są opisane w [PLAN.md](PLAN.md).

## Najważniejsze konwencje metodologiczne

- Silnik liczy w złotych realnych (dzisiejszych). Stopy nominalne deflowane są raz, dokładnym wzorem Fishera.
- Średnia arytmetyczna (optymalizacja) jest oddzielona od geometrycznej (projekcja deterministyczna, mediana).
- Kolejność zdarzeń w miesiącu: decyzja o FIRE, przepływy na początku miesiąca, stopa zwrotu, opłata za zarządzanie.
- Backtest wyznacza wagi tylko na danych sprzed dnia decyzji (`data.iloc[:t]`). Jest test na brak look-ahead.
- Sukces planu nie oznacza tylko dodatniego kapitału. Wymaga osiągnięcia FIRE, niewyczerpania kapitału i pokrycia co najmniej minimalnego poziomu wydatków w każdym miesiącu.
- Wyniki są powtarzalne dzięki ziarnu losowemu. Wyniki obliczone dla starszych parametrów są oznaczane jako nieaktualne na podstawie odcisków parametrów scenariusza.

## Ograniczenia

- **Dane:** Yahoo Finance nie było dostępne w środowisku, w którym powstawała aplikacja, więc tryb na żywo przetestowano tylko na danych zastępczych (testy jednostkowe parsowania i przeliczania FX). Dane demo są syntetyczne i nie nadają się do wniosków empirycznych. Katalog ETF ma obciążenie przeżycia (survivorship bias), a historia części funduszy zaczyna się po 2000 roku.
- **Inflacja i stopa wolna od ryzyka są stałe** w całym horyzoncie. Nie są modelowane stochastycznie ani skorelowane ze stopami zwrotu.
- **Podatki są uproszczone:** efektywna stawka od wypłat (gross-up). To nie jest model polskiego PIT, IKE/IKZE ani podatku Belki od zrealizowanych zysków.
- **Model zwrotów:** parametryczny Monte Carlo zakłada stałe w czasie średnie i kowariancje (bez zmiany reżimów). Block bootstrap powtarza tylko historię z okresu estymacji.
- **Guardrails** to uproszczona wersja reguł Guytona–Klingera.
- **Formaty liczb:** pola `number_input` Streamlit zawsze pokazują kropkę dziesiętną, a paski postępu w tabelach (`ProgressColumn`) formatują procenty według języka przeglądarki. Pozostałe liczby (tabele, wykresy, wskaźniki) są formatowane po polsku. W edytorach puste komórki są oznaczane przez Streamlit jako „None”.
- **Przechowywanie:** SQLite to jeden plik na serwerze aplikacji. Na hostingu z ulotnym dyskiem (np. Streamlit Community Cloud) baza znika przy restarcie, więc przy wdrożeniu potrzebny jest trwały dysk albo zewnętrzna baza. Goście (tryb Google bez logowania) mają scenariusze tylko w sesji, a logowanie zaczyna nową sesję, więc scenariusze gościa trzeba przed zalogowaniem wyeksportować do JSON.
- **Optymalizacja** (SLSQP) może nie zbiec przy skrajnych ograniczeniach. Status jest wtedy zgłaszany, a wyniki nie są ukrywane.
