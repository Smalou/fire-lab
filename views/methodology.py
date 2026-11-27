"""Strona 8 – Metodologia (sekcja akademicka)."""

from __future__ import annotations

import streamlit as st

from components.layout import page_header

page_header("Metodologia", "Podstawy teoretyczne, definicje wskaźników, założenia modeli i ich ograniczenia.")

tabs = st.tabs(["FIRE", "Markowitz i CAPM", "Wskaźniki", "Optymalizacja", "Backtesting", "Monte Carlo i ryzyko",
                "Dekumulacja", "Ograniczenia", "Dane i literatura"])

with tabs[0]:
    st.markdown("""
**FIRE (Financial Independence, Retire Early – niezależność finansowa, wcześniejsza emerytura)** to strategia polegająca na
zgromadzeniu kapitału, którego dochód pozwala pokrywać wydatki bez konieczności pracy zarobkowej. Kluczowa relacja wiąże
roczne wydatki *E* z bezpieczną stopą wypłat *SWR (Safe Withdrawal Rate)*.
""")
    st.latex(r"C_{FIRE} = \frac{E}{SWR}\qquad\text{np. } \frac{90\,000}{0{,}04} = 2\,250\,000")
    with st.expander("Warianty FIRE i ich logika w aplikacji", expanded=True):
        st.markdown("""
| Wariant | Kapitał docelowy | Logika symulacji |
|---|---|---|
| **Traditional** | E / SWR | wpłaty do FIRE, potem wypłaty pokrywają pełne wydatki |
| **Lean** | s·E / SWR | finansowane wyłącznie wydatki podstawowe (udział *s*) |
| **Fat** | E(1+m)/SWR + k·E(1+m) | wyższe wydatki i bufor gotówkowy *k* lat; po roku ze stratą wypłaty z bufora |
| **Coast** | E / SWR oraz C/(1+g)^(T−t) | wpłaty ustają po osiągnięciu kwoty Coast; portfel rośnie sam do wieku FIRE |
| **Barista** | (E − I_part)/SWR | dochód z pracy w niepełnym wymiarze do wskazanego wieku zmniejsza wypłaty |
""")
        st.latex(r"C_{Coast}(t) = \frac{C_{FIRE}}{(1+g_{real,netto})^{T-t}}")
    with st.expander("Realne i nominalne wartości"):
        st.markdown("Kwoty wprowadzane przez użytkownika są **realne** (w dzisiejszych złotych). Rynkowe stopy zwrotu są "
                    "**nominalne** i deflowane dokładnie raz relacją Fishera. Wartości nominalne pokazywane są przez przemnożenie "
                    "przez indeks cen – inflacja nie jest naliczana podwójnie.")
        st.latex(r"r_{real} = \frac{1+r_{nom}}{1+\pi}-1,\qquad X_{nom,t} = X_{real,t}\,(1+\pi)^{t}")
    with st.expander("Kapitał jako wartość bieżąca wypłat (horyzont skończony)"):
        st.latex(r"C_{PV} = \sum_{m=0}^{M-1} \frac{\max(E_m - I_m, 0)}{(1+r_m)^{m}}")
        st.markdown("Uzupełnienie reguły SWR: uwzględnia emeryturę i inne dochody oraz skończony horyzont życia.")

with tabs[1]:
    st.markdown("**Teoria portfelowa Markowitza** – inwestor wybiera portfel na podstawie oczekiwanej stopy zwrotu i wariancji.")
    st.latex(r"E[R_p] = \mathbf{w}^\top\boldsymbol{\mu},\qquad \sigma_p^2 = \mathbf{w}^\top\boldsymbol{\Sigma}\mathbf{w},\qquad \mathbf{1}^\top\mathbf{w}=1")
    st.markdown("Granica efektywna to zbiór portfeli o minimalnej wariancji dla każdej zadanej oczekiwanej stopy zwrotu:")
    st.latex(r"\min_{\mathbf{w}}\; \mathbf{w}^\top\boldsymbol{\Sigma}\mathbf{w}\quad\text{p.w.}\quad \mathbf{w}^\top\boldsymbol{\mu}=\mu^*,\; \mathbf{1}^\top\mathbf{w}=1,\; l\le w_i\le u")
    st.markdown("**Capital Market Line (CML)** – przy możliwości pożyczania i lokowania po stopie wolnej od ryzyka optymalne są "
                "kombinacje aktywa wolnego od ryzyka i portfela stycznego (maksymalny Sharpe):")
    st.latex(r"E[R] = r_f + \frac{E[R_T]-r_f}{\sigma_T}\,\sigma")
    st.markdown("**CAPM (Capital Asset Pricing Model)** – w równowadze oczekiwana stopa aktywa zależy od ryzyka systematycznego β:")
    st.latex(r"E[R_i] = r_f + \beta_i\left(E[R_M]-r_f\right),\qquad \beta_i = \frac{\mathrm{Cov}(R_i,R_M)}{\mathrm{Var}(R_M)}")
    st.caption("Portfel rynkowy przybliżany jest wybranym indeksem akcji – to uproszczenie empiryczne.")

with tabs[2]:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Stopa skumulowana i CAGR** (średnioroczna geometryczna stopa zwrotu)")
        st.latex(r"R_{cum} = \prod_{t=1}^{T}(1+r_t) - 1,\qquad CAGR = (1+R_{cum})^{12/T}-1")
        st.markdown("**TWR (Time-Weighted Return – stopa ważona czasem)** – eliminuje wpływ wpłat:")
        st.latex(r"r^{TWR}_t = \frac{V_t}{V_{t-1}+CF_t}-1")
        st.markdown("**XIRR (Money-Weighted Return – stopa ważona kapitałem)**:")
        st.latex(r"\sum_{k} \frac{CF_k}{(1+XIRR)^{(d_k-d_0)/365}} = 0")
        st.markdown("**Zmienność annualizowana**")
        st.latex(r"\sigma_{rok} = \sigma_{mies}\sqrt{12}")
    with c2:
        st.markdown("**Sharpe Ratio** i **Sortino Ratio** (MAR = stopa wolna od ryzyka)")
        st.latex(r"SR = \frac{\overline{r - r_f}}{s(r-r_f)}\sqrt{12},\qquad SoR = \frac{\overline{r-r_f}}{\sqrt{\tfrac1T\sum \min(r_t-r_f,0)^2}}\sqrt{12}")
        st.markdown("**Maximum Drawdown** (największe obsunięcie od szczytu indeksu TWR)")
        st.latex(r"MDD = \min_t \left(\frac{I_t}{\max_{s\le t} I_s}-1\right)")
        st.markdown("**Beta, alfa Jensena, Tracking Error, Information Ratio**")
        st.latex(r"\alpha = 12\left[\overline{r-r_f} - \beta\,\overline{r_b-r_f}\right],\quad TE = s(r-r_b)\sqrt{12},\quad IR = \frac{12\,\overline{r-r_b}}{TE}")
    st.info("Średnia arytmetyczna ≠ geometryczna: dla zmiennych stóp geometryczna jest niższa (≈ μ − σ²/2). "
            "Do długoterminowych projekcji aplikacja używa stopy geometrycznej.", icon=":material/lightbulb:")
    st.latex(r"g = \exp\!\left(\ln(1+\mu) - \tfrac12\ln\!\left(1+\tfrac{\sigma^2}{(1+\mu)^2}\right)\right) - 1")

with tabs[3]:
    st.markdown("""
- **Equal Weight** – 1/N; nie wymaga estymacji, co ogranicza błąd estymacji.
- **Minimum Variance** – minimalizacja wᵀΣw; nie korzysta z (najmniej wiarygodnych) estymacji średnich.
- **Maximum Sharpe** – portfel styczny; bardzo wrażliwy na błędy μ.
- **Risk Parity (Equal Risk Contribution)** – równe udziały w ryzyku; rozwiązywany w postaci wypukłej (Spinu), a przy
  aktywnych ograniczeniach wag – metodą SLSQP.
- **Zadana stopa zwrotu** – minimalna wariancja przy ograniczeniu wᵀμ = μ*; aplikacja sprawdza osiągalność μ*.
""")
    st.latex(r"RC_i = w_i\frac{(\boldsymbol{\Sigma}\mathbf{w})_i}{\sigma_p},\qquad \sum_i RC_i = \sigma_p")
    st.latex(r"\min_{\mathbf{y}>0}\ \tfrac12\mathbf{y}^\top\boldsymbol\Sigma\mathbf{y} - \sum_i b_i\ln y_i,\qquad \mathbf{w}=\mathbf{y}/\mathbf{1}^\top\mathbf{y}")
    with st.expander("Kontrola numeryczna"):
        st.markdown("""
- Optymalizacja SLSQP (SciPy) z kilkoma punktami startowymi; zwracany jest status zbieżności, a brak zbieżności jest
  komunikowany użytkownikowi (nie jest ukrywany).
- Suma wag kontrolowana (|Σw − 1| < 10⁻⁴), zakaz krótkiej sprzedaży egzekwowany ograniczeniami.
- Macierz kowariancji: estymator próbkowy (nieobciążony) lub Ledoit–Wolf; dodatnia półokreśloność sprawdzana wartościami
  własnymi, w razie potrzeby rzut na stożek PSD (obcięcie ujemnych wartości własnych).
""")

with tabs[4]:
    st.markdown("""
**Backtest walk-forward.** Na każdej dacie rebalancingu parametry są estymowane wyłącznie na oknie danych kończącym się
miesiąc przed decyzją. Okres rozgrzewki (długość okna) jest wspólny dla wszystkich strategii, by porównanie dotyczyło
tego samego okresu.

**Konwencja czasu (wspólna dla całej aplikacji):** na początku miesiąca następuje wpłata/wypłata i ewentualny rebalancing,
następnie realizuje się stopa zwrotu miesiąca, a na końcu potrącana jest opłata za zarządzanie.

**Koszty:** prowizja (% obrotu) i koszt obrotu/spread (pb) naliczane od wartości transakcji; opłata za zarządzanie od
wartości portfela.

**Wyniki vs wpłaty:** TWR, CAGR, zmienność, Sharpe i MDD liczone są na stopach ważonych czasem (bez wpływu wpłat);
XIRR i zysk inwestycyjny (wartość − suma wpłat) pokazują perspektywę inwestora.

**Glide Path** – udział akcji maleje liniowo od wartości początkowej do końcowej w okresie testu.
""")
    st.warning("Optymalizacja na całym okresie i prezentowanie jej wyników jako backtestu prowadzi do look-ahead bias. "
               "W aplikacji wagi strategii estymowanych są wyznaczane wyłącznie z przeszłych danych, co weryfikuje test jednostkowy.",
               icon=":material/warning:")

with tabs[5]:
    st.markdown("""
**Model parametryczny.** Miesięczne logarytmiczne stopy zwrotu aktywów modelowane są wielowymiarowym rozkładem normalnym
(lub t-Studenta o tej samej macierzy kowariancji – „grube ogony”). Korelacje między aktywami są zachowane.

**Historyczny block bootstrap.** Losowane są bloki kolejnych miesięcy historycznych (kołowo); zachowuje to rozkład brzegowy,
korelacje i krótkookresową autokorelację, ale nie tworzy scenariuszy spoza historii.

**Reprodukowalność.** Generator `numpy.random.default_rng(seed)`; ścieżki generowane w stałych paczkach.

**Cykl życia.** Każda ścieżka przechodzi przez akumulację (wpłaty indeksowane realnie, ograniczone nadwyżką dochodu),
przejście na FIRE (w docelowym wieku lub po osiągnięciu celu) i dekumulację zgodną z wybraną strategią wypłat.
""")
    st.latex(r"\mathbf{x}_t\sim\mathcal N(\boldsymbol\mu,\boldsymbol\Sigma),\quad \mathbf{x}_t = \boldsymbol\mu + \sqrt{\tfrac{\nu-2}{\nu}}\,\frac{\mathbf{L}\mathbf{z}_t}{\sqrt{\chi^2_\nu/\nu}}\ \ (t\text{-Student})")
    st.markdown("**VaR i Expected Shortfall** – definiowane na rozkładzie skumulowanej stopy zwrotu portfela w horyzoncie H "
                "(bez przepływów) i poziomie ufności α:")
    st.latex(r"\mathrm{VaR}_{\alpha,H} = -q_{1-\alpha}(R_H),\qquad \mathrm{ES}_{\alpha,H} = -\mathbb E[R_H\mid R_H\le q_{1-\alpha}(R_H)]")
    st.markdown("**Scenariusze kryzysowe** – hipotetyczne szoki dla klas aktywów oraz historyczne okna (tylko gdy wszystkie aktywa "
                "mają dane). **Analiza wrażliwości** – zmiana jednego parametru przy tych samych ścieżkach losowych.")

with tabs[6]:
    st.markdown("""
- **Stała realna wypłata** – kwota ustalana w chwili FIRE (jako % kapitału lub równa wymaganym wydatkom) i utrzymywana realnie.
  **Reguła 4%** jest jej szczególnym przypadkiem (stopa początkowa 4%), a nie odrębnym mechanizmem.
- **Stały procent portfela** – co roku p·V_t; brak wyczerpania kapitału kosztem zmiennej konsumpcji.
- **Guardrails** (uproszczona reguła Guytona–Klingera) – korekty wypłaty po przekroczeniu korytarza wokół stopy początkowej
  i brak indeksacji po roku ze stratą.
- **Tylko dochód portfela** – wypłata dywidend i kuponów (stopa dochodu z rzeczywistych danych o wypłatach); dochód jest
  częścią stopy całkowitej, więc nie jest naliczany podwójnie.

**Miary sukcesu:** prawdopodobieństwo niewyczerpania kapitału, prawdopodobieństwo pokrycia wydatków (konsumpcja ≥ próg
× wymagane wydatki w każdym miesiącu), rozkład wieku wyczerpania, mediana majątku końcowego, łączne wypłaty, zmienność
konsumpcji i największy wymagany spadek wydatków. Dodatni kapitał nie jest tożsamy z sukcesem.

**Sequence of Returns Risk** – przy regularnych wypłatach kolejność stóp ma znaczenie: straty na początku dekumulacji
zmniejszają bazę kapitału, która później korzysta z odbicia.
""")
    st.latex(r"V_{t+1} = (V_t - W_t)(1+r_t)\quad\Rightarrow\quad V_T \ne V_0\prod(1+r_t) - \sum W_t \prod(\dots)\ \text{zależy od kolejności } r_t")

with tabs[7]:
    st.markdown("""
- **Błąd estymacji** – średnie stóp zwrotu estymowane z kilkunastu lat danych mają duży błąd standardowy; wyniki optymalizacji
  (zwłaszcza Maximum Sharpe) są na niego wrażliwe.
- **Stacjonarność** – modele zakładają stałe parametry; nie modelują zmian reżimów, stóp procentowych ani wycen.
- **Survivorship bias** – katalog obejmuje instrumenty istniejące dziś; fundusze zlikwidowane nie są uwzględnione.
- **Inflacja i stopa wolna od ryzyka** – stałe parametry, a nie historyczne szeregi.
- **Podatki** – model podstawowy bez podatków lub z jawnym, uproszczonym efektywnym podatkiem od wypłat; nie jest to model
  polskiego systemu podatkowego (np. podatku od zysków kapitałowych, IKE/IKZE).
- **Długowieczność** – horyzont życia jest parametrem deterministycznym.
- **Dane demonstracyjne** są syntetyczne i służą wyłącznie ilustracji działania metod.
- **Wyniki historyczne nie są gwarancją przyszłych wyników.**
""")

with tabs[8]:
    st.markdown("""
**Źródła danych**
- Yahoo Finance za pośrednictwem biblioteki open-source *yfinance* (ceny skorygowane o dywidendy i splity, dywidendy,
  kursy walut). Biblioteka korzysta z nieoficjalnego interfejsu; dostępność i poprawność danych nie są gwarantowane.
- Dane demonstracyjne: syntetyczne (skrypt `scripts/generate_demo_data.py`).
- Import własnych danych CSV.

**Literatura** (przed cytowaniem w pracy zweryfikuj szczegóły bibliograficzne w źródle)
- Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, 7(1).
- Sharpe, W. F. (1964). Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk. *The Journal of Finance*, 19(3).
- Lintner, J. (1965). The Valuation of Risk Assets and the Selection of Risky Investments in Stock Portfolios and Capital Budgets. *The Review of Economics and Statistics*, 47(1).
- Sharpe, W. F. (1966). Mutual Fund Performance. *The Journal of Business*, 39(1).
- Jensen, M. C. (1968). The Performance of Mutual Funds in the Period 1945–1964. *The Journal of Finance*, 23(2).
- Sortino, F. A., Price, L. N. (1994). Performance Measurement in a Downside Risk Framework. *The Journal of Investing*, 3(3).
- Bengen, W. P. (1994). Determining Withdrawal Rates Using Historical Data. *Journal of Financial Planning*, 7(4).
- Cooley, P. L., Hubbard, C. M., Walz, D. T. (1998). Retirement Savings: Choosing a Withdrawal Rate That Is Sustainable. *AAII Journal*, 20(2).
- Guyton, J. T., Klinger, W. J. (2006). Decision Rules and Maximum Initial Withdrawal Rates. *Journal of Financial Planning*, 19(3).
- Artzner, P., Delbaen, F., Eber, J.-M., Heath, D. (1999). Coherent Measures of Risk. *Mathematical Finance*, 9(3).
- Ledoit, O., Wolf, M. (2004). A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices. *Journal of Multivariate Analysis*, 88(2).
- Künsch, H. R. (1989). The Jackknife and the Bootstrap for General Stationary Observations. *The Annals of Statistics*, 17(3).
- Maillard, S., Roncalli, T., Teïletche, J. (2010). The Properties of Equally Weighted Risk Contribution Portfolios. *The Journal of Portfolio Management*, 36(4).
- DeMiguel, V., Garlappi, L., Uppal, R. (2009). Optimal Versus Naive Diversification: How Inefficient Is the 1/N Portfolio Strategy? *The Review of Financial Studies*, 22(5).
- Robin, V., Dominguez, J. (1992). *Your Money or Your Life*. Viking.
- Bodie, Z., Kane, A., Marcus, A. J. *Investments*. McGraw-Hill (wydania kolejne).
- Elton, E. J., Gruber, M. J., Brown, S. J., Goetzmann, W. N. *Modern Portfolio Theory and Investment Analysis*. Wiley (wydania kolejne).
""")
