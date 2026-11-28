"""Strona – Profil i dane użytkownika."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from components.formatting import fmt_datetime
from components.layout import page_header
from services import auth, state

state.init()
page_header("Profil i dane", "Kto korzysta z aplikacji, co jest zapisane w bazie danych i jak tym zarządzać.")

u = state.user()
mode = auth.mode()

if not u.persisted:
    if mode == "google":
        st.info("Pracujesz jako gość. Zaloguj się kontem Google, aby scenariusze, ustawienia i pliki danych "
                "zapisywały się automatycznie i były dostępne przy kolejnych wizytach.", icon=":material/person:")
        st.button("Zaloguj przez Google", icon=":material/login:", type="primary", on_click=st.login, args=("google",),
                  key="ui_profile_login")
    st.stop()

db = auth.database()
record = db.get_user(u.id)
PROVIDERS = {"google": "Konto Google", "local": "Profil lokalny (bez hasła)"}

with st.container(border=True):
    c1, c2 = st.columns([1, 12], vertical_alignment="center")
    if u.picture_url:
        c1.image(u.picture_url, width=72)
    else:
        c1.markdown("## :material/account_circle:")
    c2.markdown(f"**{u.display_name}**" + (f"  \n{u.email}" if u.email else ""))
    c2.caption(f"{PROVIDERS[u.provider]} · utworzony {fmt_datetime(pd.Timestamp(record.created_at))} · "
               f"ostatnie logowanie {fmt_datetime(pd.Timestamp(record.last_login_at))}")

stored = db.list_scenarios(u.id)
datasets = db.list_datasets(u.id)
c1, c2, c3 = st.columns(3)
c1.metric("Zapisane scenariusze", len(stored))
c2.metric("Zapisane pliki danych", len(datasets))
c3.metric("Rozmiar bazy (cała aplikacja)", f"{db.path.stat().st_size / 1024:,.0f} KB".replace(",", " "))

if u.provider == "local":
    with st.form("rename_profile"):
        name = st.text_input("Nazwa profilu", u.display_name, max_chars=60)
        if st.form_submit_button("Zmień nazwę", icon=":material/edit:"):
            if not name.strip():
                st.error("Nazwa profilu nie może być pusta.")
            else:
                db.rename_user(u.id, name)
                auth.forget_cached_user()
                st.rerun()

st.subheader("Pliki danych", anchor=False)
if not datasets:
    st.caption("Brak zapisanych plików. Pliki CSV wczytane w oknie „Dane” w panelu bocznym zapisują się tutaj automatycznie.")
else:
    active_ds = (st.session_state.get("csv_dataset") or {}).get("db_id")
    st.dataframe(pd.DataFrame([{"Plik": d.name, "Wczytano": fmt_datetime(pd.Timestamp(d.created_at)),
                                "Rozmiar (KB)": round(len(d.content) / 1024, 1), "Aktywny": d.id == active_ds}
                               for d in datasets]), hide_index=True, width="stretch")
    by_id = {d.id: d for d in datasets}
    with st.container(horizontal=True, vertical_alignment="bottom"):
        pick = st.selectbox("Plik", list(by_id), format_func=lambda i: by_id[i].name, key="ui_profile_ds")
        if st.button("Usuń plik", icon=":material/delete:"):
            db.delete_dataset(u.id, pick)
            if active_ds == pick:
                st.session_state["csv_dataset"] = None
            st.toast("Usunięto plik z profilu.", icon=":material/delete:")
            st.rerun()

st.subheader("Eksport i usunięcie danych", anchor=False)
st.download_button("Pobierz wszystkie moje dane (JSON)",
                   json.dumps(db.export_user(u.id), ensure_ascii=False, indent=2, default=str).encode("utf-8"),
                   "fire_lab_moje_dane.json", "application/json", icon=":material/download:",
                   help="Profil, ustawienia, scenariusze i lista plików danych (bez ich zawartości).")


@st.dialog("Usuń profil i dane", icon=":material/delete_forever:")
def delete_profile_dialog() -> None:
    st.markdown(f"Usunięte zostaną: profil **{u.display_name}**, {len(stored)} scenariusz(e/y), {len(datasets)} plik(i) "
                "danych i wszystkie ustawienia. Tej operacji nie można cofnąć.")
    confirm = st.text_input("Wpisz nazwę profilu, aby potwierdzić")
    if st.button("Usuń na zawsze", type="primary", icon=":material/delete_forever:", disabled=confirm.strip() != u.display_name):
        db.delete_user(u.id)
        if u.provider == "google":
            st.logout()
        auth.switch_local_profile(0)  # falls back to the default local profile
        state.reload()
        st.rerun()


if st.button("Usuń profil i wszystkie dane…", icon=":material/delete_forever:"):
    delete_profile_dialog()

with st.expander("Gdzie są przechowywane dane?", icon=":material/database:"):
    st.markdown(
        f"Baza SQLite: `{db.path}`.\n\n"
        "- **Scenariusze** zapisują się automatycznie po każdym zatwierdzeniu zmian (jako dokument JSON z wersją schematu).\n"
        "- **Ustawienia** profilu: aktywny scenariusz i aktywny plik danych.\n"
        "- **Wyniki analiz** (backtest, Monte Carlo) nie są zapisywane – liczą się ponownie z zapisanych ustawień.\n\n"
        "Logowanie Google włącza się w pliku `.streamlit/secrets.toml` (instrukcja w README)."
    )
