"""Who is using the app: Google sign-in (Streamlit OIDC) or local profiles.

Modes
-----
* **Google** – active when ``.streamlit/secrets.toml`` has an ``[auth.google]`` section
  (see ``.streamlit/secrets.example.toml``). Signed-in users get a persistent profile keyed by
  the OIDC ``sub`` claim; visitors who are not signed in work as guests (session only, as before).
  Optional ``[fire_lab] allowed_emails = [...]`` restricts who may sign in.
* **Local** – without Google configuration (e.g. running on your own computer) the app keeps
  named local profiles without passwords. They are separated, not secured.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import streamlit as st

from services.db import Database, slugify

USER_KEY = "current_user"
LOCAL_PROFILE_KEY = "local_profile_id"
DEFAULT_LOCAL_SUBJECT = "domyslny"
DEFAULT_LOCAL_NAME = "Profil lokalny"


@dataclass(frozen=True)
class CurrentUser:
    id: int | None  # None = guest, nothing is persisted
    display_name: str
    provider: str  # google | local | guest
    email: str | None = None
    picture_url: str | None = None

    @property
    def persisted(self) -> bool:
        return self.id is not None


@lru_cache(maxsize=4)
def _database(path: str | None) -> Database:
    return Database(path)


def database() -> Database:
    import os

    return _database(os.environ.get("FIRE_LAB_DB") or _secret("fire_lab", {}).get("db_path"))


def _secret(section: str, default: Any) -> Any:
    try:
        return st.secrets.get(section, default)
    except Exception:  # noqa: BLE001 - no secrets.toml at all
        return default


def google_configured() -> bool:
    google = (_secret("auth", {}) or {}).get("google") or {}
    return bool(google.get("client_id") and google.get("client_secret"))


def mode() -> str:
    return "google" if google_configured() else "local"


def _logged_in() -> bool:
    return bool(st.user.get("is_logged_in", False))


def _allowed(email: str | None) -> bool:
    allowed = [e.lower() for e in (_secret("fire_lab", {}) or {}).get("allowed_emails", [])]
    return not allowed or (email or "").lower() in allowed


def current_user() -> CurrentUser:
    """Resolve (and cache in the session) the user of this browser session."""
    ss = st.session_state
    if mode() == "google":
        if not _logged_in():
            ss.pop(USER_KEY, None)
            return CurrentUser(None, "Gość", "guest")
        cached = ss.get(USER_KEY)
        if cached and cached.provider == "google":
            return cached
        email = st.user.get("email")
        if not _allowed(email):
            return CurrentUser(None, "Gość", "guest", email=email)
        if st.user.get("email_verified") is False:
            return CurrentUser(None, "Gość", "guest", email=email)
        u = database().upsert_user("google", str(st.user.get("sub")), st.user.get("name") or email or "Użytkownik Google",
                                   email, st.user.get("picture"))
        ss[USER_KEY] = CurrentUser(u.id, u.display_name, "google", u.email, u.picture_url)
        return ss[USER_KEY]

    cached = ss.get(USER_KEY)
    if cached and cached.provider == "local":
        return cached
    db = database()
    user = db.get_user(ss[LOCAL_PROFILE_KEY]) if ss.get(LOCAL_PROFILE_KEY) else None
    if user is None:
        user = db.upsert_user("local", DEFAULT_LOCAL_SUBJECT, DEFAULT_LOCAL_NAME)
    ss[LOCAL_PROFILE_KEY] = user.id
    ss[USER_KEY] = CurrentUser(user.id, user.display_name, "local")
    return ss[USER_KEY]


def blocked_email() -> str | None:
    """E-mail of a signed-in Google account that is not on the allow-list (or unverified), else None."""
    if mode() != "google" or not _logged_in():
        return None
    email = st.user.get("email")
    if not _allowed(email) or st.user.get("email_verified") is False:
        return email or "?"
    return None


# ------------------------------------------------------------------ local profiles
def local_profiles() -> list:
    return database().list_users("local")


def create_local_profile(name: str) -> int:
    name = name.strip()
    if not name:
        raise ValueError("Nazwa profilu nie może być pusta.")
    db = database()
    subject = slugify(name)
    existing = {u.subject for u in db.list_users("local")}
    if subject in existing:
        raise ValueError("Profil o podobnej nazwie już istnieje.")
    return db.upsert_user("local", subject, name).id


def switch_local_profile(user_id: int) -> None:
    st.session_state[LOCAL_PROFILE_KEY] = user_id
    st.session_state.pop(USER_KEY, None)


def forget_cached_user() -> None:
    st.session_state.pop(USER_KEY, None)
