"""
Application settings and store-ID registry.

Settings are read from environment variables (SI_* prefix) with safe defaults
so the app runs out-of-the-box without a .env file. Tests override individual
vars via monkeypatch; no global mutable state is held after module load.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).parent.parent
_CONFIG_DIR = _ROOT / "config"


class Settings:
    def __init__(self) -> None:
        self.db_path: str = os.getenv("SI_DB_PATH", "store_intelligence.db")
        self.pos_csv: str = os.getenv("SI_POS_CSV", str(_ROOT / "data" / "pos_transactions.csv"))
        self.layout_path: str = os.getenv("SI_LAYOUT_PATH", str(_CONFIG_DIR / "store_layout.json"))
        self.stores_path: str = os.getenv("SI_STORES_PATH", str(_CONFIG_DIR / "stores.yaml"))
        self.stale_feed_minutes: int = int(os.getenv("SI_STALE_FEED_MINUTES", "10"))
        self.session_timeout_s: int = int(os.getenv("SI_SESSION_TIMEOUT_S", "1800"))
        self.reentry_gap_s: int = int(os.getenv("SI_REENTRY_GAP_S", "60"))
        self.pos_window_minutes: int = int(os.getenv("SI_POS_WINDOW_MINUTES", "5"))


# Cached at process level. Tests that need different values should use
# monkeypatch.setenv + get_settings.cache_clear().
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def _load_aliases() -> dict[str, str]:
    path = Path(get_settings().stores_path)
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("aliases", {})


def resolve_store_id(raw_id: str) -> str:
    """Map any alias/variant to the canonical store ID. Unknown IDs pass through."""
    aliases = _load_aliases()
    return aliases.get(raw_id, raw_id)


def load_store_layout() -> dict[str, Any]:
    path = Path(get_settings().layout_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
