"""SQLite persistence of install-lifetime token usage.

Session-scope token counters (app/metrics.py) reset when the proxy
restarts; this store keeps the cumulative since-installation totals so
they survive restarts and container recreation. Increments buffer in
memory and flush to one upsert batch at most every FLUSH_SECONDS (and at
interpreter exit), so a request never pays a disk round-trip unless one
is due. Event-loop-only, like the rest of the metrics: no locking beyond
SQLite's own.
"""

import atexit
import logging
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Optional, Tuple

from .config import get_settings

logger = logging.getLogger("llamaswap.token_store")

FLUSH_SECONDS = 5.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS token_totals (
    model      TEXT NOT NULL,
    direction  TEXT NOT NULL,
    tokens     INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (model, direction)
)
"""

_UPSERT = """
INSERT INTO token_totals (model, direction, tokens, updated_at)
VALUES (?, ?, ?, ?)
ON CONFLICT(model, direction)
DO UPDATE SET tokens = token_totals.tokens + excluded.tokens,
              updated_at = excluded.updated_at
"""


class TokenStore:
    """Cumulative (model, direction) token totals in one SQLite table."""

    def __init__(self, path: str = "") -> None:
        if not path:
            path = get_settings().tokens_db
        if not path:
            path = str(
                Path(__file__).resolve().parent.parent
                / "data" / "llamaswap_tokens.db"
            )
        self._path = path
        self._db: Optional[sqlite3.Connection] = None
        self._pending: DefaultDict[Tuple[str, str], int] = defaultdict(int)
        # Totals already on disk when this process started; the install
        # scope of a rendered metric is baseline + session, so a value is
        # never counted twice across the restart boundary.
        self._baseline: DefaultDict[Tuple[str, str], int] = defaultdict(int)
        self._last_flush = time.monotonic()
        self._open()
        atexit.register(self.flush)

    def _open(self) -> None:
        try:
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(self._path)
            db.execute(_SCHEMA)
            db.commit()
            for model, direction, tokens in db.execute(
                "SELECT model, direction, tokens FROM token_totals"
            ):
                self._baseline[(model, direction)] = int(tokens)
            self._db = db
            logger.info(
                "token totals in %s (%d models tracked)",
                self._path, len({m for m, _ in self._baseline}),
            )
        except (sqlite3.Error, OSError) as exc:
            logger.warning(
                "token totals disabled (%s): %s — session counters keep "
                "working, install totals will not persist", self._path, exc,
            )

    @property
    def baseline(self) -> dict:
        """Totals already persisted before this process started."""
        return dict(self._baseline)

    def add(self, model: str, direction: str, tokens: int) -> None:
        if tokens <= 0:
            return
        self._pending[(model, direction)] += tokens
        if time.monotonic() - self._last_flush >= FLUSH_SECONDS:
            self.flush()

    def flush(self) -> None:
        """Write pending increments; safe to call any number of times."""
        if not self._pending or self._db is None:
            return
        now = time.time()
        rows = [
            (model, direction, tokens, now)
            for (model, direction), tokens in self._pending.items()
        ]
        try:
            with self._db:
                self._db.executemany(_UPSERT, rows)
        except sqlite3.Error as exc:
            logger.warning(
                "token totals flush failed (%s): %s", self._path, exc,
            )
            return
        self._pending.clear()
        self._last_flush = time.monotonic()


# Process-wide singleton.
token_store = TokenStore()
