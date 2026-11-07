import sqlite3
import time

from app.token_store import TokenStore


def rows(path):
    db = sqlite3.connect(path)
    try:
        return {
            (model, direction): int(tokens)
            for model, direction, tokens in db.execute(
                "SELECT model, direction, tokens FROM token_totals"
            )
        }
    finally:
        db.close()


def test_round_trip(tmp_path):
    path = str(tmp_path / "tokens.db")
    store = TokenStore(path)
    assert store.baseline == {}
    store.add("m", "input", 100)
    store.add("m", "output", 50)
    store.add("other", "input", 7)
    store.flush()
    assert rows(path) == {
        ("m", "input"): 100,
        ("m", "output"): 50,
        ("other", "input"): 7,
    }
    again = TokenStore(path)
    assert again.baseline == rows(path)


def test_debounce_then_accumulate(tmp_path):
    path = str(tmp_path / "tokens.db")
    store = TokenStore(path)
    store.add("m", "input", 5)
    assert store._pending == {("m", "input"): 5}
    assert rows(path) == {}
    store.flush()
    assert rows(path) == {("m", "input"): 5}
    store.add("m", "input", 2)
    store.add("m", "input", 3)
    store.flush()
    assert rows(path) == {("m", "input"): 10}
    assert store._pending == {}


def test_add_due_time_flushes(tmp_path):
    path = str(tmp_path / "tokens.db")
    store = TokenStore(path)
    store.add("m", "input", 5)
    store._last_flush -= 10
    store.add("m", "output", 1)
    assert store._pending == {}
    assert rows(path) == {("m", "input"): 5, ("m", "output"): 1}


def test_non_positive_ignored(tmp_path):
    store = TokenStore(str(tmp_path / "tokens.db"))
    store.add("m", "input", 0)
    store.add("m", "input", -5)
    assert store._pending == {}


def test_broken_path_degrades_to_session_only(tmp_path):
    store = TokenStore(str(tmp_path))
    assert store._db is None
    store.add("m", "input", 5)
    store.flush()
    assert store.baseline == {}


def test_flush_is_idle_safe(tmp_path):
    store = TokenStore(str(tmp_path / "tokens.db"))
    store.flush()
    store.flush()
    time.sleep(0.01)
