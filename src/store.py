"""Long-term memory: persist every Q&A turn to SQLite (survives restarts).

Buffer memory (the checkpointer in agent.py) lives in RAM and dies with the
process. This store is the durable record -- the "long-term memory" the brief
asks for. SQLite because it is a single file with zero setup; swap `_connect`
for psycopg / SQLAlchemy to use PostgreSQL without touching the rest of the code.
"""

import sqlite3
from contextlib import contextmanager

from src import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS qa (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id  TEXT NOT NULL,
    question   TEXT NOT NULL,
    answer     TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@contextmanager
def _connect(db_path):
    """Open a connection per operation (safe with Streamlit's threaded reruns)."""
    conn = sqlite3.connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


class QAStore:
    """Tiny data-access layer around the `qa` table."""

    def __init__(self, db_path=None):
        self.db_path = db_path or config.DB_PATH
        with _connect(self.db_path) as conn:
            conn.executescript(_SCHEMA)

    def save(self, thread_id: str, question: str, answer: str) -> None:
        with _connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO qa (thread_id, question, answer) VALUES (?, ?, ?)",
                (thread_id, question, answer),  # parameterized == no SQL injection
            )

    def history(self, thread_id: str, limit: int = 20) -> list[tuple[str, str]]:
        """Return recent (question, answer) pairs for a thread, oldest first."""
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT question, answer FROM qa WHERE thread_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (thread_id, limit),
            ).fetchall()
        return list(reversed(rows))

    def all_pairs(self) -> list[tuple[str, str]]:
        """Every (question, answer) ever stored -- used to warm vector memory."""
        with _connect(self.db_path) as conn:
            return conn.execute("SELECT question, answer FROM qa ORDER BY id").fetchall()


if __name__ == "__main__":
    import os
    import tempfile

    path = os.path.join(tempfile.gettempdir(), "qa_store_demo.db")
    store = QAStore(path)
    store.save("t1", "What is Python?", "A programming language.")
    store.save("t1", "Who created it?", "Guido van Rossum.")
    print("history:", store.history("t1"))
    os.remove(path)
