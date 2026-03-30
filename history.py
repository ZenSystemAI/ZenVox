"""
history.py — SQLite transcription history for ZenScribe
"""
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from config import DB_FILE


@dataclass
class TranscriptionEntry:
    id: int
    timestamp: str
    raw_text: str
    cleaned_text: str
    language: str
    duration_sec: float
    model: str
    cleaning_preset: str


class History:
    def __init__(self, db_path=DB_FILE):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS transcriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                cleaned_text TEXT NOT NULL,
                language TEXT DEFAULT '',
                duration_sec REAL DEFAULT 0,
                model TEXT DEFAULT '',
                cleaning_preset TEXT DEFAULT 'General'
            )
        """)
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_ts ON transcriptions(timestamp DESC)")
        self._db.commit()

    def add(self, raw_text, cleaned_text, language="", duration_sec=0.0,
            model="", cleaning_preset="General"):
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO transcriptions (timestamp, raw_text, cleaned_text, language, duration_sec, model, cleaning_preset) VALUES (?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), raw_text, cleaned_text,
                 language, duration_sec, model, cleaning_preset))
            self._db.commit()
            return cur.lastrowid

    def get_recent(self, limit=50):
        with self._lock:
            rows = self._db.execute(
                "SELECT id,timestamp,raw_text,cleaned_text,language,duration_sec,model,cleaning_preset "
                "FROM transcriptions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [TranscriptionEntry(*r) for r in rows]

    def search(self, query):
        with self._lock:
            q = f"%{query}%"
            rows = self._db.execute(
                "SELECT id,timestamp,raw_text,cleaned_text,language,duration_sec,model,cleaning_preset "
                "FROM transcriptions WHERE cleaned_text LIKE ? OR raw_text LIKE ? ORDER BY id DESC LIMIT 50",
                (q, q)).fetchall()
            return [TranscriptionEntry(*r) for r in rows]

    def delete(self, entry_id):
        with self._lock:
            self._db.execute("DELETE FROM transcriptions WHERE id=?", (entry_id,))
            self._db.commit()

    def clear(self):
        with self._lock:
            self._db.execute("DELETE FROM transcriptions")
            self._db.commit()
