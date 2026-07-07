"""
history.py — SQLite transcription history for ZenVox
"""
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from config import DB_FILE

log = logging.getLogger("zenvox")


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
        try:
            self._db = self._open(db_path)
        except sqlite3.DatabaseError as e:
            # A corrupt history.db (truncated file, bad shutdown) used to abort
            # startup entirely — no tray, dictation dead. Keep the evidence and
            # start fresh instead.
            log.error(f"history.db unreadable ({e}) — moving to history.db.bad "
                      f"and starting a fresh history")
            try:
                os.replace(str(db_path), str(db_path) + ".bad")
            except OSError:
                pass
            self._db = self._open(db_path)

    @staticmethod
    def _open(db_path):
        db = sqlite3.connect(str(db_path), check_same_thread=False)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("""
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
        db.execute("CREATE INDEX IF NOT EXISTS idx_ts ON transcriptions(timestamp DESC)")
        db.commit()
        try:
            os.chmod(str(db_path), 0o600)  # every word ever dictated lives here
        except OSError:
            pass
        return db

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
            # Deleted rows stay recoverable in freed pages/WAL until vacuumed —
            # "Clear History" should actually remove the words from disk.
            self._db.execute("VACUUM")

    def stats(self):
        """Aggregate usage stats for the Home view (words, WPM, streak)."""
        with self._lock:
            rows = self._db.execute(
                "SELECT timestamp, cleaned_text, duration_sec FROM transcriptions"
            ).fetchall()
        days = {}
        total_words = 0
        total_dur = 0.0
        for ts, text, dur in rows:
            try:
                day = datetime.fromisoformat(ts).astimezone().date()
            except (ValueError, TypeError):
                continue
            words = len((text or "").split())
            d = days.setdefault(day, {"count": 0, "words": 0, "dur": 0.0})
            d["count"] += 1
            d["words"] += words
            d["dur"] += dur or 0.0
            total_words += words
            total_dur += dur or 0.0
        today = datetime.now().date()
        today_stats = days.get(today, {"count": 0, "words": 0, "dur": 0.0})
        wpm = round(total_words / (total_dur / 60.0)) if total_dur > 0 else 0
        # Streak: consecutive days with at least one dictation, ending today
        # (or yesterday, so the streak isn't 0 first thing in the morning).
        streak = 0
        day = today if today in days else today - timedelta(days=1)
        while day in days:
            streak += 1
            day -= timedelta(days=1)
        return {
            "today_count": today_stats["count"],
            "today_words": today_stats["words"],
            "wpm": wpm,
            "streak": streak,
            "total_count": len(rows),
            "total_words": total_words,
            "total_minutes": round(total_dur / 60.0, 1),
        }

    def close(self):
        with self._lock:
            self._db.close()
