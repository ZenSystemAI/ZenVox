"""
test_history_stats.py — history stats aggregation + corrupt-db recovery.
"""
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from history import History


class StatsTests(unittest.TestCase):
    def _db(self):
        return History(db_path=Path(tempfile.mktemp(suffix=".db")))

    def test_empty_stats(self):
        s = self._db().stats()
        self.assertEqual(s["total_count"], 0)
        self.assertEqual(s["wpm"], 0)
        self.assertEqual(s["streak"], 0)

    def test_wpm_and_totals(self):
        h = self._db()
        now = datetime.now(timezone.utc).isoformat()
        h.add("raw one two three four", "one two three four", duration_sec=2.0, model="m")
        h.add("raw", "five six", duration_sec=1.0, model="m")
        s = h.stats()
        self.assertEqual(s["total_count"], 2)
        self.assertEqual(s["total_words"], 6)
        # 6 words over 3s = 120 wpm
        self.assertEqual(s["wpm"], 120)

    def test_streak_counts_today(self):
        h = self._db()
        h.add("a", "hello world", duration_sec=1.0)
        s = h.stats()
        self.assertGreaterEqual(s["streak"], 1)
        self.assertEqual(s["today_count"], 1)


class CorruptDbTests(unittest.TestCase):
    def test_corrupt_db_recovers(self):
        path = Path(tempfile.mktemp(suffix=".db"))
        path.write_bytes(b"this is not a sqlite database at all, just junk bytes")
        # Must not raise — moves the bad file aside and starts fresh.
        h = History(db_path=path)
        self.assertEqual(h.stats()["total_count"], 0)
        self.assertTrue(Path(str(path) + ".bad").exists())
        h.add("x", "recovered ok", duration_sec=1.0)
        self.assertEqual(h.stats()["total_count"], 1)


if __name__ == "__main__":
    unittest.main()
