"""
dictionary.py — custom vocabulary for ZenVox (the Wispr-Flow "dictionary" feature).

Words/names/jargon the user adds so they come out spelled correctly every time.
Three correction layers, applied in order by the engine:

  A. BIAS    — `hotwords_string()` is passed to faster-whisper, nudging the
               acoustic decode toward the right spelling. Probabilistic.
  B. REPLACE — `apply_replacements()` runs deterministic, word-boundary
               find/replace on the RAW transcript. The guarantee layer; runs
               before the LLM so even the no-API raw path is corrected.
  C. PROMPT  — `prompt_block()` is appended to the LLM cleaning system prompt
               so the model treats these spellings as intentional and won't
               "fix" a brand name back to a dictionary word.

Persisted as its own JSON file (DICT_FILE) so it stays independent of
settings.json and can be exported/shared. Empty dictionary = zero overhead.
"""
import json
import logging
import re
import threading
from dataclasses import asdict, dataclass, field

from config import DICT_FILE

log = logging.getLogger("zenvox")

# Spoken forms shorter than this are ignored for find/replace (Layer B) to avoid
# over-matching common words. They still contribute to bias/prompt layers.
MIN_SPOKEN_LEN = 2
# Cap how many terms feed the hotwords bias — faster-whisper truncates the
# hotword token block, so an unbounded list silently loses its tail anyway.
MAX_HOTWORDS = 96


@dataclass
class DictionaryEntry:
    written: str                                  # canonical spelling, e.g. "ZenVox"
    spoken: list = field(default_factory=list)    # variants to replace, e.g. ["zen vox"]
    boost_only: bool = False                      # bias + prompt only, no find/replace
    case_sensitive: bool = False
    enabled: bool = True

    @classmethod
    def from_dict(cls, d):
        return cls(
            written=str(d.get("written", "")).strip(),
            spoken=[str(s).strip() for s in d.get("spoken", []) if str(s).strip()],
            boost_only=bool(d.get("boost_only", False)),
            case_sensitive=bool(d.get("case_sensitive", False)),
            enabled=bool(d.get("enabled", True)),
        )


class Dictionary:
    def __init__(self, entries=None, path=DICT_FILE):
        self._lock = threading.Lock()
        self._path = path
        self.entries = entries or []
        self._compiled = None  # cached [(pattern, written)] for Layer B

    # ── Persistence ───────────────────────────────────────────────────────
    @classmethod
    def load(cls, path=DICT_FILE):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entries = [DictionaryEntry.from_dict(d) for d in data.get("entries", [])]
            entries = [e for e in entries if e.written]
            return cls(entries=entries, path=path)
        except FileNotFoundError:
            return cls(path=path)
        except Exception as e:
            log.error(f"Dictionary load failed ({e}); starting empty")
            return cls(path=path)

    def save(self):
        with self._lock:
            data = {"version": 1, "entries": [asdict(e) for e in self.entries]}
            self._compiled = None  # vocabulary changed — drop compiled cache
        try:
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            import os
            os.replace(str(tmp), str(self._path))
        except Exception as e:
            log.error(f"Dictionary save failed: {e}")

    # ── CRUD ──────────────────────────────────────────────────────────────
    def add(self, entry):
        with self._lock:
            # Replace an existing entry with the same canonical spelling.
            self.entries = [e for e in self.entries
                            if e.written.lower() != entry.written.lower()]
            self.entries.append(entry)
        self.save()

    def delete(self, written):
        with self._lock:
            self.entries = [e for e in self.entries if e.written != written]
        self.save()

    def clear(self):
        with self._lock:
            self.entries = []
        self.save()

    def __len__(self):
        return len(self.entries)

    # ── Layer A: Whisper bias ─────────────────────────────────────────────
    def hotwords_string(self):
        terms = [e.written for e in self.entries if e.enabled and e.written]
        if not terms:
            return ""
        return ", ".join(terms[:MAX_HOTWORDS])

    # ── Layer B: deterministic find/replace on raw text ───────────────────
    def _compile(self):
        compiled = []
        # Longest spoken forms first so multi-word variants win over substrings.
        pairs = []
        for e in self.entries:
            if not e.enabled or e.boost_only or not e.written:
                continue
            for sp in e.spoken:
                # Skip only an exact-string no-op; a case-only difference
                # (e.g. "ios" -> "iOS", "zen vox" -> "ZenVox") is meaningful.
                if len(sp) >= MIN_SPOKEN_LEN and sp != e.written:
                    pairs.append((sp, e.written, e.case_sensitive))
        for sp, written, cs in sorted(pairs, key=lambda p: len(p[0]), reverse=True):
            flags = 0 if cs else re.IGNORECASE
            try:
                pat = re.compile(rf"\b{re.escape(sp)}\b", flags)
                compiled.append((pat, written))
            except re.error as e:
                log.warning(f"Dictionary pattern skipped for {sp!r}: {e}")
        return compiled

    def apply_replacements(self, text):
        if not text:
            return text
        with self._lock:
            if self._compiled is None:
                self._compiled = self._compile()
            compiled = self._compiled
        for pat, written in compiled:
            text = pat.sub(written, text)
        return text

    # ── Layer C: LLM cleaning prompt injection ────────────────────────────
    def prompt_block(self):
        terms = [e.written for e in self.entries if e.enabled and e.written]
        if not terms:
            return ""
        joined = ", ".join(terms[:MAX_HOTWORDS])
        return (
            "\n\nVOCABULARY: The following terms are intentional and correct. "
            "Spell them EXACTLY as written and never translate, expand, or "
            f"\"correct\" them: {joined}."
        )
