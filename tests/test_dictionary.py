"""test_dictionary.py — custom-vocabulary correction layers."""
import tempfile
import unittest
from pathlib import Path

from dictionary import Dictionary, DictionaryEntry


def _dict(*entries):
    d = Dictionary(entries=list(entries), path=Path(tempfile.gettempdir()) / "zv_test_dict.json")
    return d


class ReplacementTests(unittest.TestCase):
    def test_basic_case_insensitive_replace(self):
        d = _dict(DictionaryEntry(written="ZenVox", spoken=["zen vox", "zenbox"]))
        self.assertEqual(d.apply_replacements("i love zen vox"), "i love ZenVox")
        self.assertEqual(d.apply_replacements("Zen Vox is great"), "ZenVox is great")
        self.assertEqual(d.apply_replacements("the zenbox app"), "the ZenVox app")

    def test_word_boundary_no_overmatch(self):
        d = _dict(DictionaryEntry(written="ZenVox", spoken=["zen"]))
        # 'zenith' must NOT become 'ZenVoxith'
        self.assertEqual(d.apply_replacements("zenith zen"), "zenith ZenVox")

    def test_longest_spoken_wins(self):
        d = _dict(
            DictionaryEntry(written="New York City", spoken=["new york city"]),
            DictionaryEntry(written="York", spoken=["york"]),
        )
        # The longer multi-word form should be applied as a unit.
        self.assertEqual(d.apply_replacements("i live in new york city"), "i live in New York City")

    def test_boost_only_skips_replacement(self):
        d = _dict(DictionaryEntry(written="Kubernetes", spoken=["kubernetes"], boost_only=True))
        self.assertEqual(d.apply_replacements("we use kubernetes"), "we use kubernetes")

    def test_disabled_entry_skipped(self):
        d = _dict(DictionaryEntry(written="ZenVox", spoken=["zen vox"], enabled=False))
        self.assertEqual(d.apply_replacements("zen vox"), "zen vox")

    def test_min_length_guard(self):
        # 1-char spoken forms are ignored to avoid catastrophic over-matching.
        d = _dict(DictionaryEntry(written="X-Ray", spoken=["x"]))
        self.assertEqual(d.apply_replacements("x marks the x"), "x marks the x")

    def test_case_sensitive_entry(self):
        d = _dict(DictionaryEntry(written="iOS", spoken=["ios"], case_sensitive=True))
        self.assertEqual(d.apply_replacements("ios"), "iOS")
        self.assertEqual(d.apply_replacements("IOS"), "IOS")  # not matched (case-sensitive)

    def test_french_accents(self):
        d = _dict(DictionaryEntry(written="Lefebvre", spoken=["le fevre", "lefevre"]))
        self.assertEqual(d.apply_replacements("bonjour le fevre"), "bonjour Lefebvre")

    def test_empty_dictionary_is_noop(self):
        d = _dict()
        self.assertEqual(d.apply_replacements("anything goes"), "anything goes")
        self.assertEqual(d.hotwords_string(), "")
        self.assertEqual(d.prompt_block(), "")


class BiasAndPromptTests(unittest.TestCase):
    def test_hotwords_lists_written_forms(self):
        d = _dict(
            DictionaryEntry(written="ZenVox", spoken=["zen vox"]),
            DictionaryEntry(written="Lefebvre"),
            DictionaryEntry(written="hidden", enabled=False),
        )
        hw = d.hotwords_string()
        self.assertIn("ZenVox", hw)
        self.assertIn("Lefebvre", hw)
        self.assertNotIn("hidden", hw)

    def test_prompt_block_mentions_terms(self):
        d = _dict(DictionaryEntry(written="faster-whisper"))
        block = d.prompt_block()
        self.assertIn("faster-whisper", block)
        self.assertIn("EXACTLY", block)


class PersistenceTests(unittest.TestCase):
    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dictionary.json"
            d = Dictionary(path=path)
            d.add(DictionaryEntry(written="ZenVox", spoken=["zen vox"], boost_only=False))
            d.add(DictionaryEntry(written="Lefebvre", spoken=["lefevre"]))
            loaded = Dictionary.load(path=path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded.apply_replacements("zen vox by lefevre"), "ZenVox by Lefebvre")

    def test_add_replaces_same_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dictionary.json"
            d = Dictionary(path=path)
            d.add(DictionaryEntry(written="ZenVox", spoken=["zen vox"]))
            d.add(DictionaryEntry(written="zenvox", spoken=["zen box"]))  # same (case-insensitive)
            self.assertEqual(len(d), 1)


if __name__ == "__main__":
    unittest.main()
