"""
test_features.py — regressions for the Wispr-parity features and review fixes:
snippets, per-app preset rules, dictionary kinds/import-export, stats,
language-flip guard, terminal newline collapse, hotkey parse hardening.
"""
import json
import tempfile
import unittest
from pathlib import Path

from config import Settings
from dictionary import Dictionary, DictionaryEntry, KIND_SNIPPET
from providers import translation_detected


class SnippetTests(unittest.TestCase):
    def _dict(self, entries):
        d = Dictionary(entries=entries, path=Path(tempfile.mktemp(suffix=".json")))
        return d

    def test_snippet_expands_after_clean(self):
        d = self._dict([DictionaryEntry(written="Best regards,\nSteven",
                                        spoken=["sign off"], kind=KIND_SNIPPET)])
        self.assertEqual(d.apply_snippets("Thanks. sign off"),
                         "Thanks. Best regards,\nSteven")

    def test_snippet_excluded_from_term_layers(self):
        d = self._dict([DictionaryEntry(written="my email address",
                                        spoken=["myemail"], kind=KIND_SNIPPET)])
        # Snippets must NOT feed hotwords bias, replace, or the prompt lock.
        self.assertEqual(d.hotwords_string(), "")
        self.assertEqual(d.prompt_block(), "")
        self.assertEqual(d.apply_replacements("myemail"), "myemail")

    def test_disabled_snippet_ignored(self):
        d = self._dict([DictionaryEntry(written="X", spoken=["trig"],
                                        kind=KIND_SNIPPET, enabled=False)])
        self.assertEqual(d.apply_snippets("say trig now"), "say trig now")


class DictionaryKindTests(unittest.TestCase):
    def test_roundtrip_kind(self):
        e = DictionaryEntry.from_dict({"written": "A", "kind": "snippet"})
        self.assertTrue(e.is_snippet)
        e2 = DictionaryEntry.from_dict({"written": "B"})
        self.assertFalse(e2.is_snippet)

    def test_bad_kind_falls_back_to_term(self):
        e = DictionaryEntry.from_dict({"written": "A", "kind": "garbage"})
        self.assertEqual(e.kind, "term")

    def test_import_export_roundtrip(self):
        path = Path(tempfile.mktemp(suffix=".json"))
        src = Dictionary(entries=[DictionaryEntry(written="ZenVox", spoken=["zen vox"])],
                         path=path)
        export_path = tempfile.mktemp(suffix=".json")
        src.export_json(export_path)
        dst = Dictionary(entries=[], path=Path(tempfile.mktemp(suffix=".json")))
        n = dst.import_json(export_path)
        self.assertEqual(n, 1)
        self.assertEqual(dst.entries[0].written, "ZenVox")

    def test_import_merges_by_written(self):
        path = Path(tempfile.mktemp(suffix=".json"))
        d = Dictionary(entries=[DictionaryEntry(written="ZenVox", spoken=["old"])], path=path)
        imp = tempfile.mktemp(suffix=".json")
        Path(imp).write_text(json.dumps({"entries": [
            {"written": "ZenVox", "spoken": ["new"]}, {"written": "Fresh"}]}))
        d.import_json(imp)
        writtens = sorted(e.written for e in d.entries)
        self.assertEqual(writtens, ["Fresh", "ZenVox"])  # ZenVox replaced, not duplicated


class PerAppRuleTests(unittest.TestCase):
    def test_rule_match_first_wins(self):
        s = Settings(cleaning_preset="Minimal", app_rules=[
            {"pattern": "*slack*|*discord*", "preset": "General"},
            {"pattern": "*code*|*terminal*", "preset": "Technical"},
        ])
        self.assertEqual(s.preset_for_window("slack", "Team - Slack"), "General")
        self.assertEqual(s.preset_for_window("gnome-terminal-server", "bash"), "Technical")
        self.assertEqual(s.preset_for_window("firefox", "Gmail"), "Minimal")  # no match -> global

    def test_rule_matches_title_too(self):
        s = Settings(cleaning_preset="Minimal", app_rules=[
            {"pattern": "*gmail*", "preset": "Email"}])
        self.assertEqual(s.preset_for_window("firefox", "Gmail - Inbox - Firefox"), "Email")

    def test_invalid_preset_in_rule_skipped(self):
        s = Settings(cleaning_preset="Minimal", app_rules=[
            {"pattern": "*slack*", "preset": "Nonexistent"}])
        self.assertEqual(s.preset_for_window("slack", ""), "Minimal")


class TranslationGuardTests(unittest.TestCase):
    def test_fr_to_en_flip_detected(self):
        self.assertTrue(translation_detected(
            "C'est un meilleur travail que tu n'as jamais fait pour moi.",
            "It's the best work you have ever done for me."))

    def test_franglais_preserved(self):
        self.assertFalse(translation_detected(
            "j'ai besoin de checker le workflow pour voir si ca marche",
            "J'ai besoin de checker le workflow pour voir si ca marche."))

    def test_english_cleanup_not_flagged(self):
        self.assertFalse(translation_detected(
            "um so what is the best way to like deploy this thing",
            "What is the best way to deploy this thing?"))

    def test_short_text_no_false_positive(self):
        self.assertFalse(translation_detected("oui", "yes"))


class HotkeyParseTests(unittest.TestCase):
    def test_unknown_token_disables(self):
        from zenvox import ZenVoxApp
        # An unrepresentable token must return None (not a partial combo).
        self.assertIsNone(ZenVoxApp._parse_hotkey_pynput("Ctrl+Alt+F13"))

    def test_valid_combo_parses(self):
        from zenvox import ZenVoxApp
        combo = ZenVoxApp._parse_hotkey_pynput("Ctrl+Alt+F11")
        self.assertIsNotNone(combo)
        self.assertEqual(len(combo), 3)

    def test_bare_f6_parses(self):
        from zenvox import ZenVoxApp
        self.assertIsNotNone(ZenVoxApp._parse_hotkey_pynput("f6"))


class TerminalNewlineTests(unittest.TestCase):
    def test_terminal_detection_class_only(self):
        from zenvox import _is_terminal_window, ActiveWindowInfo
        # Title mentions a terminal but class is a browser -> NOT a terminal.
        self.assertFalse(_is_terminal_window(
            ActiveWindowInfo(window_class="firefox", window_name="kitty vs alacritty")))
        self.assertTrue(_is_terminal_window(
            ActiveWindowInfo(window_class="gnome-terminal-server", window_name="bash")))


class DayLabelTests(unittest.TestCase):
    """Regression: %-d is glibc-only and crashed the day label on Windows."""
    def test_no_glibc_strftime(self):
        from zenvox import ZenVoxApp
        from datetime import datetime
        # A fixed past date in the current year — must not raise and must have no zero-pad.
        d = datetime(datetime.now().year, 3, 5, 9, 0)
        label = ZenVoxApp._day_label(d)
        self.assertIn("Mar 5", label)
        self.assertNotIn("Mar 05", label)

    def test_prior_year_includes_year(self):
        from zenvox import ZenVoxApp
        from datetime import datetime
        d = datetime(2020, 3, 5, 9, 0)
        self.assertIn("2020", ZenVoxApp._day_label(d))


if __name__ == "__main__":
    unittest.main()
