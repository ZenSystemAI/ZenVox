"""
test_engine_reliability.py — regressions for the recording-reliability fixes.

These guard the failure modes pulled from a real zenvox.log:
  * float32-index resample overflow → IndexError → dead hotkey listener
  * resample correctness / edge cases
"""
import unittest

import numpy as np

from config import Settings
from zenvox import ZenVoxEngine


class RawModeTests(unittest.TestCase):
    def test_raw_preset_skips_llm(self):
        """Raw mode must return the exact transcribed text with no provider call."""
        eng = ZenVoxEngine(Settings(cleaning_preset="Raw"))
        result = eng.clean_text("um so this is the exact words")
        self.assertEqual(result.text, "um so this is the exact words")
        self.assertFalse(result.used_fallback)
        self.assertIsNone(eng._get_cleaning_provider())  # never builds a provider


class ResampleTests(unittest.TestCase):
    def test_long_clip_does_not_overflow(self):
        """~9 min at 48kHz (25.8M samples) once crashed with IndexError because
        float32 indices rounded past len(audio)-1. Must resample cleanly now."""
        n = 25_803_264  # the exact length from the logged crash
        audio = (np.sin(np.linspace(0, 5000, n)) * 0.2).astype(np.float32)
        out = ZenVoxEngine._resample_linear(audio, 48000, 16000)
        self.assertEqual(out.dtype, np.float32)
        self.assertAlmostEqual(len(out), int(round(n * 16000 / 48000)), delta=1)

    def test_old_float32_index_would_have_overflowed(self):
        """Document the root cause: the old approach produced an index == len."""
        n = 25_803_264
        indices = np.linspace(0, n - 1, int(n * 16000 / 48000)).astype(np.float32)
        idx_floor = np.floor(indices).astype(int)
        self.assertGreaterEqual(int(idx_floor.max()), n)  # out of bounds

    def test_amplitude_preserved(self):
        sr, freq = 48000, 1000
        x = (np.sin(2 * np.pi * freq * np.arange(int(sr * 0.5)) / sr) * 0.3).astype(np.float32)
        y = ZenVoxEngine._resample_linear(x, sr, 16000)
        self.assertEqual(len(y), 8000)
        self.assertAlmostEqual(float(np.abs(y).max()), 0.3, delta=0.02)

    def test_edge_cases(self):
        self.assertEqual(ZenVoxEngine._resample_linear(np.array([], np.float32), 48000, 16000).size, 0)
        single = ZenVoxEngine._resample_linear(np.array([0.5], np.float32), 48000, 16000)
        self.assertEqual(single.tolist(), [0.5])
        same = ZenVoxEngine._resample_linear(np.ones(1000, np.float32), 16000, 16000)
        self.assertEqual(len(same), 1000)


if __name__ == "__main__":
    unittest.main()
