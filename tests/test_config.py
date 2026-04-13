import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class FakeKeyring(types.ModuleType):
    def __init__(self):
        super().__init__("keyring")
        self._store = {}

    def get_password(self, service, username):
        return self._store.get((service, username))

    def set_password(self, service, username, password):
        self._store[(service, username)] = password

    def delete_password(self, service, username):
        self._store.pop((service, username), None)


class SettingsPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.keyring = FakeKeyring()
        fake_sounddevice = types.ModuleType("sounddevice")
        fake_sounddevice.query_devices = lambda *args, **kwargs: []

        fake_pil = types.ModuleType("PIL")
        fake_image = types.ModuleType("PIL.Image")
        fake_draw = types.ModuleType("PIL.ImageDraw")

        class DummyDrawer:
            def ellipse(self, *args, **kwargs):
                return None

        fake_image.new = lambda *args, **kwargs: object()
        fake_draw.Draw = lambda image: DummyDrawer()
        fake_pil.Image = fake_image
        fake_pil.ImageDraw = fake_draw

        self.module_patcher = patch.dict(
            sys.modules,
            {
                "keyring": self.keyring,
                "sounddevice": fake_sounddevice,
                "PIL": fake_pil,
                "PIL.Image": fake_image,
                "PIL.ImageDraw": fake_draw,
            },
            clear=False,
        )
        self.module_patcher.start()
        self.addCleanup(self.module_patcher.stop)
        sys.modules.pop("config", None)
        self.config = importlib.import_module("config")
        self.config = importlib.reload(self.config)
        self.addCleanup(sys.modules.pop, "config", None)

    def test_empty_api_key_removes_keyring_entry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            self.config.SETTINGS_FILE = settings_path

            settings = self.config.Settings(clean_provider="OpenAI")
            settings.openai_api_key = "secret-value"
            settings.save()

            self.assertEqual(
                self.keyring.get_password(self.config.KEYRING_SERVICE, "openai"),
                "secret-value",
            )
            self.assertEqual(
                json.loads(settings_path.read_text(encoding="utf-8"))["openai_api_key"],
                "",
            )

            settings.openai_api_key = ""
            settings.save()

            self.assertIsNone(
                self.keyring.get_password(self.config.KEYRING_SERVICE, "openai")
            )
            self.assertEqual(
                json.loads(settings_path.read_text(encoding="utf-8"))["openai_api_key"],
                "",
            )

            loaded = self.config.Settings.load()
            self.assertEqual(loaded.openai_api_key, "")


if __name__ == "__main__":
    unittest.main()
