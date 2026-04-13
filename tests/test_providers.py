import json
import sys
import types
import unittest
from unittest.mock import patch

from providers import OllamaProvider, OpenAIProvider


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


class OllamaProviderTests(unittest.TestCase):
    def test_clean_uses_openai_compatible_http_endpoint(self):
        seen = {}

        def fake_urlopen(request, timeout):
            seen["url"] = request.full_url
            seen["timeout"] = timeout
            seen["body"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse({
                "choices": [{"message": {"content": " cleaned text "}}],
            })

        provider = OllamaProvider(
            api_key="",
            model_name="llama3.2:3b",
            system_prompt="Clean the transcription.",
            endpoint="http://localhost:11434/v1/",
            timeout=9.5,
        )

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = provider.clean("raw text", max_tokens=321)

        self.assertEqual(result, "cleaned text")
        self.assertEqual(seen["url"], "http://localhost:11434/v1/chat/completions")
        self.assertEqual(seen["timeout"], 9.5)
        self.assertEqual(seen["body"]["model"], "llama3.2:3b")
        self.assertEqual(seen["body"]["max_tokens"], 321)
        self.assertEqual(
            seen["body"]["messages"][1]["content"],
            "[RAW] raw text [/RAW]",
        )


class OpenAIProviderTests(unittest.TestCase):
    def test_client_uses_timeout_and_disables_retries_when_supported(self):
        seen = {}

        class FakeCompletions:
            def create(self, **kwargs):
                seen["request"] = kwargs
                message = types.SimpleNamespace(content=" cleaned ")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        class FakeOpenAI:
            def __init__(self, api_key, timeout=None, max_retries=None):
                seen["client_kwargs"] = {
                    "api_key": api_key,
                    "timeout": timeout,
                    "max_retries": max_retries,
                }
                self.chat = types.SimpleNamespace(completions=FakeCompletions())

        fake_module = types.ModuleType("openai")
        fake_module.OpenAI = FakeOpenAI

        provider = OpenAIProvider(
            api_key="openai-key",
            model_name="gpt-4o-mini",
            system_prompt="Clean it.",
            timeout=7.25,
        )

        with patch.dict(sys.modules, {"openai": fake_module}, clear=False):
            result = provider.clean("raw text", max_tokens=111)

        self.assertEqual(result, "cleaned")
        self.assertEqual(
            seen["client_kwargs"],
            {"api_key": "openai-key", "timeout": 7.25, "max_retries": 0},
        )
        self.assertEqual(seen["request"]["model"], "gpt-4o-mini")
        self.assertEqual(seen["request"]["max_tokens"], 111)
        self.assertEqual(
            seen["request"]["messages"][1]["content"],
            "[RAW] raw text [/RAW]",
        )


if __name__ == "__main__":
    unittest.main()
