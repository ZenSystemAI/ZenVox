"""
providers.py — Multi-provider LLM cleaning backends
Supports: Gemini, OpenAI, Anthropic, Groq, Ollama
"""
import inspect
import json
import logging
from urllib import error as urllib_error
from urllib import request as urllib_request

log = logging.getLogger("zenvox")
DEFAULT_REQUEST_TIMEOUT = 15.0

# Provider name → (default model, label)
PROVIDERS = {
    "Gemini":    {"default_model": "gemini-3.1-flash-lite-preview", "needs_key": True},
    "OpenAI":    {"default_model": "gpt-4o-mini", "needs_key": True},
    "Anthropic": {"default_model": "claude-haiku-4-5-20251001", "needs_key": True},
    "Groq":      {"default_model": "llama-3.3-70b-versatile", "needs_key": True},
    "Ollama":    {"default_model": "llama3.2:3b", "needs_key": False},
}

PROVIDER_NAMES = list(PROVIDERS.keys())


class CleaningProvider:
    """Base interface for text cleaning via LLM."""

    def __init__(self, api_key, model_name, system_prompt, timeout=DEFAULT_REQUEST_TIMEOUT):
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.timeout = timeout

    def clean(self, raw_text, max_tokens=4096):
        raise NotImplementedError


def _supports_parameter(callable_obj, name):
    try:
        return name in inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False


def _build_client_kwargs(client_cls, api_key, timeout, max_retries=0):
    kwargs = {"api_key": api_key}
    if _supports_parameter(client_cls, "timeout"):
        kwargs["timeout"] = timeout
    if _supports_parameter(client_cls, "max_retries"):
        kwargs["max_retries"] = max_retries
    return kwargs


class GeminiProvider(CleaningProvider):
    def __init__(self, api_key, model_name, system_prompt, timeout=DEFAULT_REQUEST_TIMEOUT):
        super().__init__(api_key, model_name, system_prompt, timeout=timeout)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            kwargs = {"api_key": self.api_key}
            if _supports_parameter(genai.Client, "http_options"):
                kwargs["http_options"] = {"timeout": int(self.timeout * 1000)}
            self._client = genai.Client(**kwargs)
        return self._client

    def clean(self, raw_text, max_tokens=4096):
        client = self._get_client()
        r = client.models.generate_content(
            model=self.model_name,
            contents=f"[RAW] {raw_text} [/RAW]",
            config={
                "system_instruction": self.system_prompt,
                "temperature": 0.2,
                "max_output_tokens": max_tokens,
            })
        return r.text.strip()


class OpenAIProvider(CleaningProvider):
    def __init__(self, api_key, model_name, system_prompt, timeout=DEFAULT_REQUEST_TIMEOUT):
        super().__init__(api_key, model_name, system_prompt, timeout=timeout)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(**_build_client_kwargs(OpenAI, self.api_key, self.timeout))
        return self._client

    def clean(self, raw_text, max_tokens=4096):
        client = self._get_client()
        r = client.chat.completions.create(
            model=self.model_name,
            temperature=0.2,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"[RAW] {raw_text} [/RAW]"},
            ])
        return r.choices[0].message.content.strip()


class AnthropicProvider(CleaningProvider):
    def __init__(self, api_key, model_name, system_prompt, timeout=DEFAULT_REQUEST_TIMEOUT):
        super().__init__(api_key, model_name, system_prompt, timeout=timeout)
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(
                **_build_client_kwargs(anthropic.Anthropic, self.api_key, self.timeout))
        return self._client

    def clean(self, raw_text, max_tokens=4096):
        client = self._get_client()
        r = client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=0.2,
            system=self.system_prompt,
            messages=[
                {"role": "user", "content": f"[RAW] {raw_text} [/RAW]"},
            ])
        return r.content[0].text.strip()


class GroqProvider(CleaningProvider):
    def __init__(self, api_key, model_name, system_prompt, timeout=DEFAULT_REQUEST_TIMEOUT):
        super().__init__(api_key, model_name, system_prompt, timeout=timeout)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(**_build_client_kwargs(Groq, self.api_key, self.timeout))
        return self._client

    def clean(self, raw_text, max_tokens=4096):
        client = self._get_client()
        r = client.chat.completions.create(
            model=self.model_name,
            temperature=0.2,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"[RAW] {raw_text} [/RAW]"},
            ])
        return r.choices[0].message.content.strip()


class OllamaProvider(CleaningProvider):
    def __init__(self, api_key, model_name, system_prompt,
                 endpoint="http://localhost:11434/v1", timeout=15.0):
        super().__init__(api_key, model_name, system_prompt, timeout=timeout)
        self.endpoint = endpoint.rstrip("/")

    def clean(self, raw_text, max_tokens=4096):
        payload = {
            "model": self.model_name,
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"[RAW] {raw_text} [/RAW]"},
            ],
        }
        req = urllib_request.Request(
            f"{self.endpoint}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        try:
            return body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Ollama response: {body!r}") from exc


_PROVIDER_MAP = {
    "Gemini": GeminiProvider,
    "OpenAI": OpenAIProvider,
    "Anthropic": AnthropicProvider,
    "Groq": GroqProvider,
    "Ollama": OllamaProvider,
}


def create_provider(provider_name, api_key, model_name, system_prompt,
                    endpoint=None, timeout=DEFAULT_REQUEST_TIMEOUT):
    """Factory: create the right provider instance."""
    cls = _PROVIDER_MAP.get(provider_name)
    if cls is None:
        raise ValueError(f"Unknown provider: {provider_name}")
    if provider_name == "Ollama" and endpoint:
        return cls(api_key, model_name, system_prompt, endpoint=endpoint, timeout=timeout)
    return cls(api_key, model_name, system_prompt, timeout=timeout)
