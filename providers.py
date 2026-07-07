"""
providers.py — Multi-provider LLM cleaning backends
Supports: Local, Gemini, OpenAI, Anthropic, Groq, Ollama
"""
import inspect
import json
import logging
from urllib import error as urllib_error
from urllib import request as urllib_request

log = logging.getLogger("zenvox")
DEFAULT_REQUEST_TIMEOUT = 15.0

# ── Language-preservation guard ──────────────────────────────────────────────
# Small local cleaning models sometimes TRANSLATE the dictation instead of
# cleaning it (observed live: "C'est un meilleur travail…" → "It's the best
# work…") despite every preset forbidding translation. Cheap stopword-ratio
# check: if the raw text is clearly majority-French and the cleaned text is
# clearly majority-English (or vice versa), the cleaner rewrote the language —
# fall back to raw. Mixed franglais stays untouched (ratios land mid-range).
_FR_STOPWORDS = frozenset(
    "le la les un une des de du et est que qui pour dans sur avec pas ne je tu "
    "il elle on nous vous ils ce ca ça c'est mais ou où si au aux mon ma mes ton "
    "ta tes son sa ses faire fait être avoir plus tout bien besoin voir comme "
    "alors donc quand même aussi très ont sont était j'ai n'est d'un d'une".split())
_EN_STOPWORDS = frozenset(
    "the a an of and is are that which for in on with not no i you he she we "
    "they it this but or if to at my your his her its do does done be have has "
    "more all well need see like so then was were what by when even also very "
    "had been it's don't isn't i'm".split())


def _fr_ratio(text):
    words = [w.strip(".,!?;:'\"()«»").lower() for w in text.split()]
    fr = sum(1 for w in words if w in _FR_STOPWORDS)
    en = sum(1 for w in words if w in _EN_STOPWORDS)
    total = fr + en
    if total < 3:
        return None  # not enough signal to judge
    return fr / total


def translation_detected(raw_text, cleaned_text):
    """True when the cleaned text flipped language wholesale vs the raw text."""
    rs, cs = _fr_ratio(raw_text), _fr_ratio(cleaned_text)
    if rs is None or cs is None:
        return False
    return (rs >= 0.7 and cs <= 0.3) or (rs <= 0.3 and cs >= 0.7)

# Provider name → (default model, label)
PROVIDERS = {
    "Gemini":    {"default_model": "gemini-3.1-flash-lite", "needs_key": True},
    "OpenAI":    {"default_model": "gpt-4o-mini", "needs_key": True},
    "Anthropic": {"default_model": "claude-haiku-4-5-20251001", "needs_key": True},
    "Groq":      {"default_model": "llama-3.3-70b-versatile", "needs_key": True},
    "Ollama":    {"default_model": "llama3.2:3b", "needs_key": False},
    "Local":     {"default_model": "qwen3.6-moe", "needs_key": False},
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
        return (r.text or "").strip()


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
        return (r.choices[0].message.content or "").strip()


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
        return (r.content[0].text or "").strip()


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
        return (r.choices[0].message.content or "").strip()


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
    "Local": OllamaProvider,
}


def create_provider(provider_name, api_key, model_name, system_prompt,
                    endpoint=None, timeout=DEFAULT_REQUEST_TIMEOUT):
    """Factory: create the right provider instance."""
    cls = _PROVIDER_MAP.get(provider_name)
    if cls is None:
        raise ValueError(f"Unknown provider: {provider_name}")
    if provider_name in ("Ollama", "Local") and endpoint:
        return cls(api_key, model_name, system_prompt, endpoint=endpoint, timeout=timeout)
    return cls(api_key, model_name, system_prompt, timeout=timeout)
