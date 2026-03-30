"""
providers.py — Multi-provider LLM cleaning backends
Supports: Gemini, OpenAI, Anthropic, Groq, Ollama
"""
import logging

log = logging.getLogger("whisper")

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

    def __init__(self, api_key, model_name, system_prompt):
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt

    def clean(self, raw_text, max_tokens=4096):
        raise NotImplementedError


class GeminiProvider(CleaningProvider):
    def __init__(self, api_key, model_name, system_prompt):
        super().__init__(api_key, model_name, system_prompt)
        self._model = None

    def _get_model(self):
        if self._model is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=self.system_prompt)
        return self._model

    def clean(self, raw_text, max_tokens=4096):
        model = self._get_model()
        r = model.generate_content(
            f"[RAW] {raw_text} [/RAW]",
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": max_tokens,
            })
        return r.text.strip()


class OpenAIProvider(CleaningProvider):
    def __init__(self, api_key, model_name, system_prompt):
        super().__init__(api_key, model_name, system_prompt)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
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
    def __init__(self, api_key, model_name, system_prompt):
        super().__init__(api_key, model_name, system_prompt)
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
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
    def __init__(self, api_key, model_name, system_prompt):
        super().__init__(api_key, model_name, system_prompt)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
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
    def __init__(self, api_key, model_name, system_prompt):
        super().__init__(api_key, model_name, system_prompt)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama")
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


_PROVIDER_MAP = {
    "Gemini": GeminiProvider,
    "OpenAI": OpenAIProvider,
    "Anthropic": AnthropicProvider,
    "Groq": GroqProvider,
    "Ollama": OllamaProvider,
}


def create_provider(provider_name, api_key, model_name, system_prompt):
    """Factory: create the right provider instance."""
    cls = _PROVIDER_MAP.get(provider_name)
    if cls is None:
        raise ValueError(f"Unknown provider: {provider_name}")
    return cls(api_key, model_name, system_prompt)
