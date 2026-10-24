"""
Generic OpenAI-compatible API client for LLM inference.

Works with any provider that exposes /v1/chat/completions:
  - OpenRouter (openrouter.ai)
  - Together AI (together.xyz)
  - Fireworks AI (fireworks.ai)
  - Any vLLM / TGI deployment

The returned call_fn matches the Agent interface:
    call_fn(system_prompt: str, user_prompt: str) -> Dict[str, Any]
"""

from __future__ import annotations
import json
import requests
import time
from typing import Dict, Any, Optional


PROVIDER_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "together": "https://api.together.xyz/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "groq": "https://api.groq.com/openai/v1",
}

PROVIDER_RATE_DELAY = {
    "openrouter": 0.5,
    "together": 0.3,
    "fireworks": 0.3,
    "groq": 3.5,
}


class APILLMClient:
    """Client for any OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        timeout: int = 120,
        max_retries: int = 3,
        extra_headers: Optional[Dict[str, str]] = None,
        rate_delay: float = 0.5,
    ):
        self.base_url = base_url.rstrip("/")
        self.chat_url = f"{self.base_url}/chat/completions"
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_headers = extra_headers or {}
        self.rate_delay = rate_delay

    def _headers(self) -> Dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        h.update(self.extra_headers)
        return h

    def __call__(
        self, system_prompt: str, user_prompt: str
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if "openrouter.ai" in self.base_url:
            payload["reasoning"] = {"effort": "none"}

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.rate_delay)
                resp = requests.post(
                    self.chat_url,
                    json=payload,
                    timeout=self.timeout,
                    headers=self._headers(),
                )
                resp.raise_for_status()
                data = resp.json()

                msg = data["choices"][0]["message"]
                text = msg.get("content") or ""
                if not text and msg.get("reasoning"):
                    text = str(msg["reasoning"])
                text = text.strip()
                if not text:
                    return self._fallback()
                return self._parse_json(text)

            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait = (2 ** attempt) + 0.5
                    print(f"  API retry {attempt+1}/{self.max_retries}: {e}")
                    time.sleep(wait)

            except requests.exceptions.HTTPError as e:
                last_exception = e
                status = e.response.status_code if e.response is not None else 0
                if status in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                    wait = (2 ** attempt) + 0.5
                    print(f"  API HTTP {status}, retry {attempt+1}: {e}")
                    time.sleep(wait)
                else:
                    body = ""
                    try:
                        body = e.response.text[:200]
                    except Exception:
                        pass
                    print(f"  API request failed ({status}): {body}")
                    return self._fallback()

            except Exception as e:
                print(f"  API request failed: {e}")
                return self._fallback()

        print(f"  API exhausted retries: {last_exception}")
        return self._fallback()

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        cleaned = text.strip()

        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(cleaned[start:end])
                except json.JSONDecodeError:
                    pass
            return {"choice": "COOPERATE", "reasoning": cleaned}

    @staticmethod
    def _fallback() -> Dict[str, Any]:
        return {"choice": "COOPERATE", "reasoning": "[LLM call failed]"}


def create_api_call_fn(
    provider: str,
    model: str,
    api_key: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
    base_url: Optional[str] = None,
) -> APILLMClient:
    """
    Factory: create a call_fn backed by an external API provider.

    Args:
        provider: "openrouter", "together", "fireworks", or "custom"
        model: Provider-specific model ID (e.g. "qwen/qwen3-8b")
        api_key: API key for the provider
        temperature: Sampling temperature
        max_tokens: Max tokens per response
        base_url: Override the provider's default base URL
    """
    url = base_url or PROVIDER_BASE_URLS.get(provider)
    if not url:
        raise ValueError(
            f"Unknown provider '{provider}'. Use one of: "
            f"{list(PROVIDER_BASE_URLS.keys())} or pass base_url."
        )

    extra_headers = {}
    if provider == "openrouter":
        extra_headers["HTTP-Referer"] = "https://github.com/Jeff9497/collusion-peer-swap"
        extra_headers["X-Title"] = "Collusion Peer Swap Experiment"

    rate_delay = PROVIDER_RATE_DELAY.get(provider, 0.5)

    return APILLMClient(
        base_url=url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_headers=extra_headers,
        rate_delay=rate_delay,
    )
