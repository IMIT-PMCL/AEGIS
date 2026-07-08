"""A generic chat-completions provider.

Works with any server that exposes a ``/chat/completions`` endpoint (many local
and hosted inference servers do). No vendor, product name, endpoint or credential
is hardcoded -- everything comes from configuration.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from .base import LLMProvider


class ChatCompletionsProvider(LLMProvider):
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 120.0,
    ):
        if not base_url or not model:
            raise ValueError("base_url and model are required.")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        # imported lazily so the package installs/imports without `requests`
        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body.update(kwargs)
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=body,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def provider_from_env() -> Optional["ChatCompletionsProvider"]:
    """Build a provider from ``AEGIS_LLM_*`` environment variables, or None."""
    base_url = os.environ.get("AEGIS_LLM_BASE_URL", "").strip()
    model = os.environ.get("AEGIS_LLM_MODEL", "").strip()
    api_key = os.environ.get("AEGIS_LLM_API_KEY", "").strip()
    if base_url and model:
        return ChatCompletionsProvider(base_url=base_url, model=model, api_key=api_key)
    return None
