"""Abstract language-model provider interface (vendor-neutral)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class LLMProvider(ABC):
    """Minimal interface an orchestration backend must implement.

    Implement :meth:`complete` for any model or service. Nothing in AEGIS depends
    on a particular provider; the reference orchestration runner treats this as an
    opaque plug-in.
    """

    @abstractmethod
    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        """Return the assistant text for a list of ``{"role", "content"}`` messages."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


class OfflineProvider(LLMProvider):
    """A no-network provider.

    Returns a fixed, inert response. Lets the orchestration runner execute in a
    fully offline / air-gapped environment where the analyses (which need no model)
    are all that matter.
    """

    def __init__(self, canned: Optional[str] = None):
        self._canned = canned or "[offline provider: no model configured]"

    def complete(self, messages, temperature: float = 0.0, max_tokens: int = 1024, **kwargs) -> str:
        return self._canned
