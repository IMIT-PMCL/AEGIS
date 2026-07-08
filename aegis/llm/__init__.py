"""Optional, model-agnostic language-model interface.

The scientific analyses and the evidence gate are fully deterministic and do NOT
require a language model. This subpackage exists only for the optional
orchestration layer: a thin interface so that *any* model can be plugged in.

No model vendor or product is referenced anywhere in this package. Configure a
backend entirely through environment variables or a config file:

    AEGIS_LLM_BASE_URL   e.g. http://localhost:8000/v1   (any chat-completions server)
    AEGIS_LLM_API_KEY    the key for that server (may be empty for local servers)
    AEGIS_LLM_MODEL      the model identifier the server expects

To use a different backend, implement :class:`LLMProvider` and pass it in.
"""

from .base import LLMProvider, OfflineProvider
from .chat_completions import ChatCompletionsProvider, provider_from_env

__all__ = [
    "LLMProvider",
    "OfflineProvider",
    "ChatCompletionsProvider",
    "provider_from_env",
]
