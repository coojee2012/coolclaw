from .config import Config

try:
    from .local_llm import LocalLLM
    from .gemini_client import GeminiClient
    from .router import Router
except ImportError:
    LocalLLM = None  # type: ignore[assignment,misc]
    GeminiClient = None  # type: ignore[assignment,misc]
    Router = None  # type: ignore[assignment,misc]

__version__ = "0.1.0"

__all__ = ["Config", "LocalLLM", "GeminiClient", "Router"]
