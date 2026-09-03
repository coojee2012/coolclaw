from .base import BaseProvider, ProviderConfig, ProviderResponse, ProviderStreamChunk
from .google_ai import GoogleAIProvider
from .openrouter import OpenRouterProvider
from .openai_compat import OpenAICompatProvider

PROVIDER_MAP: dict[str, type[BaseProvider]] = {
    "google_ai": GoogleAIProvider,
    "gemma": GoogleAIProvider,
    "openrouter": OpenRouterProvider,
    "openai_compat": OpenAICompatProvider,
}


def create_provider(name: str, config: ProviderConfig) -> BaseProvider:
    cls = PROVIDER_MAP.get(name)
    if cls is None:
        raise ValueError(f"Unknown provider: {name}. Available: {list(PROVIDER_MAP)}")
    return cls(config)


__all__ = [
    "BaseProvider",
    "ProviderConfig",
    "ProviderResponse",
    "ProviderStreamChunk",
    "GoogleAIProvider",
    "OpenRouterProvider",
    "OpenAICompatProvider",
    "create_provider",
]
