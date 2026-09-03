from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import AsyncIterator, AsyncGenerator


@dataclass
class ProviderResponse:
    content: str
    model: str
    usage: dict = field(default_factory=dict)
    finish_reason: str = ""
    tool_calls: list[dict] | None = None


@dataclass
class ProviderStreamChunk:
    content: str = ""
    tool_call: dict | None = None
    is_final: bool = False


@dataclass
class ProviderConfig:
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    timeout: int = 60


class BaseProvider(abc.ABC):
    def __init__(self, config: ProviderConfig):
        self.config = config

    @abc.abstractmethod
    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
    ) -> ProviderResponse:
        ...

    @abc.abstractmethod
    def chat_stream(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
    ) -> AsyncGenerator[ProviderStreamChunk, None]:
        ...

    @abc.abstractmethod
    async def list_models(self) -> list[dict]:
        ...

    async def health_check(self) -> bool:
        try:
            await self.list_models()
            return True
        except Exception:
            return False
