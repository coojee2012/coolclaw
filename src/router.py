import os
import logging
from typing import Optional, Iterator
from dataclasses import dataclass
from enum import Enum

from .config import Config, get_config
from .local_llm import LocalLLM, CompletionChunk
from .gemini_client import GeminiClient, GeminiChunk


logger = logging.getLogger(__name__)


class Provider(Enum):
    LOCAL = "local"
    GEMINI = "gemini"


@dataclass
class Response:
    content: str
    provider: Provider
    model: str
    usage: Optional[dict] = None
    finish_reason: Optional[str] = None


@dataclass
class StreamResponse:
    chunks: Iterator[CompletionChunk | GeminiChunk]
    provider: Provider
    model: str


class Router:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self._local_llm: Optional[LocalLLM] = None
        self._gemini_client: Optional[GeminiClient] = None
        self._mode = self.config.routing.mode
        self._default_local_model = self.config.models.local.default
        self._default_cloud_model = self.config.models.cloud.default

    def _get_local_llm(self) -> LocalLLM:
        if self._local_llm is None:
            model_key = self._default_local_model
            resolved_path = self.config.resolve_model_path(model_key)

            local_config = self.config.models.local
            model_data = local_config.model_dump(exclude_none=True).get(
                model_key.replace("-", "_"), {}
            )

            self._local_llm = LocalLLM(
                model_path=resolved_path,
                n_ctx=model_data.get("n_ctx", 8192),
                n_gpu_layers=model_data.get("n_gpu_layers", 35),
                n_threads=model_data.get("n_threads", 6),
                n_batch=model_data.get("n_batch", 512),
                low_vram=model_data.get("low_vram", False),
            )
        return self._local_llm

    def _get_gemini_client(self) -> GeminiClient:
        if self._gemini_client is None:
            self._gemini_client = GeminiClient(self.config.models.cloud.gemini)
        return self._gemini_client

    def _estimate_tokens(self, messages: list[dict]) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            total += len(content) // 4
        return total

    def _should_use_cloud(self, messages: list[dict]) -> bool:
        if self._mode == "local_only":
            return False
        if self._mode == "cloud_only":
            return True

        token_count = self._estimate_tokens(messages)
        return token_count > self.config.routing.auto_threshold_tokens

    def chat(
        self,
        messages: list[dict],
        provider: Optional[Provider] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Response | StreamResponse:
        if provider is None:
            provider = (
                Provider.LOCAL
                if not self._should_use_cloud(messages)
                else Provider.GEMINI
            )

        try:
            if provider == Provider.LOCAL:
                return self._chat_local(messages, max_tokens, temperature, stream)
            else:
                return self._chat_gemini(messages, max_tokens, temperature, stream)
        except Exception as e:
            logger.error(f"Error with {provider.value}: {e}")

            if provider == Provider.LOCAL and self._mode != "local_only":
                logger.info("Falling back to Gemini")
                return self._chat_gemini(messages, max_tokens, temperature, stream)
            elif provider == Provider.GEMINI and self._mode != "cloud_only":
                logger.info("Falling back to local model")
                return self._chat_local(messages, max_tokens, temperature, stream)

            raise

    def _chat_local(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        stream: bool,
    ) -> Response | StreamResponse:
        llm = self._get_local_llm()
        result = llm.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )

        if stream:
            return StreamResponse(
                chunks=result,
                provider=Provider.LOCAL,
                model=self._default_local_model,
            )

        return Response(
            content=result.content,
            provider=Provider.LOCAL,
            model=self._default_local_model,
        )

    def _chat_gemini(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        stream: bool,
    ) -> Response | StreamResponse:
        client = self._get_gemini_client()
        result = client.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )

        if stream:
            return StreamResponse(
                chunks=result,
                provider=Provider.GEMINI,
                model=client.model,
            )

        return Response(
            content=result.content,
            provider=Provider.GEMINI,
            model=client.model,
            usage=result.usage,
            finish_reason=result.finish_reason,
        )

    def complete(
        self,
        prompt: str,
        provider: Optional[Provider] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Response | StreamResponse:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(
            messages=messages,
            provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )

    def set_mode(self, mode: str):
        if mode in ("local_only", "cloud_only", "auto"):
            self._mode = mode
            logger.info(f"Router mode set to: {mode}")
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def unload_local_model(self):
        if self._local_llm:
            self._local_llm.unload()
            self._local_llm = None

    def is_local_model_loaded(self) -> bool:
        return self._local_llm is not None and self._local_llm.is_loaded()

    def get_status(self) -> dict:
        return {
            "mode": self._mode,
            "local_model": self._default_local_model,
            "local_loaded": self.is_local_model_loaded(),
            "gemini_configured": bool(self.config.models.cloud.gemini.api_key),
        }


def create_router(config: Optional[Config] = None) -> Router:
    return Router(config)
