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
    GOOGLE_AI = "google_ai"


class TaskType(Enum):
    CODING = "coding"
    GENERAL = "general"
    FAST = "fast"


@dataclass
class Response:
    content: str
    provider: Provider
    model: str
    usage: Optional[dict] = None
    finish_reason: Optional[str] = None
    tool_calls: Optional[list[dict]] = None


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
        self._db_models: list[dict] = []
        self.load_db_configs()

    def load_db_configs(self):
        try:
            from .database import db
            cloud_models = db.list_model_configs(provider_type=None, is_active=True)
            if cloud_models:
                self._db_models = cloud_models
                logger.info(f"Loaded {len(cloud_models)} active model configs from DB")
        except Exception as e:
            logger.debug(f"DB model configs not available yet: {e}")

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
                n_ctx=model_data.get("n_ctx", 131072),
                n_gpu_layers=model_data.get("n_gpu_layers", -1),
                n_threads=model_data.get("n_threads", 6),
                n_batch=model_data.get("n_batch", 512),
                low_vram=model_data.get("low_vram", False),
                flash_attn=model_data.get("flash_attn", True),
                cache_type_k=model_data.get("cache_type_k", "q4_0"),
                cache_type_v=model_data.get("cache_type_v", "q4_0"),
                use_mmap=model_data.get("use_mmap", True),
            )
        return self._local_llm

    def _get_google_ai_client(self) -> GeminiClient:
        if self._gemini_client is None:
            self._gemini_client = GeminiClient(self.config.models.cloud.google_ai)
        return self._gemini_client

    def _estimate_tokens(self, messages: list[dict]) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            total += len(content) // 4
        return total

    def _classify_task(self, messages: list[dict]) -> TaskType:
        combined = " ".join(msg.get("content", "") for msg in messages).lower()

        coding_keywords = [
            "code",
            "function",
            "class",
            "python",
            "javascript",
            "bug",
            "debug",
            "refactor",
            "optimize",
            "sql",
            "query",
            "api",
            "algorithm",
            "test",
            "import",
            "compile",
            "script",
            "programming",
            "implement",
            "syntax",
        ]
        fast_keywords = [
            "quick",
            "brief",
            "simple",
            "short",
            "one line",
            "summary",
            "what is",
        ]

        coding_score = sum(1 for kw in coding_keywords if kw in combined)
        fast_score = sum(1 for kw in fast_keywords if kw in combined)

        if coding_score >= 2:
            return TaskType.CODING
        elif fast_score >= 1 and len(combined) < 100:
            return TaskType.FAST
        return TaskType.GENERAL

    def _select_model_for_task(self, task: TaskType) -> str:
        models = self.config.models.local.model_dump(exclude_none=True)

        for key, data in models.items():
            if isinstance(data, dict) and data.get("role") == task.value:
                return key.replace("_", "-")

        return self._default_local_model

    def _should_use_cloud(self, messages: list[dict]) -> bool:
        if self._mode == "local_only":
            return False
        if self._mode == "cloud_only" or self._mode == "cloud_first":
            return True

        token_count = self._estimate_tokens(messages)
        return token_count > self.config.routing.auto_threshold_tokens

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        provider: Optional[Provider] = None,
        task: Optional[TaskType] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stream: bool = False,
        model: Optional[str] = None,
    ) -> Response | StreamResponse:
        if provider is None:
            if self._should_use_cloud(messages):
                provider = Provider.GOOGLE_AI
            else:
                provider = Provider.LOCAL

        logger.info(
            f"[ROUTER] provider={provider.value} model={model or 'default'} "
            f"stream={stream} messages={len(messages)}"
        )

        if provider == Provider.LOCAL and task is None:
            task = self._classify_task(messages)
            logger.info(f"Task classified as: {task.value}")

            selected_model = self._select_model_for_task(task)
            if selected_model != self._default_local_model:
                logger.info(f"Routing to model: {selected_model}")
                self._default_local_model = selected_model

        try:
            if provider == Provider.LOCAL:
                return self._chat_local(
                    messages, tools, max_tokens, temperature, stream
                )
            else:
                return self._chat_google_ai(
                    messages, tools, max_tokens, temperature, stream, model
                )
        except Exception as e:
            logger.error(f"Error with {provider.value}: {e}")

            if provider == Provider.LOCAL and self._mode != "local_only":
                logger.info("Falling back to Google AI")
                return self._chat_google_ai(
                    messages, tools, max_tokens, temperature, stream, model
                )
            elif provider == Provider.GOOGLE_AI and self._mode != "cloud_only":
                logger.info("Falling back to local model")
                return self._chat_local(
                    messages, tools, max_tokens, temperature, stream
                )

            raise

    def _chat_local(
        self,
        messages: list[dict],
        tools: Optional[list[dict]],
        max_tokens: int,
        temperature: float,
        stream: bool,
    ) -> Response | StreamResponse:
        llm = self._get_local_llm()
        result = llm.chat(
            messages=messages,
            tools=tools,
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
            tool_calls=result.tool_calls,
        )

    def _chat_google_ai(
        self,
        messages: list[dict],
        tools: Optional[list[dict]],
        max_tokens: int,
        temperature: float,
        stream: bool,
        model: Optional[str] = None,
    ) -> Response | StreamResponse:
        client = self._get_google_ai_client()
        target_model = model or client.model
        logger.info(f"[ROUTER] → Google AI: {target_model} (stream={stream})")
        import time
        t0 = time.monotonic()

        result = client.chat(
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            model=model,
        )

        elapsed = (time.monotonic() - t0) * 1000

        if stream:
            logger.info(f"[ROUTER] ← Google AI: {target_model} stream started ({elapsed:.0f}ms connect)")
            return StreamResponse(
                chunks=result,
                provider=Provider.GOOGLE_AI,
                model=client.model,
            )

        usage = result.usage or {}
        tokens = usage.get("totalTokenCount", usage.get("total_tokens", "?"))
        content_len = len(result.content) if result.content else 0
        logger.info(
            f"[ROUTER] ← Google AI: {target_model} | {elapsed:.0f}ms "
            f"tokens={tokens} finish={result.finish_reason} content={content_len}c"
        )
        return Response(
            content=result.content,
            provider=Provider.GOOGLE_AI,
            model=client.model,
            usage=result.usage,
            finish_reason=result.finish_reason,
            tool_calls=result.tool_calls,
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
        if mode in ("local_only", "cloud_only", "cloud_first", "auto"):
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
            "google_ai_model": self.config.models.cloud.google_ai.model,
            "google_ai_configured": bool(self.config.models.cloud.google_ai.api_key),
        }


def create_router(config: Optional[Config] = None) -> Router:
    return Router(config)
