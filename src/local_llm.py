import os
import logging
import threading
from pathlib import Path
from typing import Optional, Iterator
from dataclasses import dataclass, field

from llama_cpp import Llama


logger = logging.getLogger(__name__)

_LOCAL_LLM_CACHE: dict[str, "LocalLLM"] = {}
_LOCAL_LLM_CACHE_LOCK = threading.Lock()


def _resolve_model_path(model_path: str) -> str:
    if os.path.isabs(model_path):
        return model_path
    return os.path.abspath(os.path.join(os.getcwd(), model_path))


def get_cached_local_llm(model_path: str, **kwargs) -> "LocalLLM":
    """Return a shared LocalLLM for the same on-disk model (avoids reload per request)."""
    resolved = _resolve_model_path(model_path)
    with _LOCAL_LLM_CACHE_LOCK:
        cached = _LOCAL_LLM_CACHE.get(resolved)
        if cached is not None:
            return cached
        llm = LocalLLM(model_path=resolved, **kwargs)
        _LOCAL_LLM_CACHE[resolved] = llm
        return llm


def clear_local_llm_cache() -> int:
    """Unload and drop all cached local models (e.g. after config hot-reload)."""
    with _LOCAL_LLM_CACHE_LOCK:
        count = len(_LOCAL_LLM_CACHE)
        for llm in _LOCAL_LLM_CACHE.values():
            llm.unload()
        _LOCAL_LLM_CACHE.clear()
        return count


@dataclass
class CompletionChunk:
    content: str
    is_final: bool = False
    tool_calls: Optional[list[dict]] = None
    finish_reason: str = ""


class LocalLLM:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_gpu_layers: int = 35,
        n_threads: int = 6,
        n_batch: int = 512,
        low_vram: bool = False,
        verbose: bool = False,
        flash_attn: bool = True,
        cache_type_k: str = "q4_0",
        cache_type_v: str = "q4_0",
        use_mmap: bool = True,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers if not low_vram else 0
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.verbose = verbose
        self.flash_attn = flash_attn
        self.cache_type_k = cache_type_k
        self.cache_type_v = cache_type_v
        self.use_mmap = use_mmap
        self._llm: Optional[Llama] = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._llm is None:
            with self._lock:
                if self._llm is None:
                    resolved_path = self.model_path
                    if not os.path.isabs(resolved_path):
                        resolved_path = os.path.join(os.getcwd(), self.model_path)

                    if not os.path.exists(resolved_path):
                        raise FileNotFoundError(f"Model not found: {resolved_path}")

                    logger.info(f"Loading model from {resolved_path}")
                    logger.info(
                        f"Settings: n_ctx={self.n_ctx}, n_gpu_layers={self.n_gpu_layers}, "
                        f"flash_attn={self.flash_attn}, cache_type_k={self.cache_type_k}"
                    )

                    kwargs = dict(
                        model_path=resolved_path,
                        n_ctx=self.n_ctx,
                        n_gpu_layers=self.n_gpu_layers,
                        n_threads=self.n_threads,
                        n_batch=self.n_batch,
                        verbose=self.verbose,
                        chat_format="qwen",
                        use_mmap=self.use_mmap,
                    )

                    if self.flash_attn:
                        kwargs["flash_attn"] = True

                    if self.cache_type_k and self.cache_type_v:
                        kwargs["cache_type_k"] = self.cache_type_k
                        kwargs["cache_type_v"] = self.cache_type_v

                    self._llm = Llama(**kwargs)
                    logger.info("Model loaded successfully")

    def complete(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
        stream: bool = False,
    ) -> CompletionChunk | Iterator[CompletionChunk]:
        self._ensure_loaded()

        params = dict(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
            stream=stream,
        )

        if stream:

            def generator():
                for output in self._llm(**params):
                    chunk = output["choices"][0]["text"]
                    yield CompletionChunk(content=chunk)
                yield CompletionChunk(content="", is_final=True)

            return generator()
        else:
            output = self._llm(**params)
            content = output["choices"][0]["text"]
            return CompletionChunk(content=content, is_final=True)

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
        stream: bool = False,
    ) -> CompletionChunk | Iterator[CompletionChunk]:
        self._ensure_loaded()

        formatted_messages = self._format_messages(messages)

        params = dict(
            messages=formatted_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
            stream=stream,
        )

        if tools:
            params["tools"] = tools

        if stream:

            def generator():
                for output in self._llm.create_chat_completion(**params):
                    delta = output["choices"][0]["delta"]
                    if "content" in delta and delta["content"]:
                        yield CompletionChunk(content=delta["content"])
                    elif "tool_calls" in delta:
                        yield CompletionChunk(
                            content="", tool_calls=delta["tool_calls"]
                        )
                yield CompletionChunk(content="", is_final=True)

            return generator()
        else:
            output = self._llm.create_chat_completion(**params)
            message = output["choices"][0]["message"]
            content = message.get("content", "")
            tool_calls = message.get("tool_calls")
            return CompletionChunk(
                content=content or "", is_final=True, tool_calls=tool_calls
            )

    def _format_messages(self, messages: list[dict]) -> list[dict]:
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                role = "system"
            elif role == "assistant":
                role = "assistant"
            else:
                role = "user"
            formatted.append(
                {
                    "role": role,
                    "content": msg.get("content", ""),
                }
            )
        return formatted

    def unload(self):
        with self._lock:
            if self._llm is not None:
                del self._llm
                self._llm = None
                logger.info("Model unloaded")

    def get_context_window(self) -> int:
        return self.n_ctx

    def is_loaded(self) -> bool:
        return self._llm is not None
