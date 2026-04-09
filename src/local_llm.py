import os
import logging
import threading
from pathlib import Path
from typing import Optional, Iterator, AsyncIterator
from dataclasses import dataclass

from llama_cpp import Llama
from llama_cpp.server import app as llama_server
from llama_cpp.server.types import (
    CreateCompletionRequest,
    CreateChatCompletionRequest,
    CreateEmbeddingRequest,
)
import uvicorn


logger = logging.getLogger(__name__)


@dataclass
class CompletionChunk:
    content: str
    is_final: bool = False


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
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers if not low_vram else 0
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.verbose = verbose
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
                    self._llm = Llama(
                        model_path=resolved_path,
                        n_ctx=self.n_ctx,
                        n_gpu_layers=self.n_gpu_layers,
                        n_threads=self.n_threads,
                        n_batch=self.n_batch,
                        verbose=self.verbose,
                        chat_format="function_call",
                    )
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

        if stream:

            def generator():
                for output in self._llm.create_chat_completion(**params):
                    delta = output["choices"][0]["delta"]
                    if "content" in delta:
                        yield CompletionChunk(content=delta["content"])
                yield CompletionChunk(content="", is_final=True)

            return generator()
        else:
            output = self._llm.create_chat_completion(**params)
            content = output["choices"][0]["message"]["content"]
            return CompletionChunk(content=content, is_final=True)

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


class LocalLLMServer:
    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self.kwargs = kwargs
        self._server = None
        self._thread: Optional[threading.Thread] = None

    def start(self, host: str = "127.0.0.1", port: int = 8080):
        if self._server is not None:
            logger.warning("Server already running")
            return

        self._server = uvicorn.Server(
            uvicorn.Config(
                llama_server,
                host=host,
                port=port,
                log_level="info",
            )
        )

        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        logger.info(f"Server started at http://{host}:{port}")

    def stop(self):
        if self._server is not None:
            self._server.should_exit = True
            self._server = None
            if self._thread:
                self._thread.join(timeout=5)
            logger.info("Server stopped")
