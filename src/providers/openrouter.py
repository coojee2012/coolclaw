from __future__ import annotations

import json
import logging
import time
from typing import AsyncGenerator

import httpx

from .base import BaseProvider, ProviderConfig, ProviderResponse, ProviderStreamChunk

logger = logging.getLogger(__name__)


class OpenRouterProvider(BaseProvider):
    """OpenRouter — OpenAI-compatible API at openrouter.ai."""

    DEFAULT_BASE = "https://openrouter.ai/api/v1"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or self.DEFAULT_BASE
        self._proxy = self._resolve_proxy()

    def _resolve_proxy(self) -> str | None:
        import os
        return os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "HTTP-Referer": "https://github.com/coolclaw",
            "X-Title": "coolclaw-proxy",
            "Content-Type": "application/json",
        }

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
    ) -> ProviderResponse:
        active_model = model or self.config.model
        payload: dict = {
            "model": active_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        logger.info(
            f"[OPENROUTER] → {active_model} | messages={len(messages)} "
            f"tools={len(tools) if tools else 0} max_tokens={max_tokens}"
        )
        t0 = time.monotonic()

        async with httpx.AsyncClient(timeout=self.config.timeout, proxy=self._proxy) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()

        elapsed = (time.monotonic() - t0) * 1000
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        tool_calls = msg.get("tool_calls") or None
        usage = data.get("usage", {})
        tokens = usage.get("total_tokens", "?")
        content_len = len(msg.get("content", "") or "")

        logger.info(
            f"[OPENROUTER] ← {active_model} | {elapsed:.0f}ms "
            f"status={resp.status_code} tokens={tokens} "
            f"finish={choice.get('finish_reason', '?')} content={content_len}c"
        )

        return ProviderResponse(
            content=msg.get("content", "") or "",
            model=data.get("model", active_model),
            usage=usage,
            finish_reason=choice.get("finish_reason", ""),
            tool_calls=tool_calls,
        )

    async def chat_stream(  # type: ignore[override]
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
    ) -> AsyncGenerator[ProviderStreamChunk, None]:
        active_model = model or self.config.model
        payload: dict = {
            "model": active_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        logger.info(f"[OPENROUTER] → {active_model} | stream messages={len(messages)}")
        t0 = time.monotonic()
        chunk_count = 0

        client = httpx.AsyncClient(timeout=self.config.timeout, proxy=self._proxy)
        response = None
        try:
            response = await client.send(
                client.build_request(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._headers(),
                ),
                stream=True,
            )
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                chunk_str = line[6:]
                if chunk_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(chunk_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        chunk_count += 1
                        yield ProviderStreamChunk(content=content)
                    tool_calls = delta.get("tool_calls")
                    if tool_calls:
                        for tc in tool_calls:
                            func = tc.get("function", {})
                            args_str = func.get("arguments", "{}")
                            yield ProviderStreamChunk(
                                tool_call={
                                    "name": func.get("name", ""),
                                    "args": json.loads(args_str) if args_str else {},
                                    "id": tc.get("id", ""),
                                }
                            )
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue
        finally:
            if response:
                await response.aclose()
            await client.aclose()

        elapsed = (time.monotonic() - t0) * 1000
        logger.info(f"[OPENROUTER] ← {active_model} | stream done {elapsed:.0f}ms chunks={chunk_count}")
        yield ProviderStreamChunk(is_final=True)

    async def list_models(self) -> list[dict]:
        async with httpx.AsyncClient(timeout=self.config.timeout, proxy=self._proxy) as client:
            resp = await client.get(
                f"{self.base_url}/models", headers=self._headers()
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                {"id": m["id"], "owned_by": m.get("owned_by", "openrouter")}
                for m in data.get("data", [])
            ]
