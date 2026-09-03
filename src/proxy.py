from __future__ import annotations

import fnmatch
import json
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

from .config import Config, get_config
from .rate_limiter import RateLimiter, RateLimitExceeded
from .providers import create_provider, BaseProvider, ProviderConfig, ProviderResponse, ProviderStreamChunk

logger = logging.getLogger(__name__)


@dataclass
class ProxyRequest:
    model: str
    messages: list[dict]
    max_tokens: int = 2048
    temperature: float = 0.7
    stream: bool = False
    tools: list[dict] | None = None
    tool_choice: str | None = None


@dataclass
class ProxyResponse:
    id: str
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


class Proxy:
    """OpenAI-compatible proxy with per-provider rate limiting.

    Accepts requests in OpenAI format, routes to the appropriate upstream
    provider, and applies rate limiting to stay within free-tier limits.
    Supports automatic fallback when a provider returns 429.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self._providers: dict[str, BaseProvider] = {}
        self._rate_limiter = RateLimiter()
        self._fallback_order: list[str] = []
        self._setup()

    def _setup(self):
        proxy_cfg = self.config.proxy
        if not proxy_cfg.enabled:
            logger.info("Proxy mode disabled")
            return

        for name, ep_cfg in proxy_cfg.providers.items():
            if not ep_cfg.api_key and name != "openai_compat":
                env_key = {
                    "google_ai": "GOOGLE_AI_API_KEY",
                    "openrouter": "OPENROUTER_API_KEY",
                }.get(name, "")
                import os
                ep_cfg.api_key = os.environ.get(env_key, ep_cfg.api_key)

            if ep_cfg.api_key or name == "openai_compat":
                provider_config = ProviderConfig(
                    api_key=ep_cfg.api_key,
                    base_url=ep_cfg.base_url,
                    model=ep_cfg.model,
                    timeout=ep_cfg.timeout,
                )
                try:
                    self._providers[name] = create_provider(name, provider_config)
                except Exception as e:
                    logger.error(f"Failed to create provider '{name}': {e}")
                    continue

            rl_cfg = proxy_cfg.rate_limits.get(name)
            if rl_cfg:
                self._rate_limiter.add_provider(
                    name,
                    rpm=rl_cfg.rpm,
                    burst=rl_cfg.burst,
                    queue_size=rl_cfg.queue_size,
                    timeout=rl_cfg.timeout,
                    rpd=rl_cfg.rpd,
                    cooldown_seconds=rl_cfg.cooldown_seconds,
                )

        logger.info(
            f"Proxy initialized: {list(self._providers.keys())} providers, "
            f"model routes: {proxy_cfg.model_routes}"
        )
        if proxy_cfg.fallback_order:
            self._fallback_order = [
                n for n in proxy_cfg.fallback_order if n in self._providers
            ]
        else:
            self._fallback_order = list(self._providers.keys())

    def _resolve_provider(self, model: str) -> str:
        proxy_cfg = self.config.proxy
        model_lower = model.lower()

        for pattern, provider_name in proxy_cfg.model_routes.items():
            if fnmatch.fnmatch(model_lower, pattern.lower()):
                if provider_name in self._providers:
                    return provider_name

        return proxy_cfg.default_provider

    def _get_provider(self, name: str) -> BaseProvider:
        provider = self._providers.get(name)
        if provider is None:
            raise ValueError(
                f"Provider '{name}' not configured. "
                f"Available: {list(self._providers.keys())}"
            )
        return provider

    def _generate_id(self) -> str:
        return f"chatcmpl-{int(time.time() * 1000)}"

    def _mark_exhausted_with_peers(self, provider_name: str):
        """Mark provider and all peers sharing the same API key as RPM-cooldown."""
        peer_key = self._providers[provider_name].config.api_key if provider_name in self._providers else None
        if not peer_key:
            self._rate_limiter.mark_exhausted(provider_name)
            return
        for name, prov in self._providers.items():
            if prov.config.api_key == peer_key and name != provider_name:
                self._rate_limiter.mark_exhausted(name)
        self._rate_limiter.mark_exhausted(provider_name)

    async def chat(self, request: ProxyRequest) -> ProxyResponse:
        primary_name = self._resolve_provider(request.model)
        msg_count = len(request.messages)
        tools_count = len(request.tools) if request.tools else 0

        tried = set()
        providers_to_try = [primary_name] + [
            name for name in self._fallback_order if name != primary_name
        ]

        last_error = None
        for provider_name in providers_to_try:
            provider = self._providers.get(provider_name)
            if not provider or provider_name in tried:
                continue
            tried.add(provider_name)

            if self._rate_limiter.is_exhausted(provider_name):
                remaining = self._rate_limiter.get_cooldown_remaining(provider_name)
                if remaining > 0:
                    logger.info(f"[PROXY] skip {provider_name} (cooldown {remaining:.0f}s remaining)")
                else:
                    logger.info(f"[PROXY] skip {provider_name} (RPD exhausted)")
                continue

            if provider_name != primary_name:
                effective_model = provider.config.model
                logger.info(f"[PROXY] fallback → {provider_name} using model={effective_model}")
            else:
                effective_model = request.model

            logger.info(
                f"[PROXY] → {provider_name} | model={effective_model} "
                f"messages={msg_count} tools={tools_count} max_tokens={request.max_tokens}"
            )

            stats = self._rate_limiter.get_stats().get(provider_name, {})
            rpd = stats.get("rpd_used", 0)
            rpd_max = stats.get("rpd_limit", 0)
            if rpd_max:
                logger.info(f"[PROXY] {provider_name} daily: {rpd}/{rpd_max}")

            t0 = time.monotonic()
            try:
                await self._rate_limiter.acquire(provider_name)
            except RateLimitExceeded as e:
                elapsed = (time.monotonic() - t0) * 1000
                logger.warning(f"[PROXY] ✗ {provider_name} rate limited ({elapsed:.0f}ms): {e}")
                last_error = e
                continue

            try:
                result = await provider.chat(
                    messages=request.messages,
                    model=effective_model,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    tools=request.tools,
                )
                elapsed = (time.monotonic() - t0) * 1000
                content_len = len(result.content) if result.content else 0
                tc_count = len(result.tool_calls) if result.tool_calls else 0
                usage = result.usage
                tokens = usage.get("totalTokenCount", usage.get("total_tokens", "?"))
                logger.info(
                    f"[PROXY] ← {provider_name} | {elapsed:.0f}ms "
                    f"content={content_len}c tool_calls={tc_count} tokens={tokens} "
                    f"finish={result.finish_reason}"
                )
                return self._to_response(request.model, result)
            except Exception as e:
                elapsed = (time.monotonic() - t0) * 1000
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    logger.warning(f"[PROXY] ✗ {provider_name} 429/rate limit ({elapsed:.0f}ms): {e}")
                    self._mark_exhausted_with_peers(provider_name)
                    last_error = e
                    continue
                if "404" in error_str:
                    logger.warning(f"[PROXY] ✗ {provider_name} 404 model not found ({elapsed:.0f}ms): {e}")
                    self._rate_limiter.mark_exhausted(provider_name)
                    last_error = e
                    continue
                logger.error(f"[PROXY] ✗ {provider_name} error ({elapsed:.0f}ms): {e}")
                last_error = e
                continue
            finally:
                self._rate_limiter.release(provider_name)

        logger.error(f"[PROXY] ✗ all providers exhausted, last error: {last_error}")
        return ProxyResponse(
            id=self._generate_id(),
            created=int(time.time()),
            model=request.model,
            choices=[],
            usage={},
        )

    async def chat_stream(
        self, request: ProxyRequest
    ) -> AsyncIterator[str]:
        primary_name = self._resolve_provider(request.model)
        msg_count = len(request.messages)
        tools_count = len(request.tools) if request.tools else 0

        tried = set()
        providers_to_try = [primary_name] + [
            name for name in self._fallback_order if name != primary_name
        ]

        for provider_name in providers_to_try:
            provider = self._providers.get(provider_name)
            if not provider or provider_name in tried:
                continue
            tried.add(provider_name)

            if self._rate_limiter.is_exhausted(provider_name):
                remaining = self._rate_limiter.get_cooldown_remaining(provider_name)
                if remaining > 0:
                    logger.info(f"[PROXY] skip {provider_name} (cooldown {remaining:.0f}s remaining)")
                else:
                    logger.info(f"[PROXY] skip {provider_name} (RPD exhausted)")
                continue

            if provider_name != primary_name:
                effective_model = provider.config.model
                logger.info(f"[PROXY] fallback → {provider_name} using model={effective_model}")
            else:
                effective_model = request.model

            logger.info(
                f"[PROXY] → {provider_name} | STREAM model={effective_model} "
                f"messages={msg_count} tools={tools_count}"
            )

            stats = self._rate_limiter.get_stats().get(provider_name, {})
            rpd = stats.get("rpd_used", 0)
            rpd_max = stats.get("rpd_limit", 0)
            if rpd_max:
                logger.info(f"[PROXY] {provider_name} daily: {rpd}/{rpd_max}")

            t0 = time.monotonic()
            try:
                await self._rate_limiter.acquire(provider_name)
            except RateLimitExceeded as e:
                elapsed = (time.monotonic() - t0) * 1000
                logger.warning(f"[PROXY] ✗ {provider_name} stream rate limited ({elapsed:.0f}ms): {e}")
                continue

            chunk_id = self._generate_id()
            created = int(time.time())
            chunk_count = 0

            try:
                initial = json.dumps({
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                })
                yield f"data: {initial}\n\n"

                async for chunk in provider.chat_stream(
                    messages=request.messages,
                    model=effective_model,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    tools=request.tools,
                ):
                    if chunk.is_final:
                        break

                    if chunk.tool_call:
                        chunk_count += 1
                        data = json.dumps({
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "tool_calls": [{
                                        "id": chunk.tool_call.get("id", ""),
                                        "type": "function",
                                        "function": {
                                            "name": chunk.tool_call.get("name", ""),
                                            "arguments": json.dumps(chunk.tool_call.get("args", {})),
                                        },
                                    }]
                                },
                                "finish_reason": None,
                            }],
                        })
                        yield f"data: {data}\n\n"
                    elif chunk.content:
                        chunk_count += 1
                        data = json.dumps({
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": chunk.content},
                                "finish_reason": None,
                            }],
                        })
                        yield f"data: {data}\n\n"

                final = json.dumps({
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                })
                yield f"data: {final}\n\n"
                yield "data: [DONE]\n\n"
                elapsed = (time.monotonic() - t0) * 1000
                logger.info(
                    f"[PROXY] ← {provider_name} | STREAM done {elapsed:.0f}ms "
                    f"chunks={chunk_count}"
                )
                return
            except Exception as e:
                elapsed = (time.monotonic() - t0) * 1000
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    logger.warning(f"[PROXY] ✗ {provider_name} stream 429 ({elapsed:.0f}ms): {e}")
                    self._mark_exhausted_with_peers(provider_name)
                    continue
                logger.error(f"[PROXY] ✗ {provider_name} stream error ({elapsed:.0f}ms): {e}")
                error = json.dumps({"error": {"message": str(e), "type": "provider_error"}})
                yield f"data: {error}\n\n"
                return
            finally:
                self._rate_limiter.release(provider_name)

        logger.error(f"[PROXY] ✗ all providers exhausted for stream")
        error = json.dumps({"error": {"message": "All providers rate limited", "type": "rate_limit_exceeded"}})
        yield f"data: {error}\n\n"

    async def chat_with_candidates(self, candidates: list, request: ProxyRequest) -> ProxyResponse:
        """Try cloud candidates in order (rate-limit / 429 aware fallback)."""
        from .providers import create_provider, ProviderConfig

        last_error = None
        for cand in candidates:
            if getattr(cand, "source", "") not in ("cloud",):
                continue
            provider_key = f"pool_{cand.key}"
            provider = self._providers.get(cand.provider_type)
            if provider is None and cand.api_key:
                try:
                    provider = create_provider(
                        cand.provider_type,
                        ProviderConfig(
                            api_key=cand.api_key,
                            base_url=cand.base_url,
                            model=cand.model_name,
                            timeout=cand.extra_config.get("timeout", 180),
                        ),
                    )
                except Exception as e:
                    logger.warning("[PROXY] skip %s: %s", cand.key, e)
                    last_error = e
                    continue

            if provider is None:
                provider = self._providers.get(cand.extra_config.get("yaml_provider", ""))
            if provider is None:
                continue

            rl_key = provider_key if provider_key in self._rate_limiter._buckets else cand.provider_type
            if cand.rpm or cand.rpd:
                if rl_key not in self._rate_limiter._buckets:
                    self._rate_limiter.add_provider(
                        rl_key, rpm=cand.rpm or 30, rpd=cand.rpd or 0,
                        burst=max(2, (cand.rpm or 30) // 4), queue_size=20,
                    )

            if self._rate_limiter.is_exhausted(rl_key):
                logger.info("[PROXY] skip %s (rate limited)", cand.name)
                continue

            effective_model = cand.model_name or request.model
            logger.info("[PROXY] pool → %s model=%s", cand.name, effective_model)

            try:
                await self._rate_limiter.acquire(rl_key)
                result = await provider.chat(
                    messages=request.messages,
                    model=effective_model,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    tools=request.tools,
                )
                return self._to_response(effective_model, result)
            except Exception as e:
                err = str(e)
                if "429" in err or "quota" in err.lower() or "RESOURCE_EXHAUSTED" in err:
                    self._rate_limiter.mark_exhausted(rl_key)
                    yaml_prov = cand.extra_config.get("yaml_provider", "")
                    if yaml_prov and yaml_prov != rl_key:
                        self._rate_limiter.mark_exhausted(yaml_prov)
                logger.warning("[PROXY] candidate %s failed: %s", cand.name, e)
                last_error = e
            finally:
                if rl_key in self._rate_limiter._buckets:
                    self._rate_limiter.release(rl_key)

        logger.error("[PROXY] all pool candidates exhausted: %s", last_error)
        return ProxyResponse(
            id=self._generate_id(),
            created=int(time.time()),
            model=request.model,
            choices=[],
            usage={},
        )

    def _to_response(self, model: str, result: ProviderResponse) -> ProxyResponse:
        choice: dict = {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": result.content,
            },
            "finish_reason": result.finish_reason or "stop",
        }
        if result.tool_calls:
            choice["message"]["tool_calls"] = result.tool_calls

        return ProxyResponse(
            id=self._generate_id(),
            created=int(time.time()),
            model=model,
            choices=[choice],
            usage=result.usage,
        )

    def get_status(self) -> dict:
        return {
            "enabled": self.config.proxy.enabled,
            "providers": list(self._providers.keys()),
            "model_routes": self.config.proxy.model_routes,
            "rate_limit_stats": self._rate_limiter.get_stats(),
        }
