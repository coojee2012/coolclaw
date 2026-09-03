from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncGenerator

import httpx

from .base import BaseProvider, ProviderConfig, ProviderResponse, ProviderStreamChunk

logger = logging.getLogger(__name__)

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GoogleAIProvider(BaseProvider):
    """Async Google AI (Gemini) provider.

    Converts OpenAI-format messages to Gemini format and back.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.api_key = config.api_key
        self.base_url = config.base_url or GEMINI_BASE
        self._proxy = self._resolve_proxy()

    def _resolve_proxy(self) -> str | None:
        import os
        return os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")

    def _build_url(self, model: str, endpoint: str) -> str:
        return f"{self.base_url}/models/{model}:{endpoint}?key={self.api_key}"

    def _content_to_parts(self, content) -> list[dict]:
        """OpenAI multimodal content → Gemini parts (text + inline_data images)."""
        if isinstance(content, str):
            return [{"text": content}] if content else []
        if isinstance(content, list):
            parts: list[dict] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                kind = item.get("type", "")
                if kind == "text":
                    parts.append({"text": item.get("text", "")})
                elif kind == "image_url":
                    url = (item.get("image_url") or {}).get("url", "")
                    if url.startswith("data:"):
                        try:
                            header, b64data = url.split(",", 1)
                            mime = header.split(";")[0].replace("data:", "") or "image/jpeg"
                            parts.append({"inline_data": {"mime_type": mime, "data": b64data}})
                        except ValueError:
                            logger.warning("Invalid data URL for Gemini image part")
                    elif url:
                        parts.append({"text": f"[image attachment: {url}]"})
            return parts or [{"text": ""}]
        return [{"text": str(content)}]

    def _map_model(self, model: str | None) -> str:
        m = (model or self.config.model).lower()
        if "gemini-3.5" in m:
            return "gemini-3.5-flash-lite"
        if "gemini-3.1" in m:
            return "gemini-3.1-flash-lite"
        if "gemini-3" in m:
            return "gemini-3-flash"
        if "gemma" in m:
            if "26b" in m:
                return "gemma-4-26b-a4b-it"
            return "gemma-4-31b-it"
        return model or self.config.model

    def _convert_messages(
        self, messages: list[dict], model: str
    ) -> tuple[list[dict], dict | None]:
        active = model.lower()
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        system_instruction = None
        if system_msgs:
            sys_text = "\n".join(m.get("content", "") for m in system_msgs)
            system_instruction = {"role": "system", "parts": [{"text": sys_text}]}

        if "gemma" in active:
            if not system_instruction:
                system_instruction = {
                    "role": "system",
                    "parts": [{"text": "You are a helpful assistant."}],
                }

            total_chars = sum(len(m.get("content", "")) for m in non_system)
            while total_chars > 14000 and len(non_system) > 1:
                removed = non_system.pop(0)
                total_chars -= len(removed.get("content", ""))

            if len(non_system) <= 1:
                content = non_system[0].get("content", "") if non_system else ""
                contents = [{"parts": [{"text": content}]}]
            else:
                parts = []
                for msg in non_system:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "assistant":
                        parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
                    else:
                        parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
                parts.append("<start_of_turn>model\n")
                contents = [{"parts": [{"text": "\n".join(parts)}]}]

            return contents, system_instruction

        gemini_msgs = []
        for msg in non_system:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])

            if role == "assistant":
                parts = []
                if content:
                    parts.append({"text": content})
                for tc in tool_calls:
                    parts.append(
                        {
                            "functionCall": {
                                "name": tc.get("function", {}).get("name"),
                                "args": json.loads(
                                    tc.get("function", {}).get("arguments", "{}")
                                ),
                            }
                        }
                    )
                gemini_msgs.append({"role": "model", "parts": parts})
            elif role == "tool":
                gemini_msgs.append(
                    {
                        "role": "function",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": msg.get("name", ""),
                                    "response": {"result": content},
                                }
                            }
                        ],
                    }
                )
            else:
                gemini_msgs.append({"role": "user", "parts": self._content_to_parts(content)})

        return gemini_msgs, system_instruction

    def _convert_tools(self, tools: list[dict]) -> list[dict] | None:
        gemini_tools = []
        for tool in tools:
            func = tool.get("function", {})
            if func:
                params = self._sanitize_gemini_schema(func.get("parameters", {}))
                gemini_tools.append(
                    {
                        "function_declarations": [
                            {
                                "name": func.get("name", ""),
                                "description": func.get("description", ""),
                                "parameters": params,
                            }
                        ]
                    }
                )
        return gemini_tools if gemini_tools else None

    @staticmethod
    def _sanitize_gemini_schema(schema: dict) -> dict:
        """Strip JSON-Schema fields Gemini function calling rejects."""
        if not isinstance(schema, dict):
            return {"type": "object", "properties": {}}

        def _clean(obj):
            if not isinstance(obj, dict):
                return obj
            out = {}
            for k, v in obj.items():
                if k in ("additionalProperties", "$schema", "$id", "definitions"):
                    continue
                if k == "properties" and isinstance(v, dict):
                    out[k] = {pk: _clean(pv) for pk, pv in v.items()}
                elif k == "items" and isinstance(v, dict):
                    out[k] = _clean(v)
                elif isinstance(v, dict):
                    out[k] = _clean(v)
                else:
                    out[k] = v
            if out.get("type") == "object" and "properties" not in out:
                out["properties"] = {}
            return out

        return _clean(schema)

    def _parse_response(self, data: dict, model: str) -> ProviderResponse:
        content = ""
        tool_calls = []
        finish_reason = ""

        for candidate in data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if "text" in part:
                    content += part["text"]
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    tool_calls.append(
                        {
                            "id": fc.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": fc.get("name", ""),
                                "arguments": json.dumps(fc.get("args", {})),
                            },
                        }
                    )
            finish_reason = candidate.get("finishReason", "")

        usage = data.get("usageMetadata", {})
        return ProviderResponse(
            content=content,
            model=model,
            usage=usage,
            finish_reason=finish_reason,
            tool_calls=tool_calls if tool_calls else None,
        )

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
    ) -> ProviderResponse:
        active_model = self._map_model(model)
        if active_model != model:
            logger.info(f"[GEMINI] model mapped: {model} → {active_model}")
        contents, system_instruction = self._convert_messages(messages, active_model)

        payload: dict = {"contents": contents}
        gen_config: dict = {}
        temp = temperature
        if temp is not None:
            gen_config["temperature"] = temp
        if "gemma" not in active_model.lower():
            gen_config["topP"] = 0.9
            gen_config["maxOutputTokens"] = max_tokens
        if gen_config:
            payload["generationConfig"] = gen_config
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        if tools and "gemma" not in active_model.lower():
            gemini_tools = self._convert_tools(tools)
            if gemini_tools:
                payload["tools"] = gemini_tools

        url = self._build_url(active_model, "generateContent")
        serialized = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        max_retries = 4
        backoff = 0.5
        last_error = None

        logger.info(
            f"[GEMINI] → {active_model} | generateContent "
            f"messages={len(contents)} tools={len(tools) if tools else 0} "
            f"max_tokens={max_tokens} temp={temperature}"
        )
        t0 = time.monotonic()

        for attempt in range(max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.config.timeout, proxy=self._proxy) as client:
                    resp = await client.post(
                        url, content=serialized, headers={"Content-Type": "application/json"}
                    )
                    if resp.status_code == 400:
                        error_body = resp.text[:500] if resp.text else "empty"
                        logger.error(f"[GEMINI] ✗ 400 from {active_model}: {error_body}")
                        logger.error(f"[GEMINI] Request payload: {json.dumps(payload, ensure_ascii=False)[:1000]}")
                    resp.raise_for_status()
                    result = self._parse_response(resp.json(), active_model)
                    elapsed = (time.monotonic() - t0) * 1000
                    usage = result.usage
                    tokens = usage.get("totalTokenCount", "?")
                    logger.info(
                        f"[GEMINI] ← {active_model} | {elapsed:.0f}ms "
                        f"status={resp.status_code} tokens={tokens} "
                        f"finish={result.finish_reason} content={len(result.content)}c"
                    )
                    return result
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                last_error = e

                if status == 500 and "gemma" in active_model.lower() and attempt < max_retries:
                    prev_model = active_model
                    active_model = (
                        "gemma-4-26b-a4b-it" if "31b" in active_model else "gemma-4-31b-it"
                    )
                    url = self._build_url(active_model, "generateContent")
                    sleep_time = backoff * (2**attempt)
                    logger.warning(
                        f"[GEMINI] ✗ 500 from {prev_model}, "
                        f"retrying {active_model} in {sleep_time:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(sleep_time)
                    continue

                if status == 429:
                    logger.error(f"[GEMINI] ✗ 429 from {active_model}: quota exhausted, no retry")
                    raise

                if status == 503 and attempt < max_retries:
                    sleep_time = backoff * (2**attempt)
                    logger.warning(
                        f"[GEMINI] ✗ 503 from {active_model}, "
                        f"retrying in {sleep_time:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(sleep_time)
                    continue
                logger.error(f"[GEMINI] ✗ {status} from {active_model}: {e}")
                raise
            except httpx.RequestError as e:
                last_error = e
                if attempt < max_retries:
                    sleep_time = backoff * (2**attempt)
                    logger.warning(
                        f"[GEMINI] ✗ network error: {e}, "
                        f"retrying in {sleep_time:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(sleep_time)
                    continue
                logger.error(f"[GEMINI] ✗ network error (exhausted retries): {e}")
                raise

        raise last_error  # type: ignore[misc]

    async def chat_stream(  # type: ignore[override]
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
    ) -> AsyncGenerator[ProviderStreamChunk, None]:
        active_model = self._map_model(model)
        contents, system_instruction = self._convert_messages(messages, active_model)

        payload: dict = {"contents": contents}
        gen_config: dict = {"temperature": temperature}
        if "gemma" not in active_model.lower():
            gen_config["topP"] = 0.9
            gen_config["maxOutputTokens"] = max_tokens
        payload["generationConfig"] = gen_config
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        can_stream = "gemma" not in active_model.lower()
        endpoint = "streamGenerateContent" if can_stream else "generateContent"
        url = self._build_url(active_model, endpoint)
        if can_stream:
            sep = "&" if "?" in url else "?"
            url += f"{sep}alt=sse"
        serialized = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        if not can_stream:
            logger.info(f"[GEMINI] {active_model} does not support streaming, falling back to sync")
            resp = await self.chat(messages, model, max_tokens, temperature, tools)
            if resp.tool_calls:
                for tc in resp.tool_calls:
                    yield ProviderStreamChunk(
                        tool_call={
                            "name": tc["function"]["name"],
                            "args": json.loads(tc["function"]["arguments"]),
                            "id": tc.get("id", ""),
                        }
                    )
            else:
                yield ProviderStreamChunk(content=resp.content)
            yield ProviderStreamChunk(is_final=True)
            return

        max_retries = 3
        backoff = 0.5
        response = None
        client = None
        chunk_count = 0

        logger.info(f"[GEMINI] → {active_model} | streamGenerateContent")
        t0 = time.monotonic()

        for attempt in range(max_retries + 1):
            try:
                client = httpx.AsyncClient(timeout=self.config.timeout, proxy=self._proxy)
                response = await client.send(
                    client.build_request(
                        "POST", url, content=serialized, headers={"Content-Type": "application/json"}
                    ),
                    stream=True,
                )
                response.raise_for_status()
                break
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                if response:
                    await response.aclose()
                if client:
                    await client.aclose()
                status = getattr(getattr(e, "response", None), "status_code", None)
                if (status in (429, 503) or isinstance(e, httpx.RequestError)) and attempt < max_retries:
                    sleep_time = backoff * (2**attempt)
                    logger.warning(
                        f"[GEMINI] ✗ stream connect failed ({status or 'network'}), "
                        f"retrying in {sleep_time:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(sleep_time)
                    continue
                logger.error(f"[GEMINI] ✗ stream connect failed: {e}")
                raise

        try:
            async for line in response.aiter_lines():  # type: ignore[union-attr]
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data: "):
                    line = line[6:]
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for candidate in data.get("candidates", []):
                    for part in candidate.get("content", {}).get("parts", []):
                        if "text" in part:
                            chunk_count += 1
                            yield ProviderStreamChunk(content=part["text"])
                        elif "functionCall" in part:
                            fc = part["functionCall"]
                            yield ProviderStreamChunk(
                                tool_call={
                                    "name": fc.get("name", ""),
                                    "args": fc.get("args", {}),
                                    "id": fc.get("id", ""),
                                }
                            )
        finally:
            if response:
                await response.aclose()
            if client:
                await client.aclose()

        elapsed = (time.monotonic() - t0) * 1000
        logger.info(f"[GEMINI] ← {active_model} | stream done {elapsed:.0f}ms chunks={chunk_count}")
        yield ProviderStreamChunk(is_final=True)

    async def list_models(self) -> list[dict]:
        url = f"{self.base_url}/models?key={self.api_key}"
        async with httpx.AsyncClient(timeout=30, proxy=self._proxy) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            return [
                {"id": m["name"].split("/")[-1], "owned_by": "Google"}
                for m in data.get("models", [])
            ]
