import os
import logging
import json
from typing import Optional, Iterator
from dataclasses import dataclass, field

import httpx

from .config import Config, CloudModelConfig


logger = logging.getLogger(__name__)


@dataclass
class GeminiMessage:
    role: str
    content: str
    tool_calls: Optional[list[dict]] = None


@dataclass
class GeminiResponse:
    content: str
    model: str
    usage: dict
    finish_reason: str
    tool_calls: Optional[list[dict]] = None


@dataclass
class GeminiChunk:
    content: str
    is_final: bool = False
    tool_call: Optional[dict] = None


class GeminiClient:
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, config: CloudModelConfig):
        self.api_key = config.api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            logger.warning("No Gemini API key provided")
        self.model = config.model
        self.temperature = config.temperature
        self.top_p = config.top_p
        self.max_tokens = config.max_tokens
        self.safety_threshold = config.safety_threshold
        self._cache = {}
        self._proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")

    def _map_model(self, model: str) -> str:
        if not model:
            return self.model
        m = model.lower()
        if "gemini-3.5" in m:
            return "gemini-3.5-flash-lite"
        if "gemini-3.1" in m:
            return "gemini-3.1-flash-lite"
        if "gemini-3" in m:
            return "gemini-2.5-flash"
        return model

    def _build_url(self, endpoint: str, model: Optional[str] = None) -> str:
        active_model = model or self.model
        active_model = self._map_model(active_model)
        return f"{self.BASE_URL}/models/{active_model}:{endpoint}?key={self.api_key}"

    def _convert_tools_to_gemini(self, tools: list) -> list[dict]:
        gemini_tools = []

        for tool in tools:
            if hasattr(tool, "function"):
                func = tool.function
                gemini_tools.append(
                    {
                        "function_declarations": [
                            {
                                "name": func.name,
                                "description": func.description or "",
                                "parameters": func.parameters or {},
                            }
                        ]
                    }
                )
            elif isinstance(tool, dict) and tool.get("type") == "function":
                func = tool.get("function", {})
                gemini_tools.append(
                    {
                        "function_declarations": [
                            {
                                "name": func.get("name"),
                                "description": func.get("description", ""),
                                "parameters": func.get("parameters", {}),
                            }
                        ]
                    }
                )

        return gemini_tools if gemini_tools else None

    def _convert_messages_to_gemini(self, messages: list[dict], model: str) -> tuple[list[dict], Optional[dict]]:
        active_model = model.lower()
        
        # Extract system messages
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        non_system_messages = [msg for msg in messages if msg.get("role") != "system"]
        
        system_instruction = None
        if system_messages:
            sys_text = "\n".join([msg.get("content", "") for msg in system_messages])
            system_instruction = {
                "role": "system",
                "parts": [{"text": sys_text}]
            }
            
        if "gemma" in active_model:
            # Gemma models MUST have a systemInstruction (even if dummy) to prevent single-turn 500 parser crashes!
            if not system_instruction:
                system_instruction = {
                    "role": "system",
                    "parts": [{"text": "You are a helpful assistant."}]
                }
                
            # Truncate conversation history to fit in Gemma's strict 8192-token context window
            # We want to keep the most recent messages.
            # Let's target a maximum character length of 14,000 characters for non-system messages.
            total_chars = sum(len(msg.get("content", "")) for msg in non_system_messages)
            while total_chars > 14000 and len(non_system_messages) > 1:
                removed_msg = non_system_messages.pop(0)
                total_chars -= len(removed_msg.get("content", ""))

            if len(non_system_messages) <= 1:
                content = non_system_messages[0].get("content", "") if non_system_messages else ""
                contents = [{"parts": [{"text": content}]}]
            else:
                # Multi-turn: format using Gemma template tokens
                formatted_text = ""
                for msg in non_system_messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "assistant":
                        formatted_text += f"<start_of_turn>model\n{content}<end_of_turn>\n"
                    else:
                        formatted_text += f"<start_of_turn>user\n{content}<end_of_turn>\n"
                formatted_text += "<start_of_turn>model\n"
                contents = [{"parts": [{"text": formatted_text}]}]
                
            return contents, system_instruction

        # Standard Gemini models
        gemini_messages = []
        for msg in non_system_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            tool_call_id = msg.get("tool_call_id")

            if role == "assistant":
                parts = []
                if content:
                    parts.append({"text": content})
                if tool_calls:
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
                gemini_messages.append({"role": "model", "parts": parts})
            elif role == "tool":
                gemini_messages.append(
                    {
                        "role": "function",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": msg.get("name", ""),
                                    "response": {"result": content},
                                    "id": tool_call_id,
                                }
                            }
                        ],
                    }
                )
            else:
                gemini_messages.append({"role": "user", "parts": [{"text": content}]})
        return gemini_messages, system_instruction

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
        model: Optional[str] = None,
    ) -> GeminiResponse | Iterator[GeminiChunk]:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")

        active_model = model or self.model
        contents, system_instruction = self._convert_messages_to_gemini(messages, active_model)

        # Caching logic
        import time
        cache_key = None
        if not stream:
            try:
                now = time.time()
                expired_keys = [k for k, (_, ts) in self._cache.items() if now - ts > 60]
                for k in expired_keys:
                    del self._cache[k]

                msg_str = json.dumps(messages, sort_keys=True)
                tools_str = json.dumps(tools, sort_keys=True) if tools else ""
                cache_key = (active_model, msg_str, tools_str, temperature, top_p, max_tokens)
                
                if cache_key in self._cache:
                    cached_val, timestamp = self._cache[cache_key]
                    if now - timestamp < 60:
                        logger.info("Serving response from local Gemini cache")
                        return cached_val
            except Exception as e:
                logger.error(f"Error checking cache: {e}")

        payload = {
            "contents": contents,
        }

        # Build generationConfig conditionally to bypass Google's Gemma-4 500 crash bugs
        gen_config = {}
        if "gemma" in active_model.lower():
            temp_val = temperature or self.temperature
            if temp_val is not None:
                gen_config["temperature"] = temp_val
        else:
            temp_val = temperature or self.temperature
            if temp_val is not None:
                gen_config["temperature"] = temp_val
            top_p_val = top_p or self.top_p
            if top_p_val is not None:
                gen_config["topP"] = top_p_val
            max_tokens_val = max_tokens or self.max_tokens
            if max_tokens_val is not None:
                gen_config["maxOutputTokens"] = max_tokens_val

        if gen_config:
            payload["generationConfig"] = gen_config

        if system_instruction:
            payload["systemInstruction"] = system_instruction

        if tools and "gemma" not in active_model.lower():
            gemini_tools = self._convert_tools_to_gemini(tools)
            if gemini_tools:
                payload["tools"] = gemini_tools

        # Gemma models don't support streaming
        can_stream = stream and "gemma" not in active_model.lower()

        endpoint = "streamGenerateContent" if can_stream else "generateContent"
        url = self._build_url(endpoint, active_model)

        if can_stream:
            return self._stream_generate(url, payload)

        # Non-streaming: always return as iterator for compatibility
        result = self._generate(url, payload, active_model)

        if cache_key is not None:
            try:
                self._cache[cache_key] = (result, time.time())
            except Exception as e:
                logger.error(f"Error writing to cache: {e}")

        if not stream:
            return result

        # Wrap GeminiResponse as iterator
        def wrapper():
            yield GeminiChunk(
                content=result.content,
                tool_call=result.tool_calls[0] if result.tool_calls else None,
            )
            yield GeminiChunk(content="", is_final=True)

        return wrapper()

    def _generate(self, url: str, payload: dict, model: str) -> GeminiResponse:
        headers = {"Content-Type": "application/json"}
        import time

        active_model = model
        max_retries = 4
        backoff = 0.5

        # Serialize payload using ensure_ascii=False to prevent \uXXXX escape crashing on Gemma backend
        serialized_payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        for attempt in range(max_retries + 1):
            try:
                # Dynamically build active url for the selected model
                active_url = self._build_url("generateContent", active_model)
                with httpx.Client(timeout=60.0, proxy=self._proxy) as client:
                    response = client.post(active_url, content=serialized_payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    break
            except httpx.HTTPStatusError as e:
                status = e.response.status_code

                # Automatic cross-model fallback & retry for Gemma 500 errors
                if status == 500 and "gemma" in active_model.lower() and attempt < max_retries:
                    if "31b" in active_model:
                        next_model = "gemma-4-26b-a4b-it"
                    else:
                        next_model = "gemma-4-31b-it"
                    
                    sleep_time = backoff * (2 ** attempt)
                    logger.warning(f"🔥 [Gemini Client] {active_model} returned 500 Internal Error. Retrying with {next_model} in {sleep_time:.2f}s (attempt {attempt + 1}/{max_retries})...")
                    active_model = next_model
                    time.sleep(sleep_time)
                    continue

                if status == 429:
                    logger.error(f"Google AI returned 429 quota exhausted, no retry")
                    raise e

                if status == 503 and attempt < max_retries:
                    sleep_time = backoff * (2 ** attempt)
                    logger.warning(f"Google AI returned 503, retrying in {sleep_time:.2f}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(sleep_time)
                    continue
                raise e
            except httpx.RequestError as e:
                if attempt < max_retries:
                    sleep_time = backoff * (2 ** attempt)
                    logger.warning(f"Network error: {e}, retrying in {sleep_time:.2f}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(sleep_time)
                    continue
                raise e

        content = ""
        tool_calls = []
        finish_reason = ""

        if "candidates" in data:
            for candidate in data["candidates"]:
                if "content" in candidate:
                    for part in candidate["content"].get("parts", []):
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

        return GeminiResponse(
            content=content,
            model=model,
            usage=usage,
            finish_reason=finish_reason,
            tool_calls=tool_calls if tool_calls else None,
        )

    def _stream_generate(self, url: str, payload: dict) -> Iterator[GeminiChunk]:
        headers = {"Content-Type": "application/json"}
        import time

        print(f"\n🔥 [Gemini Client Log] Stream call with payload: {json.dumps(payload, ensure_ascii=False)}\n", flush=True)

        max_retries = 3
        backoff = 0.5

        response = None
        client = None
        # Serialize payload using ensure_ascii=False to prevent \uXXXX escape crashing on Gemma backend
        serialized_payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        for attempt in range(max_retries + 1):
            try:
                client = httpx.Client(timeout=120.0, proxy=self._proxy)
                response = client.send(
                    client.build_request("POST", url, content=serialized_payload, headers=headers),
                    stream=True
                )
                response.raise_for_status()
                break
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                if response:
                    response.close()
                if client:
                    client.close()

                status = getattr(getattr(e, "response", None), "status_code", None)
                if status == 429:
                    logger.error(f"Stream 429 quota exhausted, no retry")
                    raise e
                if (status == 503 or isinstance(e, httpx.RequestError)) and attempt < max_retries:
                    sleep_time = backoff * (2 ** attempt)
                    logger.warning(f"Stream connect failed ({status or 'network error'}), retrying in {sleep_time:.2f}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(sleep_time)
                    continue
                raise e

        try:
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "candidates" in data:
                            for candidate in data["candidates"]:
                                if "content" in candidate:
                                    for part in candidate["content"].get("parts", []):
                                        if "text" in part:
                                            yield GeminiChunk(content=part["text"])
                                        elif "functionCall" in part:
                                            fc = part["functionCall"]
                                            yield GeminiChunk(
                                                content="",
                                                tool_call={
                                                    "name": fc.get("name", ""),
                                                    "args": fc.get("args", {}),
                                                    "id": fc.get("id", ""),
                                                },
                                            )
                    except json.JSONDecodeError:
                        continue
        finally:
            if response:
                response.close()
            if client:
                client.close()

        yield GeminiChunk(content="", is_final=True)

    def count_tokens(self, text: str) -> int:
        if not self.api_key:
            return len(text) // 4

        url = self._build_url("countTokens")
        payload = {"contents": [{"parts": [{"text": text}]}]}
        headers = {"Content-Type": "application/json"}

        try:
            with httpx.Client(timeout=30.0, proxy=self._proxy) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                return data.get("totalTokens", len(text) // 4)
        except Exception:
            return len(text) // 4


def create_gemini_client(config: Config) -> GeminiClient:
    cloud_config = config.models.cloud.google_ai
    return GeminiClient(cloud_config)
