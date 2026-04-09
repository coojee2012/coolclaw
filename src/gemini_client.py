import os
import logging
from typing import Optional, Iterator
from dataclasses import dataclass

import httpx

from .config import Config, CloudModelConfig


logger = logging.getLogger(__name__)


@dataclass
class GeminiMessage:
    role: str
    content: str


@dataclass
class GeminiResponse:
    content: str
    model: str
    usage: dict
    finish_reason: str


@dataclass
class GeminiChunk:
    content: str
    is_final: bool = False


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

    def _build_url(self, endpoint: str) -> str:
        return f"{self.BASE_URL}/models/{self.model}:{endpoint}?key={self.api_key}"

    def _convert_messages_to_gemini(self, messages: list[dict]) -> list[dict]:
        gemini_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                gemini_messages.append(
                    {"role": "user", "parts": [{"text": f"[System] {content}"}]}
                )
            elif role == "assistant":
                gemini_messages.append({"role": "model", "parts": [{"text": content}]})
            else:
                gemini_messages.append({"role": "user", "parts": [{"text": content}]})
        return gemini_messages

    def chat(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
    ) -> GeminiResponse | Iterator[GeminiChunk]:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")

        contents = self._convert_messages_to_gemini(messages)

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature or self.temperature,
                "topP": top_p or self.top_p,
                "maxOutputTokens": max_tokens or self.max_tokens,
            },
        }

        endpoint = "streamGenerateContent" if stream else "generateContent"
        url = self._build_url(endpoint)

        if stream:
            return self._stream_generate(url, payload)
        else:
            return self._generate(url, payload)

    def _generate(self, url: str, payload: dict) -> GeminiResponse:
        headers = {"Content-Type": "application/json"}

        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        content = ""
        if "candidates" in data:
            for candidate in data["candidates"]:
                if "content" in candidate:
                    for part in candidate["content"]["parts"]:
                        if "text" in part:
                            content += part["text"]

        finish_reason = ""
        if "candidates" in data and data["candidates"]:
            finish_reason = data["candidates"][0].get("finishReason", "")

        usage = data.get("usageMetadata", {})

        return GeminiResponse(
            content=content,
            model=self.model,
            usage=usage,
            finish_reason=finish_reason,
        )

    def _stream_generate(self, url: str, payload: dict) -> Iterator[GeminiChunk]:
        headers = {"Content-Type": "application/json"}

        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", url, json=payload, headers=headers) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        import json

                        try:
                            data = json.loads(line)
                            if "candidates" in data:
                                for candidate in data["candidates"]:
                                    if "content" in candidate:
                                        for part in candidate["content"]["parts"]:
                                            if "text" in part:
                                                yield GeminiChunk(content=part["text"])
                        except json.JSONDecodeError:
                            continue

        yield GeminiChunk(content="", is_final=True)

    def count_tokens(self, text: str) -> int:
        if not self.api_key:
            return len(text) // 4

        url = self._build_url("countTokens")
        payload = {"contents": [{"parts": [{"text": text}]}]}
        headers = {"Content-Type": "application/json"}

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                return data.get("totalTokens", len(text) // 4)
        except Exception:
            return len(text) // 4


def create_gemini_client(config: Config) -> GeminiClient:
    cloud_config = config.models.cloud.gemini
    return GeminiClient(cloud_config)
