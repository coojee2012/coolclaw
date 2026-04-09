import logging
from typing import Optional, Iterator, AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from .router import Router, Provider


logger = logging.getLogger(__name__)


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "local"
    messages: list[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    stop: Optional[list[str]] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class CompletionRequest(BaseModel):
    model: str = "local"
    prompt: str
    suffix: Optional[str] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    stop: Optional[list[str]] = None


class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage


class ModelList(BaseModel):
    object: str = "list"
    data: list[dict]


class Model(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "opencode-helper"


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[dict]


def generate_id() -> str:
    import time

    return f"chatcmpl-{int(time.time() * 1000)}"


def router_dependency(router: Router):
    return router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("OpenCode Helper API starting...")
    yield
    logger.info("OpenCode Helper API shutting down...")


def create_app(router: Optional[Router] = None) -> FastAPI:
    app = FastAPI(
        title="OpenCode Helper API",
        description="Lightweight local AI assistant with Gemini fallback",
        version="0.1.0",
        lifespan=lifespan,
    )

    _router = router or Router()

    @app.get("/health")
    async def health():
        return {"status": "ok", "router": _router.get_status()}

    @app.get("/v1/models")
    async def list_models():
        return ModelList(
            data=[
                {"id": "qwen2.5-coder-7b", "object": "model", "owned_by": "Qwen"},
                {"id": "qwen2.5-coder-14b", "object": "model", "owned_by": "Qwen"},
                {"id": "gemini-2.5-flash", "object": "model", "owned_by": "Google"},
                {"id": "gemini-3-flash", "object": "model", "owned_by": "Google"},
            ]
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        if request.stream:
            return StreamingResponse(
                _stream_chat(request, messages, _router),
                media_type="text/event-stream",
            )

        try:
            response = _router.chat(
                messages=messages,
                max_tokens=request.max_tokens or 2048,
                temperature=request.temperature or 0.7,
                stream=False,
            )

            return ChatCompletionResponse(
                id=generate_id(),
                created=int(__import__("time").time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=response.content,
                        ),
                        finish_reason=response.finish_reason or "stop",
                    )
                ],
                usage=Usage(
                    prompt_tokens=len(str(messages)) // 4,
                    completion_tokens=len(response.content) // 4,
                    total_tokens=len(str(messages)) // 4 + len(response.content) // 4,
                ),
            )
        except Exception as e:
            logger.error(f"Chat completion error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        if request.stream:
            return StreamingResponse(
                _stream_complete(request, _router),
                media_type="text/event-stream",
            )

        try:
            response = _router.complete(
                prompt=request.prompt,
                max_tokens=request.max_tokens or 2048,
                temperature=request.temperature or 0.7,
                stream=False,
            )

            return CompletionResponse(
                id=generate_id(),
                created=int(__import__("time").time()),
                model=request.model,
                choices=[
                    CompletionChoice(
                        text=response.content,
                        index=0,
                        finish_reason="stop",
                    )
                ],
                usage=Usage(
                    prompt_tokens=len(request.prompt) // 4,
                    completion_tokens=len(response.content) // 4,
                    total_tokens=len(request.prompt) // 4 + len(response.content) // 4,
                ),
            )
        except Exception as e:
            logger.error(f"Completion error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/router/status")
    async def router_status():
        return _router.get_status()

    @app.post("/router/mode")
    async def set_router_mode(mode: str):
        if mode not in ("local_only", "cloud_only", "auto"):
            raise HTTPException(status_code=400, detail="Invalid mode")
        _router.set_mode(mode)
        return {"mode": mode}

    @app.post("/router/unload")
    async def unload_model():
        _router.unload_local_model()
        return {"status": "unloaded"}

    return app


async def _stream_chat(
    request: ChatCompletionRequest,
    messages: list[dict],
    router: Router,
) -> AsyncIterator[str]:
    import json
    import time

    chunk_id = generate_id()

    try:
        response = router.chat(
            messages=messages,
            max_tokens=request.max_tokens or 2048,
            temperature=request.temperature or 0.7,
            stream=True,
        )

        for i, chunk in enumerate(response.chunks):
            if hasattr(chunk, "content") and chunk.content:
                yield f"data: {
                    json.dumps(
                        {
                            'id': chunk_id,
                            'object': 'chat.completion.chunk',
                            'created': int(time.time()),
                            'model': request.model,
                            'choices': [
                                {
                                    'index': 0,
                                    'delta': {'content': chunk.content},
                                    'finish_reason': None,
                                }
                            ],
                        }
                    )
                }\n\n"

        yield f"data: {
            json.dumps(
                {
                    'id': chunk_id,
                    'object': 'chat.completion.chunk',
                    'created': int(time.time()),
                    'model': request.model,
                    'choices': [
                        {
                            'index': 0,
                            'delta': {},
                            'finish_reason': 'stop',
                        }
                    ],
                }
            )
        }\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


async def _stream_complete(
    request: CompletionRequest,
    router: Router,
) -> AsyncIterator[str]:
    import json
    import time

    chunk_id = generate_id()

    try:
        response = router.complete(
            prompt=request.prompt,
            max_tokens=request.max_tokens or 2048,
            temperature=request.temperature or 0.7,
            stream=True,
        )

        for i, chunk in enumerate(response.chunks):
            if hasattr(chunk, "content") and chunk.content:
                yield f"data: {
                    json.dumps(
                        {
                            'id': chunk_id,
                            'object': 'text_completion',
                            'created': int(time.time()),
                            'model': request.model,
                            'choices': [
                                {
                                    'index': 0,
                                    'text': chunk.content,
                                    'logprobs': None,
                                    'finish_reason': None,
                                }
                            ],
                        }
                    )
                }\n\n"

        yield f"data: {
            json.dumps(
                {
                    'id': chunk_id,
                    'object': 'text_completion',
                    'created': int(time.time()),
                    'model': request.model,
                    'choices': [
                        {
                            'index': 0,
                            'text': '',
                            'logprobs': None,
                            'finish_reason': 'stop',
                        }
                    ],
                }
            )
        }\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
