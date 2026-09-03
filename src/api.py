import logging
import json
from typing import Optional, Iterator, AsyncIterator
from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import (
    FastAPI,
    HTTPException,
    Header,
    Request,
    UploadFile,
    File,
    Form,
    Body,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from .router import Provider
from .auth import (
    AuthMiddleware, create_session_token, destroy_session_token,
    COOKIE_NAME, get_current_user, require_auth, require_admin,
)
from .database import db


logger = logging.getLogger(__name__)


class Function(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None


class Tool(BaseModel):
    type: str = "function"
    function: Function


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: dict  # {"name": str, "arguments": str}


class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "local"
    messages: list[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    stop: Optional[list[str]] = None
    tools: Optional[list[Tool]] = None
    tool_choice: Optional[str] = "auto"


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None


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


def router_dependency(router: "Router"):  # type: ignore # noqa: F821
    return router


_proxy_instance = None
_runtime_dispatcher = None


def _get_proxy():
    global _proxy_instance
    if _proxy_instance is None:
        from .proxy import Proxy
        _proxy_instance = Proxy()
    return _proxy_instance


def reset_runtime_singletons() -> dict:
    """Clear cached proxy / dispatcher LLM instances after config reload."""
    from .local_llm import clear_local_llm_cache

    global _proxy_instance, _runtime_dispatcher
    _proxy_instance = None
    cleared = {"proxy": True, "dispatcher_llm": False, "local_llm_cache": 0}
    cleared["local_llm_cache"] = clear_local_llm_cache()
    if _runtime_dispatcher is not None:
        _runtime_dispatcher._dispatcher_llm = None
        _runtime_dispatcher._specialist_llm = None
        cleared["dispatcher_llm"] = True
    from .dispatcher import SmartDispatcher
    if hasattr(SmartDispatcher, "_agent_llm_proxies"):
        SmartDispatcher._agent_llm_proxies = {}
    if hasattr(SmartDispatcher, "_agent_model"):
        delattr(SmartDispatcher, "_agent_model")
    return cleared


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("CoolClaw API starting...")
    yield
    logger.info("CoolClaw API shutting down...")


class ChatRequest(BaseModel):
    message: str
    stream: bool = False
    session_id: Optional[str] = None
    expert: Optional[str] = None  # force expert id; None/"" = auto route


class ChatResponse(BaseModel):
    success: bool
    content: str
    model: str
    confidence: float = 0.0
    reasoning: str = ""
    session_id: Optional[str] = None
    error: Optional[str] = None
    tool_calls: Optional[list[dict]] = None
    expert: str = ""
    expert_name: str = ""
    expert_icon: str = ""
    trace_id: str = ""
    trace: Optional[dict] = None


def create_app(router=None, web_dir: Optional[str] = None) -> FastAPI:
    from .router import Router
    from .dispatcher import Dispatcher, DispatcherConfig

    app = FastAPI(
        title="CoolClaw API",
        description="Local AI Agent platform with MCP tools and skill automation",
        version="0.2.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Auth middleware
    app.add_middleware(AuthMiddleware)

    try:
        from .config import get_config
        imported = db.seed_model_configs_from_yaml(get_config().model_dump())
        if imported:
            logger.info(f"Seeded {imported} model configs from config.yaml")
    except Exception as e:
        logger.debug(f"Model config seed skipped: {e}")

    _router = router
    _dispatcher = None
    _web_dir = web_dir or os.path.join(os.path.dirname(__file__), "../../web")

    def get_router():
        nonlocal _router
        if _router is None:
            from .router import create_router
            from .config import get_config

            _router = create_router(get_config())
        return _router

    def get_dispatcher():
        nonlocal _dispatcher
        global _runtime_dispatcher
        if _dispatcher is None:
            memory_path = os.path.join(
                os.path.dirname(__file__), "../../data/memory.json"
            )
            os.makedirs(os.path.dirname(memory_path), exist_ok=True)
            _dispatcher = Dispatcher(config=DispatcherConfig(memory_path=memory_path))
            logger.info("Dispatcher initialized")
        _runtime_dispatcher = _dispatcher
        return _dispatcher

    from .server_control import register_reload_hook

    def _app_reload_runtime():
        cleared = reset_runtime_singletons()
        nonlocal _router
        from .config import get_config
        from .router import create_router
        _router = create_router(get_config())
        return {"router": True, **cleared}

    register_reload_hook(_app_reload_runtime)

    def _resolve_chat_session(session_id: Optional[str], user: dict):
        from .session import session_manager

        uid = user["id"]
        if session_id:
            session = session_manager.get_session(session_id, uid)
            if session:
                session_manager.set_current_session(session_id, uid)
                return session
        session = session_manager.get_current_session(uid)
        if not session:
            session = session_manager.create_session(user_id=uid)
        session_manager.set_current_session(session.id, uid)
        return session

    def _require_user_session(session_id: str, user: dict):
        from .session import session_manager

        session = session_manager.get_session(session_id, user["id"])
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session

    @app.get("/health")
    async def health():
        disp = get_dispatcher()
        return {"status": "ok", "dispatcher": disp.get_status()}

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest, http_request: Request):
        from .session import session_manager, build_session_context, build_dispatch_context
        from .run_trace import RunTrace

        user = require_auth(http_request)
        try:
            disp = get_dispatcher()
            target_session = _resolve_chat_session(request.session_id, user)

            workdir = target_session.workdir or ""
            logger.info(
                f"[CHAT] user={user['id']} session={target_session.id} "
                f"workdir={workdir or '(none)'} message={request.message[:80]}"
            )

            context = build_session_context(
                target_session.messages,
                summary=target_session.summary,
                query=request.message,
                session_id=target_session.id,
            )
            dispatch_context = build_dispatch_context(
                target_session.messages,
                summary=target_session.summary,
            )

            trace = RunTrace().bind()
            result = disp.chat(
                user_input=request.message, context=context,
                dispatch_context=dispatch_context,
                skip_memory=True,
                workdir=workdir,
                forced_expert=request.expert or "",
                trace=trace,
            )

            session_manager.add_message(
                target_session.id, "user", request.message, result.specialist_used,
                user_id=user["id"],
            )
            session_manager.add_message(
                target_session.id, "assistant", result.content, result.specialist_used,
                metadata={"tool_calls": result.tool_calls} if result.tool_calls else {},
                user_id=user["id"],
            )

            return ChatResponse(
                success=result.success,
                content=result.content,
                model=result.specialist_used,
                confidence=result.confidence,
                reasoning=result.reasoning,
                session_id=target_session.id,
                error=result.error,
                tool_calls=result.tool_calls,
                expert=result.expert_id,
                expert_name=result.expert_name,
                expert_icon=result.expert_icon,
                trace_id=result.trace_id or trace.trace_id,
                trace=result.trace or trace.to_dict(),
            )
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return ChatResponse(
                success=False,
                content="",
                model="",
                confidence=0.0,
                reasoning="",
                error=str(e),
            )

    @app.post("/api/chat/stream")
    async def chat_stream(request: ChatRequest, http_request: Request):
        from .session import session_manager, build_session_context, build_dispatch_context
        from .run_trace import RunTrace
        import json as _json

        user = require_auth(http_request)
        disp = get_dispatcher()
        target_session = _resolve_chat_session(request.session_id, user)
        workdir = target_session.workdir or ""
        stream_trace = RunTrace().bind()

        context = build_session_context(
            target_session.messages,
            summary=target_session.summary,
            query=request.message,
            session_id=target_session.id,
        )
        dispatch_context = build_dispatch_context(
            target_session.messages,
            summary=target_session.summary,
        )

        async def event_generator():
            collected: list[str] = []
            model_name = ""
            tool_calls: list = []
            try:
                yield {"event": "session", "data": _json.dumps({"session_id": target_session.id})}
                result = disp.chat_stream(
                    user_input=request.message, workdir=workdir,
                    context=context, dispatch_context=dispatch_context,
                    forced_expert=request.expert or "",
                    trace=stream_trace,
                )
                async for event_data in result:
                    if event_data.get("event") == "token":
                        collected.append(event_data.get("data", ""))
                    elif event_data.get("event") == "done":
                        try:
                            done = _json.loads(event_data.get("data", "{}"))
                            model_name = done.get("model", "")
                            tool_calls = done.get("tool_calls", [])
                            done_content = done.get("content") or done.get("direct_answer") or ""
                            if done_content and not collected:
                                collected.append(done_content)
                        except Exception:
                            pass
                    yield event_data
                assistant_content = "".join(collected)
                session_manager.add_message(
                    target_session.id, "user", request.message, model_name,
                    user_id=user["id"],
                )
                session_manager.add_message(
                    target_session.id, "assistant", assistant_content, model_name,
                    metadata={"tool_calls": tool_calls} if tool_calls else {},
                    user_id=user["id"],
                )
            except Exception as e:
                logger.error(f"Chat stream error: {e}")
                yield {"event": "error", "data": str(e)}

        return EventSourceResponse(event_generator())

    @app.websocket("/ws/chat")
    async def websocket_chat(websocket: WebSocket):
        from .session import session_manager
        import json

        await websocket.accept()
        logger.info("[WS] Client connected")

        try:
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)

                if msg.get("type") == "chat":
                    message = msg.get("message", "")
                    session_id = msg.get("session_id")

                    if not message:
                        await websocket.send_json({"type": "error", "data": "Empty message"})
                        continue

                    target_session = None
                    if session_id:
                        target_session = session_manager.get_session(session_id)
                    if not target_session:
                        target_session = session_manager.get_current_session()
                        if not target_session:
                            target_session = session_manager.create_session()
                    session_manager.set_current_session(target_session.id)
                    workdir = getattr(target_session, 'workdir', None) or ""

                    disp = get_dispatcher()

                    await websocket.send_json({
                        "type": "session",
                        "data": {"session_id": target_session.id}
                    })

                    async for event_data in disp.chat_stream(
                        user_input=message, workdir=workdir,
                    ):
                        ws_msg = {
                            "type": event_data.get("event", "unknown"),
                            "data": event_data.get("data", ""),
                        }
                        await websocket.send_json(ws_msg)

                    session_manager.add_message(target_session.id, "user", message)

                elif msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

        except WebSocketDisconnect:
            logger.info("[WS] Client disconnected")
        except Exception as e:
            logger.error(f"[WS] Error: {e}")
            try:
                await websocket.send_json({"type": "error", "data": str(e)})
            except Exception:
                pass

    @app.post("/api/clear")
    async def clear_memory(http_request: Request, session_id: Optional[str] = None):
        from .session import session_manager

        user = require_auth(http_request)
        try:
            if session_id:
                session_manager.clear_session(session_id, user["id"])
            else:
                target = session_manager.get_current_session(user["id"])
                if target:
                    session_manager.clear_session(target.id, user["id"])
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.get("/api/sessions")
    async def list_sessions(http_request: Request):
        from .session import session_manager

        user = require_auth(http_request)
        return {"sessions": session_manager.list_sessions(user["id"])}

    @app.post("/api/sessions")
    async def create_session(http_request: Request, name: str = Body(""), workdir: str = Body("")):
        from .session import session_manager
        from pathlib import Path

        user = require_auth(http_request)
        session = session_manager.create_session(name, user_id=user["id"])
        if workdir:
            target = Path(workdir).expanduser().resolve()
            if target.is_dir():
                session.workdir = str(target)
                session_manager._save(session)
        return {"session": session.to_dict()}

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str, http_request: Request):
        user = require_auth(http_request)
        session = _require_user_session(session_id, user)
        return {"session": session.to_dict()}

    @app.put("/api/sessions/{session_id}")
    async def update_session(
        session_id: str,
        http_request: Request,
        name: str = Body(None),
        model: str = Body(None),
        workdir: str = Body(None),
    ):
        from .session import session_manager

        user = require_auth(http_request)
        session = session_manager.update_session(
            session_id, user["id"], name=name, model=model, workdir=workdir,
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"session": session.to_dict()}

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str, http_request: Request):
        from .session import session_manager

        user = require_auth(http_request)
        success = session_manager.delete_session(session_id, user["id"])
        return {"success": success}

    @app.post("/api/sessions/{session_id}/activate")
    async def activate_session(session_id: str, http_request: Request):
        from .session import session_manager

        user = require_auth(http_request)
        success = session_manager.set_current_session(session_id, user["id"])
        if success:
            db.set_user_preference(user["id"], "last_session_id", session_id)
        return {"success": success, "current_session": session_id}

    @app.get("/api/sessions/current")
    async def get_current_session(http_request: Request):
        from .session import session_manager

        user = require_auth(http_request)
        session = session_manager.get_current_session(user["id"])
        if not session:
            raise HTTPException(status_code=404, detail="No session found")
        return {"session": {"id": session.id, "name": session.name, "workdir": session.workdir}}

    @app.get("/api/skills")
    async def list_skills(http_request: Request, workdir: str = ""):
        from .skills.registry import SkillRegistry

        require_auth(http_request)
        registry = SkillRegistry(workdir=workdir)
        registry.scan()
        skills = [
            {
                "name": s.name,
                "description": s.description,
                "version": s.version,
                "source": s.source,
                "parameters": {
                    k: {
                        "type": v.type,
                        "description": v.description,
                        "required": v.required,
                        "default": v.default,
                    }
                    for k, v in s.parameters.items()
                },
            }
            for s in registry.skills.values()
        ]
        return {"skills": skills}

    # ── File system endpoints (sandboxed per-session) ──────────────

    @app.get("/api/sessions/{session_id}/workdir")
    async def get_workdir(session_id: str, http_request: Request):
        user = require_auth(http_request)
        session = _require_user_session(session_id, user)
        return {"workdir": session.workdir or ""}

    @app.put("/api/sessions/{session_id}/workdir")
    async def set_workdir(session_id: str, http_request: Request, path: str = Body(..., embed=True)):
        from .session import session_manager
        from pathlib import Path

        user = require_auth(http_request)
        _require_user_session(session_id, user)
        target = Path(path).expanduser().resolve()
        if not target.is_dir():
            raise HTTPException(400, f"Not a directory: {path}")
        session = session_manager.set_workdir(session_id, user["id"], str(target))
        return {"workdir": session.workdir}

    @app.get("/api/sessions/{session_id}/files")
    async def list_files(session_id: str, http_request: Request, path: str = "."):
        from .file_ops import session_workdir, list_files as _list_files

        user = require_auth(http_request)
        session = _require_user_session(session_id, user)
        root = session_workdir(session_id, session.workdir)
        try:
            return _list_files(root, path)
        except ValueError as e:
            raise HTTPException(403, str(e))

    @app.get("/api/sessions/{session_id}/files/content")
    async def read_file(session_id: str, http_request: Request, path: str):
        from .file_ops import session_workdir, read_file as _read_file

        user = require_auth(http_request)
        session = _require_user_session(session_id, user)
        root = session_workdir(session_id, session.workdir)
        try:
            return _read_file(root, path)
        except ValueError as e:
            raise HTTPException(403, str(e))
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except IsADirectoryError as e:
            raise HTTPException(400, str(e))

    @app.put("/api/sessions/{session_id}/files/content")
    async def write_file(session_id: str, http_request: Request, path: str = Body(...), content: str = Body(...)):
        from .file_ops import session_workdir, write_file as _write_file

        user = require_auth(http_request)
        session = _require_user_session(session_id, user)
        root = session_workdir(session_id, session.workdir)
        try:
            return _write_file(root, path, content)
        except ValueError as e:
            raise HTTPException(403, str(e))

    @app.post("/api/sessions/{session_id}/files/edit")
    async def edit_file(session_id: str, http_request: Request, path: str = Body(...), old_str: str = Body(...), new_str: str = Body(...)):
        from .file_ops import session_workdir, edit_file as _edit_file

        user = require_auth(http_request)
        session = _require_user_session(session_id, user)
        root = session_workdir(session_id, session.workdir)
        try:
            result = _edit_file(root, path, old_str, new_str)
            if result.get("status") == "error":
                raise HTTPException(400, result["message"])
            return result
        except ValueError as e:
            raise HTTPException(403, str(e))
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))

    @app.delete("/api/sessions/{session_id}/files")
    async def delete_file(session_id: str, http_request: Request, path: str = Body(...), recursive: bool = Body(False)):
        from .file_ops import session_workdir, delete_file as _delete_file

        user = require_auth(http_request)
        session = _require_user_session(session_id, user)
        root = session_workdir(session_id, session.workdir)
        try:
            result = _delete_file(root, path, recursive)
            if result.get("status") == "error":
                raise HTTPException(400, result["message"])
            return result
        except ValueError as e:
            raise HTTPException(403, str(e))
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))

    @app.post("/api/feedback")
    async def submit_feedback(
        rating: int = Body(..., ge=1, le=5), notes: str = Body(None)
    ):
        try:
            disp = get_dispatcher()
            disp.feedback(rating=rating, notes=notes)
            return {"success": True, "message": "反馈已记录"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.get("/api/status")
    async def api_status():
        try:
            disp = get_dispatcher()
            from .experts import experts_status
            from .model_pool import get_pool
            pool = get_pool()
            return {
                "model": disp.get_status().get("dispatcher", "unknown"),
                "specialist": disp.get_status().get("specialist", "none"),
                "memory_size": disp.get_status().get("memory_size", 0),
                "router": "omo",
                "experts": experts_status(pool),
            }
        except Exception as e:
            return {"error": str(e)}

    @app.get("/api/experts")
    async def list_experts(http_request: Request):
        require_auth(http_request)
        from .experts import experts_status, EXPERT_REGISTRY
        from .model_pool import get_pool
        pool = get_pool()
        return {
            "experts": experts_status(pool),
            "count": len(EXPERT_REGISTRY),
        }

    @app.get("/v1/models")
    async def list_models():
        from .config import get_config
        cfg = get_config()

        models = []
        if cfg.proxy.enabled:
            for name, ep in cfg.proxy.providers.items():
                if ep.api_key and ep.model:
                    models.append({
                        "id": ep.model,
                        "object": "model",
                        "owned_by": name,
                    })

        if not models:
            models = [
                {"id": "qwen2.5-coder-7b", "object": "model", "owned_by": "Qwen"},
                {"id": "gemini-2.0-flash", "object": "model", "owned_by": "Google"},
                {"id": "gemma-4-31b-it", "object": "model", "owned_by": "Google"},
            ]

        return ModelList(data=models)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        from .config import get_config
        cfg = get_config()

        if cfg.proxy.enabled:
            return await _proxy_chat(request)

        from .router import Provider

        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # Determine provider based on model name
        provider = None
        model_name = request.model.lower().split("/")[
            -1
        ]  # Handle "coolclaw/gemma-4-31b-it"
        cloud_keywords = ["gemma", "gemini", "google"]
        if any(keyword in model_name for keyword in cloud_keywords):
            provider = Provider.GOOGLE_AI

        router = get_router()

        if request.stream:
            return StreamingResponse(
                _stream_chat(request, messages, router, provider),
                media_type="text/event-stream",
            )

        try:
            response = router.chat(
                messages=messages,
                tools=request.tools if "gemini" in request.model.lower() else None,
                provider=provider,
                max_tokens=request.max_tokens or 2048,
                temperature=request.temperature or 0.7,
                stream=False,
                model=model_name,
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
                            tool_calls=response.tool_calls,
                        ),
                        finish_reason=response.finish_reason or "stop",
                    )
                ],
                usage=Usage(
                    prompt_tokens=response.usage.get("promptTokenCount", 0)
                    if response.usage
                    else 0,
                    completion_tokens=response.usage.get("candidatesTokenCount", 0)
                    if response.usage
                    else 0,
                    total_tokens=response.usage.get("totalTokenCount", response.usage.get("promptTokenCount", 0) + response.usage.get("candidatesTokenCount", 0)) if response.usage else 0,
                ),
            )
        except Exception as e:
            logger.error(f"Chat completion error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        if request.stream:
            return StreamingResponse(
                _stream_complete(request, get_router()),
                media_type="text/event-stream",
            )

        try:
            response = get_router().complete(
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
        return get_router().get_status()

    @app.post("/router/mode")
    async def set_router_mode(mode: str):
        if mode not in ("local_only", "cloud_only", "cloud_first", "auto"):
            raise HTTPException(status_code=400, detail="Invalid mode")
        get_router().set_mode(mode)
        return {"mode": mode}

    @app.post("/router/unload")
    async def unload_model():
        get_router().unload_local_model()
        return {"status": "unloaded"}

    @app.get("/proxy/status")
    async def proxy_status():
        return _get_proxy().get_status()

    @app.get("/proxy/health")
    async def proxy_health():
        proxy = _get_proxy()
        return {"enabled": proxy.config.proxy.enabled, "providers": list(proxy._providers.keys())}

    @app.get("/api/capabilities")
    async def list_capabilities():
        from .capabilities import register_all, CapabilityRegistry, CapabilityCategory

        register_all()
        capabilities = CapabilityRegistry.list_all()
        return {
            "capabilities": [
                {
                    "name": c.name,
                    "description": c.description,
                    "category": c.category.value,
                    "input_schema": c.input_schema,
                    "memory_mb": c.memory_mb,
                    "examples": c.examples,
                }
                for c in capabilities
            ]
        }

    @app.get("/api/tasks")
    async def list_tasks(status: Optional[str] = None):
        from .task_manager import task_manager, TaskStatus

        filter_status = None
        if status:
            try:
                filter_status = TaskStatus(status)
            except ValueError:
                filter_status = None
        tasks = task_manager.list_tasks(status=filter_status)
        return {"tasks": [t.to_dict() for t in tasks]}

    @app.post("/api/tasks")
    async def create_task(
        name: str = Body(...),
        description: str = Body(""),
        steps: list = Body([]),
        trigger: dict = Body({"type": "manual"}),
        output_template: str = Body(""),
        notification_enabled: bool = Body(False),
        notification_channels: list = Body([]),
    ):
        from .task_manager import task_manager

        task = task_manager.create_task(
            name=name,
            description=description,
            steps=steps,
            trigger=trigger,
            output_template=output_template,
            notification_enabled=notification_enabled,
            notification_channels=notification_channels,
        )
        return {"task": task.to_dict()}

    @app.get("/api/tasks/{task_id}")
    async def get_task(task_id: str):
        from .task_manager import task_manager

        task = task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return {"task": task.to_dict()}

    @app.put("/api/tasks/{task_id}")
    async def update_task(task_id: str, updates: dict = Body(...)):
        from .task_manager import task_manager

        task = task_manager.update_task(task_id, **updates)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return {"task": task.to_dict()}

    @app.delete("/api/tasks/{task_id}")
    async def delete_task(task_id: str):
        from .task_manager import task_manager

        success = task_manager.delete_task(task_id)
        return {"success": success}

    @app.post("/api/tasks/{task_id}/run")
    async def run_task(task_id: str):
        from .task_manager import task_manager
        import asyncio

        result = await task_manager.run_task(task_id)
        return result

    @app.post("/api/tasks/{task_id}/pause")
    async def pause_task(task_id: str):
        from .task_manager import task_manager

        success = task_manager.pause_task(task_id)
        return {"success": success}

    @app.post("/api/tasks/{task_id}/resume")
    async def resume_task(task_id: str):
        from .task_manager import task_manager

        success = task_manager.resume_task(task_id)
        return {"success": success}

    @app.get("/api/tasks/{task_id}/logs")
    async def get_task_logs(task_id: str, limit: int = 20):
        from .task_manager import task_manager

        logs = task_manager.get_execution_logs(task_id=task_id, limit=limit)
        return {"logs": logs}

    @app.get("/api/logs")
    async def get_all_logs(limit: int = 50):
        from .task_manager import task_manager

        logs = task_manager.get_execution_logs(limit=limit)
        return {"logs": logs}

    @app.get("/api/logs/{log_id}")
    async def get_log(log_id: str):
        from .task_manager import task_manager

        log = task_manager.get_log(log_id)
        if not log:
            raise HTTPException(status_code=404, detail="Log not found")
        return {"log": log}

    @app.get("/api/logs/{log_id}/output")
    async def get_log_output(log_id: str):
        from .task_manager import task_manager

        output_file = task_manager.get_output_file(log_id)
        if not output_file:
            raise HTTPException(status_code=404, detail="Output file not found")
        return FileResponse(
            output_file, media_type="text/plain", filename=f"output_{log_id}.txt"
        )

    @app.post("/api/tasks/{task_id}/settings")
    async def update_task_settings(
        task_id: str,
        notification_enabled: bool = Body(False),
        notification_channels: list = Body([]),
    ):
        from .task_manager import task_manager

        task = task_manager.update_task(
            task_id,
            notification_enabled=notification_enabled,
            notification_channels=notification_channels,
        )
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return {"task": task.to_dict()}

    @app.post("/api/capabilities/execute")
    async def execute_capability(capability: str = Body(...), params: dict = Body({})):
        from .capabilities import register_all, CapabilityRegistry

        register_all()
        result = await CapabilityRegistry.execute(capability, params)
        return {
            "success": result.success,
            "data": result.data,
            "error": result.error,
        }

    @app.post("/api/secrets")
    async def set_secret(key: str = Body(...), value: str = Body(...)):
        from .storage import secrets

        secrets.set(key, value)
        return {"success": True}

    @app.get("/api/secrets/{key}")
    async def get_secret(key: str):
        from .storage import secrets

        value = secrets.get(key)
        if value is None:
            raise HTTPException(status_code=404, detail="Secret not found")
        return {"key": key, "value": value}

    @app.delete("/api/secrets/{key}")
    async def delete_secret(key: str):
        from .storage import secrets

        success = secrets.delete(key)
        return {"success": success}

    @app.get("/api/knowledge")
    async def list_knowledge_documents():
        from .knowledge_base import knowledge_base

        docs = knowledge_base.list_documents()
        return {"documents": docs, "count": len(docs)}

    @app.post("/api/knowledge")
    async def add_knowledge_document(
        file_name: str = Body(...), content: str = Body(...)
    ):
        from .knowledge_base import knowledge_base

        try:
            doc_id = knowledge_base.add_document(name=file_name, content=content)
            docs = knowledge_base.list_documents()
            doc_info = next((d for d in docs if d["id"] == doc_id), None)
            return {
                "success": True,
                "id": doc_id,
                "name": file_name,
                "chunks": doc_info["chunks"] if doc_info else 0,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.delete("/api/knowledge/{doc_id}")
    async def delete_knowledge_document(doc_id: str):
        from .knowledge_base import knowledge_base

        success = knowledge_base.delete_document(doc_id)
        return {"success": success}

    @app.post("/api/knowledge/search")
    async def search_knowledge(query: str = Body(...), top_k: int = Body(5)):
        from .knowledge_base import knowledge_base

        results = knowledge_base.search(query, top_k=top_k)
        return {
            "results": [
                {"content": r.content, "score": r.score, "source": r.source}
                for r in results
            ],
            "count": len(results),
        }

    @app.post("/api/knowledge/rebuild")
    async def rebuild_knowledge_index():
        from .knowledge_base import knowledge_base

        knowledge_base.clear()
        return {"success": True, "message": "索引已重建"}

    # ── Auth Routes ──────────────────────────────────────────────────────

    @app.post("/api/auth/login")
    async def auth_login(request: Request):
        import json
        body = await request.json()
        username = body.get("username", "").strip()
        password = body.get("password", "")

        if not username or not password:
            raise HTTPException(400, "用户名和密码不能为空")

        user = db.authenticate_user(username, password)
        if not user:
            raise HTTPException(401, "用户名或密码错误")

        token = create_session_token(user["id"])
        response = JSONResponse({
            "success": True,
            "username": user["username"],
            "is_admin": bool(user["is_admin"]),
        })
        response.set_cookie(
            COOKIE_NAME, token,
            max_age=7 * 24 * 3600,
            httponly=True,
            samesite="lax",
            path="/",
        )
        return response

    @app.post("/api/auth/register")
    async def auth_register(request: Request):
        import json
        body = await request.json()
        username = body.get("username", "").strip()
        password = body.get("password", "")

        if not username or not password:
            raise HTTPException(400, "用户名和密码不能为空")
        if len(username) < 2:
            raise HTTPException(400, "用户名至少 2 个字符")
        if len(password) < 4:
            raise HTTPException(400, "密码至少 4 个字符")

        try:
            result = db.create_user(username, password)
        except (ValueError, Exception) as e:
            import sqlite3
            if isinstance(e, sqlite3.IntegrityError):
                raise HTTPException(400, "用户名已存在")
            raise HTTPException(409, str(e))

        token = create_session_token(result["id"])
        response = JSONResponse({"success": True, "username": username, "is_admin": False})
        response.set_cookie(
            COOKIE_NAME, token,
            max_age=7 * 24 * 3600,
            httponly=True,
            samesite="lax",
            path="/",
        )
        return response

    @app.post("/api/auth/logout")
    async def auth_logout(request: Request):
        token = request.cookies.get(COOKIE_NAME)
        if token:
            destroy_session_token(token)
        response = JSONResponse({"success": True})
        response.delete_cookie(COOKIE_NAME, path="/")
        return response

    @app.get("/api/auth/me")
    async def auth_me(request: Request):
        user = get_current_user(request)
        if not user:
            raise HTTPException(401, "Not authenticated")
        return {"id": user["id"], "username": user["username"], "is_admin": user["is_admin"]}

    # ── User Management (Admin) ─────────────────────────────────────────

    @app.get("/api/admin/users")
    async def admin_list_users(request: Request):
        require_admin(request)
        users = db.list_users()
        return {"users": users}

    @app.post("/api/admin/users")
    async def admin_create_user(request: Request):
        require_admin(request)
        body = await request.json()
        username = body.get("username", "").strip()
        password = body.get("password", "")
        is_admin = body.get("is_admin", False)

        if not username or not password:
            raise HTTPException(400, "用户名和密码不能为空")

        try:
            result = db.create_user(username, password, is_admin=is_admin)
        except ValueError as e:
            raise HTTPException(409, str(e))

        return {"success": True, "id": result["id"], "username": username}

    @app.put("/api/admin/users/{user_id}")
    async def admin_update_user(user_id: int, request: Request):
        require_admin(request)
        body = await request.json()
        updates = {}
        if "is_admin" in body:
            updates["is_admin"] = body["is_admin"]
        if "password" in body and body["password"]:
            updates["password"] = body["password"]
        if "display_name" in body:
            updates["display_name"] = body["display_name"]

        if updates:
            db.update_user(user_id, **updates)
        return {"success": True}

    @app.delete("/api/admin/users/{user_id}")
    async def admin_delete_user(user_id: int, request: Request):
        require_admin(request)
        # Don't let admin delete themselves
        current = require_admin(request)
        if current["id"] == user_id:
            raise HTTPException(400, "不能删除自己的账号")
        db.delete_user(user_id)
        return {"success": True}

    # ── System Settings (Admin) ─────────────────────────────────────────

    @app.get("/api/admin/settings")
    async def admin_get_settings(request: Request):
        require_admin(request)
        settings = db.get_all_settings()
        return {"settings": [{"key": k, "value": v} for k, v in settings.items()]}

    @app.put("/api/admin/settings")
    async def admin_update_settings(request: Request):
        require_admin(request)
        body = await request.json()
        key = body.get("key")
        value = body.get("value")
        if not key:
            raise HTTPException(400, "key 不能为空")
        db.set_setting(key, value)
        return {"success": True}

    # ── Model Configs (Admin) ────────────────────────────────────────────

    @app.get("/api/admin/models")
    async def admin_list_models(request: Request, provider_type: str | None = None):
        require_admin(request)
        models = db.list_model_configs(provider_type=provider_type)
        for m in models:
            if m.get("extra_config"):
                try:
                    m["extra_config"] = json.loads(m["extra_config"])
                except (json.JSONDecodeError, TypeError):
                    m["extra_config"] = {}
        return {"models": models}

    @app.post("/api/admin/models")
    async def admin_create_model(request: Request):
        require_admin(request)
        body = await request.json()
        provider_type = body.get("provider_type", "openai")
        model_name = body.get("model_name", "").strip()
        if not model_name:
            raise HTTPException(400, "模型名称不能为空")
        result = db.create_model_config(
            provider_type=provider_type,
            model_name=model_name,
            display_name=body.get("display_name", ""),
            api_key=body.get("api_key", ""),
            base_url=body.get("base_url", ""),
            role=body.get("role", "general"),
            priority=body.get("priority", 10),
            rpd=body.get("rpd", 0),
            rpm=body.get("rpm", 0),
            tpm=body.get("tpm", 0),
            is_active=body.get("is_active", True),
            extra_config=body.get("extra_config", {}),
        )
        from .model_pool import reload_pool
        reload_pool()
        return {"success": True, "model": result}

    @app.put("/api/admin/models/{model_id}")
    async def admin_update_model(model_id: int, request: Request):
        require_admin(request)
        body = await request.json()
        result = db.update_model_config(model_id, **body)
        if not result:
            raise HTTPException(404, "模型配置不存在")
        if result.get("extra_config"):
            try:
                result["extra_config"] = json.loads(result["extra_config"])
            except (json.JSONDecodeError, TypeError):
                result["extra_config"] = {}
        from .model_pool import reload_pool
        reload_pool()
        return {"success": True, "model": result}

    @app.delete("/api/admin/models/{model_id}")
    async def admin_delete_model(model_id: int, request: Request):
        require_admin(request)
        db.delete_model_config(model_id)
        from .model_pool import reload_pool
        reload_pool()
        return {"success": True}

    @app.post("/api/admin/models/{model_id}/test")
    async def admin_test_model(model_id: int, request: Request):
        require_admin(request)
        model = db.get_model_config(model_id)
        if not model:
            raise HTTPException(404, "模型配置不存在")
        provider_type = model["provider_type"]
        try:
            if provider_type == "local":
                return {"success": True, "message": "本地模型需手动验证加载"}
            api_key = model.get("api_key", "")
            base_url = model.get("base_url", "")
            model_name = model.get("model_name", "")
            if not api_key:
                return {"success": False, "message": "未配置 API Key"}
            import httpx as _httpx
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            test_url = (base_url.rstrip("/") if base_url else "https://api.openai.com/v1") + "/models"
            async with _httpx.AsyncClient(proxy=None, trust_env=False, timeout=10) as client:
                resp = await client.get(test_url, headers=headers)
            if resp.status_code < 300:
                return {"success": True, "message": f"连接成功 ({resp.status_code})"}
            return {"success": False, "message": f"连接失败 ({resp.status_code}): {resp.text[:200]}"}
        except Exception as e:
            return {"success": False, "message": f"连接错误: {e}"}

    # ── Config & Model Catalog (Admin) ─────────────────────────────────

    @app.get("/api/admin/config")
    async def admin_get_config(request: Request):
        require_admin(request)
        from .config_manager import get_admin_config
        return get_admin_config()

    @app.put("/api/admin/config")
    async def admin_put_config(request: Request):
        require_admin(request)
        from .config_manager import update_admin_config
        body = await request.json()
        updates = body.get("config") or body
        return update_admin_config(updates)

    @app.get("/api/admin/config/yaml")
    async def admin_get_config_yaml(request: Request):
        require_admin(request)
        from .config_manager import read_config_yaml_text
        return read_config_yaml_text()

    @app.post("/api/admin/config/yaml/validate")
    async def admin_validate_config_yaml(request: Request):
        require_admin(request)
        from .config_manager import validate_yaml_text
        body = await request.json()
        return validate_yaml_text(body.get("content", ""))

    @app.post("/api/admin/config/yaml/format")
    async def admin_format_config_yaml(request: Request):
        require_admin(request)
        from .config_manager import format_yaml_text, validate_yaml_text
        body = await request.json()
        content = body.get("content", "")
        validation = validate_yaml_text(content)
        if not validation["valid"]:
            raise HTTPException(400, validation["error"])
        try:
            formatted = format_yaml_text(content)
        except RuntimeError as e:
            raise HTTPException(501, str(e))
        return {"content": formatted, "validation": validate_yaml_text(formatted)}

    @app.put("/api/admin/config/yaml")
    async def admin_put_config_yaml(request: Request):
        require_admin(request)
        from .config_manager import save_config_yaml_text
        from .server_control import reload_runtime, schedule_process_restart
        body = await request.json()
        content = body.get("content", "")
        # `restart` legacy flag → hot reload; `reload` explicit; `process_restart` for full execv
        hot_reload = body.get("reload", body.get("restart", True))
        process_restart = body.get("process_restart", False)
        try:
            result = save_config_yaml_text(content, apply_runtime=False)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if hot_reload:
            result["reload"] = reload_runtime()
        if process_restart:
            result["process_restarting"] = schedule_process_restart(delay=1.2)
        else:
            result["process_restarting"] = False
        return result

    @app.post("/api/admin/server/reload")
    async def admin_reload_server(request: Request):
        require_admin(request)
        from .server_control import reload_runtime
        return {"success": True, **reload_runtime()}

    @app.post("/api/admin/server/restart")
    async def admin_restart_server(request: Request):
        require_admin(request)
        from .server_control import schedule_process_restart
        return {
            "success": True,
            "process_restarting": schedule_process_restart(delay=0.8),
            "message": "将完整重启进程（一般无需使用，优先用热加载）",
        }

    @app.get("/api/admin/models/catalog/local")
    async def admin_local_catalog(request: Request):
        require_admin(request)
        from .model_catalog import get_local_catalog
        return {"models": get_local_catalog()}

    @app.get("/api/admin/models/catalog/cloud")
    async def admin_cloud_catalog(request: Request):
        require_admin(request)
        from .model_catalog import get_cloud_catalog
        return {"providers": get_cloud_catalog()}

    @app.get("/api/admin/models/installed")
    async def admin_installed_models(request: Request):
        require_admin(request)
        from .config import get_config
        from .config_manager import scan_installed_models
        cfg = get_config()
        models_dir = cfg.paths.models_dir
        models = scan_installed_models(models_dir, include_config=True)
        return {
            "models_dir": str(Path(models_dir).expanduser().resolve()) if models_dir else "",
            "models": models,
            "count": len(models),
        }

    @app.get("/api/admin/models/search")
    async def admin_search_models(request: Request, q: str = "", limit: int = 20):
        require_admin(request)
        from .huggingface_search import search_gguf_models
        from .model_catalog import get_local_catalog
        q = (q or "").strip()
        if not q:
            return {"results": [], "source": "empty"}
        # Local catalog filter
        local_hits = [
            m for m in get_local_catalog()
            if q.lower() in m["name"].lower()
            or q.lower() in m["description"].lower()
            or any(q.lower() in t for t in m.get("tags", []))
        ]
        # HuggingFace online search
        hf_hits = search_gguf_models(q, limit=limit)
        return {
            "query": q,
            "local_catalog": local_hits,
            "huggingface": hf_hits,
        }

    @app.get("/api/admin/models/search/{repo_id:path}/files")
    async def admin_search_repo_files(repo_id: str, request: Request):
        require_admin(request)
        from .huggingface_search import get_repo_gguf_files
        files = get_repo_gguf_files(repo_id)
        return {"repo_id": repo_id, "files": files}

    @app.post("/api/admin/models/download")
    async def admin_start_download(request: Request):
        require_admin(request)
        from .config import get_config
        from .model_download import download_manager
        body = await request.json()
        model_id = body.get("model_id", "")
        quant = body.get("quant", "")
        repo_id = body.get("repo_id", "")
        filename = body.get("filename", "")
        cfg = get_config()
        models_dir = body.get("models_dir") or cfg.paths.models_dir
        if not models_dir:
            raise HTTPException(400, "请先在系统配置中设置模型存放目录 (paths.models_dir)")
        try:
            if repo_id and filename:
                task = download_manager.start_hf_download(repo_id, filename, models_dir)
            elif model_id:
                task = download_manager.start_download(model_id, quant, models_dir)
            else:
                raise HTTPException(400, "需要 model_id 或 repo_id+filename")
            return {"success": True, "task": task.to_dict()}
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/admin/models/downloads")
    async def admin_list_downloads(request: Request):
        require_admin(request)
        from .model_download import download_manager
        return {"tasks": download_manager.list_tasks()}

    @app.delete("/api/admin/models/downloads/{task_id}")
    async def admin_cancel_download(task_id: str, request: Request):
        require_admin(request)
        from .model_download import download_manager
        ok = download_manager.cancel_download(task_id)
        return {"success": ok}

    @app.post("/api/admin/models/register-local")
    async def admin_register_local_model(request: Request):
        require_admin(request)
        from .config_manager import register_local_model_in_config, register_local_model_by_path
        body = await request.json()
        model_id = body.get("model_id", "")
        file_path = body.get("file_path", "")
        set_default = body.get("set_default", False)
        role = body.get("role", "general")
        if not file_path:
            raise HTTPException(400, "file_path 不能为空")
        if not Path(file_path).is_file():
            raise HTTPException(400, f"文件不存在: {file_path}")
        try:
            if model_id:
                result = register_local_model_in_config(model_id, file_path, set_default)
            else:
                result = register_local_model_by_path(file_path, role=role, set_default=set_default)
        except ValueError as e:
            raise HTTPException(400, str(e))
        from .model_catalog import find_local_model
        entry = find_local_model(model_id) if model_id else None
        existing = db.list_model_configs(provider_type="local")
        if not any(m.get("model_name") == file_path for m in existing):
            db.create_model_config(
                provider_type="local",
                model_name=file_path,
                display_name=entry.name if entry else Path(file_path).stem,
                role=entry.role if entry else role,
                priority=5,
                is_active=True,
                extra_config=entry.llama_params if entry else {},
            )
        from .model_pool import reload_pool
        reload_pool()
        return {"success": True, **result}

    @app.get("/api/admin/models/routing")
    async def admin_model_routing(request: Request):
        require_admin(request)
        from .model_pool import get_pool
        pool = get_pool()
        status = pool.routing_status()
        try:
            status["proxy"] = _get_proxy().get_status()
        except Exception:
            status["proxy"] = {}
        from .experts import experts_status
        status["experts"] = experts_status(pool)
        return status

    @app.get("/api/admin/experts")
    async def admin_experts(request: Request):
        require_admin(request)
        from .experts import experts_status, format_experts_for_dispatcher, EXPERT_REGISTRY
        from .config import get_config
        from .model_pool import get_pool
        pool = get_pool()
        cfg = get_config()
        providers = list(cfg.proxy.providers.keys()) if cfg.proxy.enabled else []
        return {
            "experts": experts_status(pool),
            "catalog_markdown": format_experts_for_dispatcher(),
            "count": len(EXPERT_REGISTRY),
            "bindings": cfg.routing.expert_bindings or {},
            "providers": providers,
        }

    @app.put("/api/admin/experts/bindings")
    async def admin_put_expert_bindings(request: Request):
        require_admin(request)
        from .config_manager import update_admin_config
        body = await request.json()
        bindings = body.get("bindings") or {}
        update_admin_config({"routing": {"expert_bindings": bindings}})
        from .model_pool import reload_pool
        reload_pool()
        return {"success": True, "bindings": bindings}

    @app.post("/api/admin/models/reload-pool")
    async def admin_reload_model_pool(request: Request):
        require_admin(request)
        from .model_pool import reload_pool
        pool = reload_pool()
        return {"success": True, "routing": pool.routing_status()}

    @app.post("/api/admin/models/setup-cloud")
    async def admin_setup_cloud_provider(request: Request):
        require_admin(request)
        from .model_catalog import find_cloud_provider
        from .config_manager import load_yaml_raw, save_yaml_raw, update_admin_config
        from .config import reset_config
        body = await request.json()
        provider_id = body.get("provider_id", "")
        api_key = body.get("api_key", "")
        model_id = body.get("model_id", "")
        set_default = body.get("set_default", True)
        if not provider_id or not api_key or not model_id:
            raise HTTPException(400, "provider_id, api_key, model_id 不能为空")

        provider = find_cloud_provider(provider_id)
        if not provider:
            raise HTTPException(404, "Provider 不存在")

        model_entry = next((m for m in provider.models if m.id == model_id), None)
        if not model_entry:
            raise HTTPException(404, "模型不存在")

        raw = load_yaml_raw()
        raw.setdefault("proxy", {}).setdefault("providers", {})[provider_id] = {
            "api_key": api_key,
            "base_url": provider.default_base_url,
            "model": model_id,
            "timeout": 180,
        }
        raw.setdefault("proxy", {})["enabled"] = True
        if set_default:
            raw["proxy"]["default_provider"] = provider_id
            raw.setdefault("models", {}).setdefault("cloud", {})["default"] = model_id

        rl = raw.setdefault("proxy", {}).setdefault("rate_limits", {}).setdefault(provider_id, {})
        if model_entry.rpm:
            rl["rpm"] = model_entry.rpm
        if model_entry.rpd:
            rl["rpd"] = model_entry.rpd

        save_yaml_raw(raw)
        reset_config()

        db.create_model_config(
            provider_type=provider.provider_type,
            model_name=model_id,
            display_name=provider.name,
            api_key=api_key,
            base_url=provider.default_base_url,
            role="general",
            priority=5 if model_entry.recommended else 10,
            rpm=model_entry.rpm,
            rpd=model_entry.rpd,
            is_active=True,
            extra_config={"catalog_provider": provider_id},
        )
        from .model_pool import reload_pool
        reload_pool()
        return {"success": True, "provider": provider_id, "model": model_id}

    # ── User Preferences ────────────────────────────────────────────────

    @app.get("/api/user/preferences")
    async def get_user_preferences(request: Request):
        user = require_auth(request)
        prefs = db.get_all_user_preferences(user["id"])
        return {"preferences": [{"key": k, "value": v} for k, v in prefs.items()]}

    @app.put("/api/user/preferences")
    async def update_user_preferences(request: Request):
        user = require_auth(request)
        body = await request.json()
        key = body.get("key")
        value = body.get("value")
        if not key:
            raise HTTPException(400, "key 不能为空")
        db.set_user_preference(user["id"], key, value)
        return {"success": True}

    # ── Static Files ────────────────────────────────────────────────────

    if os.path.exists(_web_dir):
        app.mount("/", StaticFiles(directory=_web_dir, html=True), name="web")
        logger.info(f"Serving static files from: {_web_dir}")

    return app


async def _proxy_chat(request: ChatCompletionRequest):
    from .proxy import ProxyRequest

    proxy = _get_proxy()
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    proxy_req = ProxyRequest(
        model=request.model,
        messages=messages,
        max_tokens=request.max_tokens or 2048,
        temperature=request.temperature or 0.7,
        stream=bool(request.stream),
        tools=[t.model_dump() for t in request.tools] if request.tools else None,
    )

    if request.stream:
        return StreamingResponse(
            proxy.chat_stream(proxy_req),
            media_type="text/event-stream",
        )

    result = await proxy.chat(proxy_req)
    if not result.choices:
        raise HTTPException(status_code=429, detail="Rate limit exceeded, try again later")

    choice = result.choices[0]
    msg = choice.get("message", {})
    return ChatCompletionResponse(
        id=result.id,
        created=result.created,
        model=result.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=msg.get("content"),
                    tool_calls=msg.get("tool_calls"),
                ),
                finish_reason=choice.get("finish_reason", "stop"),
            )
        ],
        usage=Usage(
            prompt_tokens=result.usage.get("prompt_tokens", 0),
            completion_tokens=result.usage.get("completion_tokens", 0),
            total_tokens=result.usage.get("total_tokens", 0),
        ),
    )


async def _stream_chat(
    request: ChatCompletionRequest,
    messages: list[dict],
    router: "Router",
    provider: Optional[Provider] = None,
) -> AsyncIterator[str]:
    import json
    import time

    chunk_id = generate_id()

    model_name = request.model.lower().split("/")[-1]
    try:
        response = router.chat(
            messages=messages,
            tools=request.tools,
            provider=provider,
            max_tokens=request.max_tokens or 2048,
            temperature=request.temperature or 0.7,
            stream=True,
            model=model_name,
        )

        # Send initial role chunk
        initial = json.dumps(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
        )
        yield f"data: {initial}\n\n"

        for chunk in response.chunks:
            if hasattr(chunk, "content") and chunk.content:
                data = json.dumps(
                    {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk.content},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                yield f"data: {data}\n\n"
            elif hasattr(chunk, "tool_call") and chunk.tool_call:
                data = json.dumps(
                    {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "id": chunk.tool_call.get("id", ""),
                                            "type": "function",
                                            "function": {
                                                "name": chunk.tool_call.get("name", ""),
                                                "arguments": json.dumps(
                                                    chunk.tool_call.get("args", {})
                                                ),
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                yield f"data: {data}\n\n"
            elif hasattr(chunk, "tool_calls") and chunk.tool_calls:
                data = json.dumps(
                    {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"tool_calls": chunk.tool_calls},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                yield f"data: {data}\n\n"

        final = json.dumps(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        yield f"data: {final}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Stream error: {e}")
        error_response = json.dumps(
            {"error": {"message": str(e), "type": "internal_error"}}
        )
        yield f"data: {error_response}\n\n"


async def _stream_complete(
    request: CompletionRequest,
    router: "Router",  # type: ignore # noqa: F821
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
