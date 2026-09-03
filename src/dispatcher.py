import asyncio
import concurrent.futures
import logging
import json
import os
import re
from typing import Optional, Iterator
from dataclasses import dataclass, field


def _run_async(coro):
    """Run an async coroutine from sync code, safe to call inside a running event loop."""
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        def _run():
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
        future = pool.submit(_run)
        return future.result(timeout=120)
from datetime import datetime

from .local_llm import LocalLLM, CompletionChunk, get_cached_local_llm
from .models import (
    ModelSpec,
    MODEL_REGISTRY,
    get_available_model,
    get_fallback_model,
    get_dispatcher_model,
    ModelRole,
)
from .memory import ConversationMemory, Message
from .protocol import (
    DispatchDecision,
    DispatchAction,
    TaskRequest,
    TaskResponse,
    DISPATCHER_SYSTEM_PROMPT,
    SPECIALIST_SYSTEM_PROMPT_TEMPLATE,
)


logger = logging.getLogger(__name__)


class UnifiedLLMProxy:
    def __init__(
        self,
        model_key: str,
        is_dispatcher: bool = False,
        role_hint: str = "",
        preferred_provider: str = "",
    ):
        self.model_key = model_key
        self.is_dispatcher = is_dispatcher
        self.role_hint = role_hint
        self.preferred_provider = preferred_provider
        self._current_model = model_key
        self._local_llm = None
        self._use_cloud = False
        self._mode = "auto"
        
        # Check routing mode
        from .config import get_config
        try:
            config = get_config()
            self._mode = config.routing.mode
            if self._mode in ("cloud_first", "cloud_only"):
                self._use_cloud = True
        except Exception:
            pass

    def _get_model_chain(self):
        from .model_pool import get_pool, registry_role_to_pool
        from .models import MODEL_REGISTRY
        pool = get_pool()
        role = self.role_hint or None
        if not role and not self.is_dispatcher:
            spec = MODEL_REGISTRY.get(self.model_key)
            if spec:
                role = registry_role_to_pool(spec.role)
        if self.is_dispatcher:
            return pool.dispatcher_candidates(self._mode)
        return pool.resolve_chain(
            target_key=self.model_key,
            role_hint=role,
            routing_mode=self._mode,
            preferred_provider=self.preferred_provider or None,
        )

    def _inference_phases(self) -> tuple[str, ...]:
        """Order of local vs cloud attempts.

        auto: dispatcher (fast) → local first; specialists/agent → cloud first.
        """
        if self._mode == "local_only":
            return ("local",)
        if self._mode == "cloud_only":
            return ("cloud",)
        if self._mode == "cloud_first" or self._use_cloud:
            return ("cloud", "local")
        if self._mode == "auto":
            return ("local", "cloud") if self.is_dispatcher else ("cloud", "local")
        return ("local", "cloud")

    def _prefers_cloud_first(self) -> bool:
        return self._inference_phases()[0] == "cloud"

    def _run_cloud_chain(
        self,
        chain: list,
        *,
        messages: list[dict] | None = None,
        prompt: str | None = None,
        tools: list | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stream: bool = False,
    ):
        from .proxy import ProxyRequest

        cloud_chain = [c for c in chain if c.source == "cloud"]
        if not cloud_chain:
            return None
        proxy = self._get_proxy()
        if messages is not None:
            proxy_req = ProxyRequest(
                model=cloud_chain[0].model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
                tools=tools,
            )
        else:
            proxy_req = ProxyRequest(
                model=cloud_chain[0].model_name,
                messages=[{"role": "user", "content": prompt or ""}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
            )
        if stream:
            return self._proxy_stream_candidates(proxy, cloud_chain, proxy_req)
        result = _run_async(proxy.chat_with_candidates(cloud_chain, proxy_req))
        if not result.choices:
            return None
        choice = result.choices[0]
        msg = choice.get("message", {})
        self._current_model = cloud_chain[0].name
        return CompletionChunk(
            content=msg.get("content", ""),
            is_final=True,
            tool_calls=msg.get("tool_calls"),
            finish_reason=choice.get("finish_reason", "") or "",
        )

    def _run_local_chain(
        self,
        chain: list,
        *,
        messages: list[dict] | None = None,
        prompt: str | None = None,
        tools: list | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list | None = None,
        stream: bool = False,
    ):
        local_chain = [c for c in chain if c.source in ("local", "registry")]
        last_err = None
        for cand in local_chain:
            try:
                self._init_local_for_candidate(cand)
                if messages is not None:
                    return self._local_llm.chat(
                        messages, tools, max_tokens, temperature, top_p, stop, stream,
                    )
                return self._local_llm.complete(
                    prompt, max_tokens, temperature, top_p, stop, stream,
                )
            except Exception as e:
                last_err = e
                logger.warning("[POOL] local %s failed: %s", cand.name, e)
                self._local_llm = None
        if last_err and self._mode == "local_only":
            raise last_err
        return None

    def complete(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
        stream: bool = False,
    ) -> CompletionChunk | Iterator[CompletionChunk]:
        chain = self._get_model_chain()
        phases = self._inference_phases()
        last_err = None
        for phase in phases:
            if phase == "local" and self._mode == "cloud_only":
                continue
            if phase == "cloud" and self._mode == "local_only":
                continue
            try:
                if phase == "cloud":
                    result = self._run_cloud_chain(
                        chain, prompt=prompt, max_tokens=max_tokens,
                        temperature=temperature, stream=stream,
                    )
                else:
                    result = self._run_local_chain(
                        chain, prompt=prompt, max_tokens=max_tokens,
                        temperature=temperature, top_p=top_p, stop=stop, stream=stream,
                    )
                if result is not None:
                    return result
            except Exception as e:
                last_err = e
                logger.error("[POOL] %s phase failed: %s", phase, e)
                if self._mode == "cloud_only" and phase == "cloud":
                    raise
        if self._local_llm is None:
            self._init_local()
        return self._local_llm.complete(prompt, max_tokens, temperature, top_p, stop, stream)

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
        chain = self._get_model_chain()
        phases = self._inference_phases()
        for phase in phases:
            if phase == "local" and self._mode == "cloud_only":
                continue
            if phase == "cloud" and self._mode == "local_only":
                continue
            try:
                if phase == "cloud":
                    result = self._run_cloud_chain(
                        chain, messages=messages, tools=tools,
                        max_tokens=max_tokens, temperature=temperature, stream=stream,
                    )
                else:
                    result = self._run_local_chain(
                        chain, messages=messages, tools=tools,
                        max_tokens=max_tokens, temperature=temperature,
                        top_p=top_p, stop=stop, stream=stream,
                    )
                if result is not None:
                    return result
            except Exception as e:
                logger.error("[POOL] %s chat failed: %s", phase, e)
                if self._mode == "cloud_only" and phase == "cloud":
                    raise
        if self._local_llm is None:
            self._init_local()
        return self._local_llm.chat(messages, tools, max_tokens, temperature, top_p, stop, stream)

    def _get_proxy(self):
        from .api import _get_proxy
        return _get_proxy()

    def _proxy_stream_candidates(self, proxy, cloud_chain, proxy_req):
        import json as _json

        async def _collect():
            chunks = []
            result = await proxy.chat_with_candidates(cloud_chain, proxy_req)
            if result.choices:
                content = result.choices[0].get("message", {}).get("content", "")
                if content:
                    chunks.append(CompletionChunk(content=content))
            chunks.append(CompletionChunk(content="", is_final=True))
            return chunks

        collected = _run_async(_collect())
        return iter(collected)

    def _proxy_stream(self, proxy, proxy_req):
        import json as _json

        async def _collect():
            chunks = []
            async for raw in proxy.chat_stream(proxy_req):
                if raw.startswith("data: ") and not raw.strip().endswith("[DONE]"):
                    try:
                        chunk_data = _json.loads(raw[6:].strip())
                        choices = chunk_data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            if delta.get("content"):
                                chunks.append(CompletionChunk(content=delta["content"]))
                            if delta.get("tool_calls"):
                                chunks.append(CompletionChunk(content="", tool_calls=delta["tool_calls"]))
                    except _json.JSONDecodeError:
                        pass
            chunks.append(CompletionChunk(content="", is_final=True))
            return chunks

        collected = _run_async(_collect())
        return iter(collected)

    def _init_local_for_candidate(self, cand):
        from .local_llm import LocalLLM
        path = cand.path or cand.model_name
        if not path or not os.path.isfile(path):
            raise ValueError(f"Local model not found: {path}")
        extra = cand.extra_config or {}
        self._local_llm = get_cached_local_llm(
            path,
            n_ctx=extra.get("n_ctx", 4096),
            n_gpu_layers=extra.get("n_gpu_layers", -1),
            n_threads=extra.get("n_threads", 6),
            n_batch=extra.get("n_batch", 512),
            flash_attn=extra.get("flash_attn", True),
            cache_type_k=extra.get("cache_type_k", "q4_0"),
            cache_type_v=extra.get("cache_type_v", "q4_0"),
            low_vram=extra.get("low_vram", False),
        )
        self._current_model = cand.name
        self.model_key = cand.registry_key or cand.key

    def _init_local(self):
        from .local_llm import LocalLLM
        n_ctx = 4096
        
        if self.is_dispatcher:
            from .models import get_dispatcher_model
            model_spec = get_dispatcher_model()
            path = model_spec.path
            n_ctx = model_spec.n_ctx
        else:
            from .models import get_available_model
            model_spec = get_available_model(self.model_key)
            path = model_spec.path if model_spec else ""
            if model_spec:
                n_ctx = model_spec.n_ctx
            
        if not path:
            raise ValueError(f"No local path for model {self.model_key}")
            
        self._local_llm = LocalLLM(
            model_path=path,
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            n_threads=6,
            n_batch=512,
            flash_attn=True,
            cache_type_k="q4_0",
            cache_type_v="q4_0",
        )

    def unload(self):
        if self._local_llm:
            self._local_llm.unload()
            self._local_llm = None

    def get_context_window(self) -> int:
        if self._use_cloud:
            return 131072
        if self._local_llm is None:
            self._init_local()
        return self._local_llm.get_context_window()

    def is_loaded(self) -> bool:
        if self._use_cloud:
            return True
        return self._local_llm is not None and self._local_llm.is_loaded()


@dataclass
class DispatcherResult:
    content: str
    specialist_used: str
    confidence: float
    success: bool
    reasoning: str = ""
    error: Optional[str] = None
    tool_calls: Optional[list] = None
    expert_id: str = ""
    expert_name: str = ""
    expert_icon: str = ""
    trace_id: str = ""
    trace: Optional[dict] = None


@dataclass
class DispatcherConfig:
    dispatcher_model_key: str = "qwen2.5-3b"
    memory_path: Optional[str] = None
    auto_save: bool = True
    enable_feedback: bool = True
    max_retries: int = 2
    retry_delay: float = 1.0
    fallback_chain: list[str] = None

    def __post_init__(self):
        if self.fallback_chain is None:
            self.fallback_chain = ["qwen2.5-coder-7b", "qwen2.5-3b"]


@dataclass
class PerformanceRecord:
    model_key: str
    success_count: int = 0
    failure_count: int = 0
    total_uses: int = 0
    total_duration_ms: int = 0
    avg_duration_ms: float = 0.0
    last_used: Optional[str] = None
    health_score: float = 1.0

    @property
    def success_rate(self) -> float:
        if self.total_uses == 0:
            return 0.5
        return self.success_count / self.total_uses

    @property
    def avg_response_time(self) -> float:
        if self.total_uses == 0:
            return 0.0
        return self.total_duration_ms / self.total_uses


class SmartDispatcher:
    def __init__(
        self,
        dispatcher_llm: Optional[LocalLLM] = None,
        specialist_llm: Optional[LocalLLM] = None,
        config: Optional[DispatcherConfig] = None,
    ):
        self.config = config or DispatcherConfig()
        self._dispatcher_llm = dispatcher_llm
        self._specialist_llm = specialist_llm
        self._memory = ConversationMemory()
        self._performance: dict[str, PerformanceRecord] = {}
        self._load_performance()

        if self.config.memory_path:
            self._memory = ConversationMemory.load(self.config.memory_path)

    def _load_performance(self):
        perf_path = self._get_perf_path()
        if os.path.exists(perf_path):
            try:
                import json

                with open(perf_path) as f:
                    data = json.load(f)
                    for key, values in data.items():
                        self._performance[key] = PerformanceRecord(
                            model_key=key, **values
                        )
                logger.info(f"Loaded performance data: {len(self._performance)} models")
            except Exception as e:
                logger.warning(f"Failed to load performance: {e}")

    def _save_performance(self):
        if not self.config.enable_feedback:
            return
        perf_path = self._get_perf_path()
        try:
            data = {
                key: {
                    "success_count": rec.success_count,
                    "failure_count": rec.failure_count,
                    "total_uses": rec.total_uses,
                }
                for key, rec in self._performance.items()
            }
            os.makedirs(os.path.dirname(perf_path), exist_ok=True)
            with open(perf_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save performance: {e}")

    def _get_perf_path(self) -> str:
        base = os.path.dirname(self.config.memory_path or "")
        return os.path.join(base, "performance.json")

    def _record_performance(self, model_key: str, success: bool, duration_ms: int = 0):
        if not self.config.enable_feedback:
            return
        if model_key not in self._performance:
            self._performance[model_key] = PerformanceRecord(model_key=model_key)
        rec = self._performance[model_key]
        rec.total_uses += 1
        rec.total_duration_ms += duration_ms
        rec.avg_duration_ms = rec.total_duration_ms / rec.total_uses
        rec.last_used = datetime.now().isoformat()

        if success:
            rec.success_count += 1
            rec.health_score = min(1.0, rec.health_score + 0.05)
        else:
            rec.failure_count += 1
            rec.health_score = max(0.0, rec.health_score - 0.1)

        self._save_performance()

    def _get_model_performance(self, model_key: str) -> float:
        rec = self._performance.get(model_key)
        return rec.success_rate if rec else 0.5

    def _get_best_available_model(
        self, preferred_key: str
    ) -> tuple[str, Optional[LocalLLM]]:
        if preferred_key:
            llm = self._load_specialist(preferred_key)
            if llm:
                rec = self._performance.get(preferred_key)
                if not rec or rec.health_score > 0.3:
                    return preferred_key, llm

        for model_key in self.config.fallback_chain:
            if model_key == preferred_key:
                continue
            llm = self._load_specialist(model_key)
            if llm:
                rec = self._performance.get(model_key)
                if not rec or rec.health_score > 0.3:
                    return model_key, llm

        return preferred_key, self._load_specialist(preferred_key)

    def _retry_with_backoff(self, func, *args, **kwargs):
        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (2**attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s..."
                    )
                    import time

                    time.sleep(delay)

        raise last_error

    @staticmethod
    def _build_extra_server(workdir: str):
        try:
            from .config import get_config
            from .mcp import create_combined_server
            cfg = get_config()
            proxy = cfg.network.https_proxy or cfg.network.http_proxy
            server = create_combined_server(workdir=workdir, project_path=workdir, proxy=proxy)
            logger.info("[MCP] Created combined server with proxy=%s", bool(proxy))
            return server
        except Exception as e:
            logger.warning("[MCP] Failed to create combined server: %s", e)
            return None

    def _ensure_dispatcher_loaded(self):
        if self._dispatcher_llm is None:
            from .model_pool import get_pool
            from .config import get_config
            pool = get_pool()
            mode = get_config().routing.mode
            dispatchers = pool.dispatcher_candidates(mode)
            if dispatchers:
                first = dispatchers[0]
                key = first.registry_key or first.key
                self._dispatcher_llm = UnifiedLLMProxy(
                    model_key=key, is_dispatcher=True, role_hint="fast",
                )
                logger.info(f"Dispatcher proxy loaded: {first.name} (pool)")
            else:
                model_spec = get_dispatcher_model()
                self._dispatcher_llm = UnifiedLLMProxy(
                    model_key=model_spec.key, is_dispatcher=True,
                )
                logger.info(f"Dispatcher proxy loaded for: {model_spec.name}")

    def _load_specialist(self, model_key: str) -> Optional[UnifiedLLMProxy]:
        from .model_pool import get_pool
        pool = get_pool()
        chain = pool.resolve_chain(target_key=model_key)
        if not chain:
            logger.warning(f"No models in pool for: {model_key}")
            return None
        first = chain[0]
        key = first.registry_key or model_key
        return UnifiedLLMProxy(model_key=key, is_dispatcher=False)

    @classmethod
    def _get_local_coder_llm(cls):
        from .config import get_config
        from .local_llm import get_cached_local_llm

        cfg = get_config()
        local_models = cfg.models.local.model_dump(exclude_none=True)
        coder_config = local_models.get("qwen2_5_coder_7b")
        if not coder_config or not coder_config.get("path"):
            raise ValueError("qwen2_5_coder_7b not configured in config.yaml")
        if not hasattr(cls, "_agent_model"):
            cls._agent_model = get_cached_local_llm(
                coder_config["path"],
                n_ctx=coder_config.get("n_ctx", 4096),
                n_gpu_layers=coder_config.get("n_gpu_layers", -1),
                n_threads=coder_config.get("n_threads", 6),
                n_batch=coder_config.get("n_batch", 512),
                flash_attn=coder_config.get("flash_attn", True),
                cache_type_k=coder_config.get("cache_type_k", "q4_0"),
                cache_type_v=coder_config.get("cache_type_v", "q4_0"),
            )
            logger.info("[MCP] 使用本地 coder 模型（缓存单例）")
        return cls._agent_model

    def _resolve_agent_llm(
        self,
        specialist_llm,
        *,
        role_hint: str = "coding",
    ):
        """Pick LLM for MCP Agent tool loop from mcp.agent_model config."""
        from .config import get_config

        cfg = get_config()
        agent_model = (cfg.mcp.agent_model or "").strip().lower()

        if not agent_model or agent_model in ("auto", "specialist", "default", "cloud"):
            return specialist_llm

        if agent_model == "local":
            try:
                return self._get_local_coder_llm()
            except Exception as e:
                logger.warning("[MCP] 本地 Agent 模型不可用，回退 specialist: %s", e)
                return specialist_llm

        providers = cfg.proxy.providers if cfg.proxy.enabled else {}
        if agent_model in providers:
            if not hasattr(SmartDispatcher, "_agent_llm_proxies"):
                SmartDispatcher._agent_llm_proxies = {}
            cache_key = f"provider:{agent_model}"
            if cache_key not in SmartDispatcher._agent_llm_proxies:
                SmartDispatcher._agent_llm_proxies[cache_key] = UnifiedLLMProxy(
                    model_key="",
                    is_dispatcher=False,
                    role_hint=role_hint,
                    preferred_provider=agent_model,
                )
                logger.info("[MCP] Agent 使用云端 Provider: %s", agent_model)
            return SmartDispatcher._agent_llm_proxies[cache_key]

        logger.warning("[MCP] 未知 agent_model=%s，使用 specialist 链路", agent_model)
        return specialist_llm

    def _build_dispatcher_prompt(self, task: str) -> str:
        from .experts import format_experts_for_dispatcher
        from .model_pool import get_pool
        from .config import get_config

        available = []
        try:
            mode = get_config().routing.mode
        except Exception:
            mode = "auto"

        pool = get_pool()
        by_role: dict[str, list] = {}
        for cand in pool.all():
            if not pool._is_available(cand, mode):
                continue
            by_role.setdefault(cand.role, []).append(cand)

        for role in sorted(by_role.keys()):
            names = []
            for c in by_role[role][:4]:
                perf = self._get_model_performance(c.registry_key or c.key)
                names.append(f"{c.name}({c.source},{perf:.0%})")
            available.append(f"- **{role}**: " + " → ".join(names))

        if not available:
            for spec in MODEL_REGISTRY.values():
                if spec.role != ModelRole.DISPATCHER and os.path.exists(spec.path):
                    perf = self._get_model_performance(spec.key)
                    available.append(
                        f"- {spec.name} ({spec.key}): {spec.description} | 成功率: {perf:.0%}"
                    )

        if not available:
            available.append("- 无可用模型（请检查 config.yaml 或云端 Provider）")

        return DISPATCHER_SYSTEM_PROMPT.format(
            experts_catalog=format_experts_for_dispatcher(),
            available_models="\n".join(available),
        )

    def _extract_dispatch_json(self, response: str) -> dict | None:
        text = response.strip()
        fence_open = re.search(r"```(?:json)?\s*", text)
        if fence_open:
            start = fence_open.end()
            fence_close = text.find("```", start)
            if fence_close > start:
                text = text[start:fence_close].strip()
        else:
            text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
        brace = text.find("{")
        if brace < 0:
            return None
        try:
            data, _ = json.JSONDecoder().raw_decode(text[brace:])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None

    def _parse_dispatch_response(
        self, response: str, fallback_key: str
    ) -> DispatchDecision:
        try:
            data = self._extract_dispatch_json(response)
            if data:
                return DispatchDecision(
                    action=data.get("action", "dispatch"),
                    target_model=data.get("target_model"),
                    target_expert=data.get("target_expert"),
                    target_role=data.get("target_role"),
                    confidence=data.get("confidence", 0.5),
                    reasoning=data.get("reasoning", ""),
                    can_answer=data.get("can_answer", False),
                    direct_answer=data.get("direct_answer"),
                    alternative_models=data.get("alternative_models", []),
                )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse dispatch response: {e}")

        for key, spec in MODEL_REGISTRY.items():
            if spec.name.lower() in response.lower() or key in response.lower():
                return DispatchDecision(
                    action=DispatchAction.DISPATCH,
                    target_model=key,
                    confidence=0.6,
                    reasoning="从响应中提取模型",
                )

        return DispatchDecision(
            action=DispatchAction.FALLBACK,
            target_model=fallback_key,
            confidence=0.3,
            reasoning="解析失败，使用后备模型",
        )

    def _enrich_dispatch_decision(
        self, decision: DispatchDecision, task: str
    ) -> DispatchDecision:
        """Merge heuristic expert classification with LLM dispatch output."""
        from .experts import (
            EXPERT_REGISTRY,
            classify_task,
            expert_to_role,
            get_expert,
            preferred_models_for_expert,
        )

        hint_id, hint_conf, hint_alts = classify_task(task)

        from .experts import needs_web_search
        if needs_web_search(task):
            if decision.action == DispatchAction.ANSWER or decision.can_answer:
                decision.action = DispatchAction.DISPATCH
                decision.can_answer = False
                decision.direct_answer = None
            decision.target_expert = decision.target_expert or "chat"
            decision.target_role = decision.target_role or "general"
            if decision.confidence < 0.75:
                decision.confidence = 0.75

        if not decision.target_expert and decision.action in (
            DispatchAction.DISPATCH,
            DispatchAction.FALLBACK,
        ):
            decision.target_expert = hint_id
            if decision.confidence < hint_conf:
                decision.confidence = hint_conf

        if decision.target_expert:
            expert = get_expert(decision.target_expert)
            if expert:
                if not decision.target_role:
                    decision.target_role = expert.role
                if not decision.target_model and expert.preferred_models:
                    decision.target_model = expert.preferred_models[0]
                pref_alts = [m for m in expert.preferred_models[1:] if m]
                for m in pref_alts:
                    if m not in decision.alternative_models:
                        decision.alternative_models.append(m)

        if not decision.target_role and decision.target_expert:
            decision.target_role = expert_to_role(decision.target_expert)

        if not decision.target_expert and decision.target_model:
            for eid, expert in EXPERT_REGISTRY.items():
                if decision.target_model in expert.preferred_models:
                    decision.target_expert = eid
                    decision.target_role = expert.role
                    break

        for alt_expert in hint_alts:
            for m in preferred_models_for_expert(alt_expert):
                if m not in decision.alternative_models:
                    decision.alternative_models.append(m)

        return decision

    def _resolve_specialist_target(
        self,
        target_key: str,
        alternatives: list[str] | None = None,
        role_hint: str | None = None,
        expert_id: str | None = None,
    ) -> tuple[str, ModelSpec, list]:
        """Pick registry spec + proxy key from model pool chain."""
        from .model_pool import get_pool
        pool = get_pool()
        chain = pool.resolve_chain(
            target_key=target_key,
            alternatives=alternatives,
            role_hint=role_hint,
            expert_id=expert_id,
        )
        logger.info(
            "[POOL] specialist chain: %s",
            " → ".join(
                f"{c.name}[{c.source}]" for c in chain[:6]
            ) or "(empty)",
        )

        specialist_spec = None
        resolved_key = target_key
        for cand in chain:
            if cand.registry_key:
                specialist_spec = MODEL_REGISTRY.get(cand.registry_key)
                if specialist_spec:
                    resolved_key = cand.registry_key
                    break
            if cand.source in ("local", "registry") and cand.path and os.path.isfile(cand.path):
                resolved_key = cand.registry_key or cand.key
                specialist_spec = (
                    MODEL_REGISTRY.get(cand.registry_key)
                    if cand.registry_key
                    else get_fallback_model()
                )
                break
            if cand.source == "cloud":
                resolved_key = cand.registry_key or target_key or cand.key
                specialist_spec = MODEL_REGISTRY.get(target_key) or get_fallback_model()
                break

        if not specialist_spec:
            for alt in alternatives or []:
                specialist_spec = get_available_model(alt)
                if specialist_spec:
                    resolved_key = alt
                    break

        if not specialist_spec:
            fallback = get_fallback_model()
            specialist_spec = fallback
            resolved_key = fallback.key

        return resolved_key, specialist_spec, chain

    def _expert_meta(self, decision: DispatchDecision) -> dict:
        from .experts import get_expert
        expert = get_expert(decision.target_expert) if decision.target_expert else None
        return {
            "expert_id": decision.target_expert or "",
            "expert_name": expert.name if expert else "",
            "expert_icon": expert.icon if expert else "",
        }

    def _specialist_label(self, decision: DispatchDecision, specialist_spec: ModelSpec) -> str:
        from .experts import get_expert
        if decision.target_expert:
            expert = get_expert(decision.target_expert)
            if expert:
                return f"{expert.icon} {expert.name} → {specialist_spec.name}"
        return specialist_spec.name

    def _build_specialist_prompt(self, model_spec: ModelSpec, task: str) -> str:
        cap = model_spec.capabilities
        return SPECIALIST_SYSTEM_PROMPT_TEMPLATE.format(
            specialist_name=model_spec.name,
            specialist_description=model_spec.description,
            strengths="\n".join(f"- {s}" for s in model_spec.strengths),
            input_format=cap.input_format,
            output_format=cap.output_format,
            limitations="\n".join(f"- {l}" for l in cap.limitations),
            task=task,
        )

    def _append_workdir_context(self, prompt: str, workdir: str) -> str:
        if not workdir:
            return prompt
        return (
            f"{prompt}\n\n## 会话工作目录\n"
            f"当前 CoolClaw 会话绑定的工作目录: `{workdir}`\n"
            f"用户询问目录/路径时直接回答此路径；"
            f"不要说「没有本地文件系统权限」——系统可通过 MCP 工具访问该目录。\n"
        )

    def _route(
        self,
        user_input: str,
        workdir: str = "",
        forced_expert: str = "",
        dispatch_context: str = "",
    ) -> DispatchDecision:
        """OMO-style routing: rules → capabilities → heuristics → optional LLM."""
        from .capability_router import CapabilityRouter, RouteContext

        router = CapabilityRouter(llm_dispatch_fn=self._llm_dispatch_fallback)
        return router.route(RouteContext(
            user_input=user_input,
            workdir=workdir,
            forced_expert=forced_expert,
            dispatch_context=dispatch_context or "",
        ))

    @staticmethod
    def _make_agent_llm_fn(agent_llm, max_tokens: int, temperature: float):
        """Bridge Agent loop to LLM with native function-calling when supported."""
        from .agent import (
            _parse_prompt_tool_calls,
            _looks_like_internal_reasoning,
            _infer_simple_tool_call,
        )

        async def _agent_llm_fn(messages, tools):
            try:
                result = agent_llm.chat(
                    messages=messages,
                    tools=tools or None,
                    max_tokens=max_tokens,
                    temperature=min(temperature, 0.3),
                )
            except Exception as e:
                logger.error("[AGENT] LLM call failed: %s", e)
                raise
            content = result.content if hasattr(result, "content") else ""
            finish_reason = getattr(result, "finish_reason", "") or ""
            tool_calls = None
            if hasattr(result, "tool_calls") and result.tool_calls:
                tool_calls = result.tool_calls
            else:
                tool_calls = _parse_prompt_tool_calls(content)
                if tool_calls:
                    content = ""
            if not tool_calls and (
                finish_reason == "MALFORMED_FUNCTION_CALL"
                or _looks_like_internal_reasoning(content)
            ):
                tool_calls = _infer_simple_tool_call(messages, tools or [])
                if tool_calls:
                    logger.info("[AGENT] recovered tool call from malformed/reasoning response")
                    content = ""
            return {
                "content": content,
                "tool_calls": tool_calls,
                "finish_reason": finish_reason,
            }

        return _agent_llm_fn

    def _llm_dispatch_fallback(self, ctx) -> DispatchDecision:
        """Slim LLM tie-breaker — only when rule/heuristic confidence is low."""
        from .capability_router import SLIM_ROUTER_PROMPT, RouteContext

        if isinstance(ctx, RouteContext):
            user_input = ctx.user_input
            context_block = ""
            if ctx.dispatch_context:
                context_block = f"最近对话:\n{ctx.dispatch_context[-1200:]}\n\n"
        else:
            user_input = str(ctx)
            context_block = ""

        self._ensure_dispatcher_loaded()
        prompt = SLIM_ROUTER_PROMPT.format(
            user_input=user_input[:800],
            context_block=context_block,
        )
        result = self._dispatcher_llm.complete(
            prompt=prompt,
            max_tokens=200,
            temperature=0.1,
        )
        logger.info("[ROUTE-LLM] raw: %s", result.content[:160])
        fallback = get_fallback_model()
        decision = self._parse_dispatch_response(result.content, fallback.key)
        decision.route_source = "llm"
        if decision.target_model:
            perf = self._get_model_performance(decision.target_model)
            decision.confidence *= 0.7 + 0.3 * perf
        return decision

    def _dispatch(self, task: str) -> DispatchDecision:
        """Legacy LLM-only dispatch — kept for compatibility, prefer _route()."""
        self._ensure_dispatcher_loaded()

        system_prompt = self._build_dispatcher_prompt(task)
        full_prompt = f"{system_prompt}\n\n用户请求：{task}\n\n请分析并输出JSON决策："

        result = self._dispatcher_llm.complete(
            prompt=full_prompt,
            max_tokens=400,
            temperature=0.1,
        )

        logger.info(f"Dispatcher raw response:\n{result.content[:200]}...")

        fallback = get_fallback_model()
        decision = self._parse_dispatch_response(result.content, fallback.key)
        decision = self._enrich_dispatch_decision(decision, task)
        decision.route_source = "llm_legacy"

        if decision.target_model:
            perf = self._get_model_performance(decision.target_model)
            decision.confidence *= 0.7 + 0.3 * perf

        expert_label = decision.target_expert or "—"
        logger.info(
            "Dispatch: %s → expert=%s role=%s model=%s (conf=%.2f)",
            decision.action,
            expert_label,
            decision.target_role,
            decision.target_model,
            decision.confidence,
        )
        return decision

    def chat(
        self,
        user_input: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        context: str = None,
        skip_memory: bool = False,
        workdir: str = "",
        dispatch_context: str = None,
        forced_expert: str = "",
        trace=None,
    ) -> DispatcherResult:
        from .run_trace import RunTrace

        trace = (trace or RunTrace()).bind()
        trace.phase("start")
        try:
            memory_context = context or self._memory.get_context_with_limit(
                max_tokens=1200
            )
            dispatch_ctx = dispatch_context or memory_context

            if memory_context:
                full_input = f"对话历史:\n{memory_context}\n\n当前请求: {user_input}"
            else:
                full_input = user_input

            decision = self._route(
                user_input,
                workdir=workdir,
                forced_expert=forced_expert,
                dispatch_context=dispatch_ctx or "",
            )
            trace.expert = decision.target_expert or ""
            trace.route_source = decision.route_source or ""
            trace.phase("route")

            if decision.action == DispatchAction.ANSWER and decision.can_answer:
                if not skip_memory:
                    self._memory.add(
                        "user", user_input, "dispatcher", {"model": "dispatcher"}
                    )
                    self._memory.add(
                        "assistant",
                        decision.direct_answer or "抱歉，我无法回答这个问题。",
                        "dispatcher",
                        {"model": "dispatcher"},
                    )
                    if self.config.auto_save and self.config.memory_path:
                        self._memory.save(self.config.memory_path)
                return DispatcherResult(
                    content=decision.direct_answer or "抱歉，我无法回答这个问题。",
                    specialist_used=decision.route_source or "规则路由",
                    confidence=decision.confidence,
                    reasoning=decision.reasoning,
                    success=True,
                    trace_id=trace.trace_id,
                    trace=trace.to_dict(),
                    **self._expert_meta(decision),
                )

            target_key = decision.target_model
            if not target_key or decision.action == DispatchAction.DECLINE:
                return DispatcherResult(
                    content="抱歉，这个任务超出了我的能力范围。",
                    specialist_used="无",
                    confidence=0.0,
                    reasoning="任务被拒绝",
                    success=False,
                    error="Task declined",
                )

            target_key, specialist_spec, _chain = self._resolve_specialist_target(
                target_key,
                decision.alternative_models,
                role_hint=decision.target_role,
                expert_id=decision.target_expert,
            )

            if (
                self._specialist_llm is None
                or getattr(self._specialist_llm, "_current_model", None) != target_key
            ):
                self._specialist_llm = self._load_specialist(target_key)
                if self._specialist_llm is None:
                    return DispatcherResult(
                        content=f"无法加载模型: {target_key}",
                        specialist_used=target_key,
                        confidence=0.0,
                        reasoning="模型加载失败",
                        success=False,
                        error="Model load failed",
                    )

            agent_mode, agent_log = self._resolve_agent_setup(
                decision, user_input, workdir,
            )
            use_agent = agent_mode is not None
            if use_agent:
                logger.info("[MCP] agent loop: %s", agent_log)
            elif workdir:
                logger.info("[MCP] skip agent loop: %s", agent_log)

            specialist_prompt = self._append_workdir_context(
                self._build_specialist_prompt(
                    specialist_spec,
                    user_input if use_agent else full_input,
                ),
                workdir,
            )

            if use_agent:
                from .agent import Agent

                agent_llm = self._resolve_agent_llm(
                    self._specialist_llm,
                    role_hint=decision.target_role or "coding",
                )

                if agent_mode == "web":
                    specialist_prompt += self._build_web_tool_prompt()
                    extra = SmartDispatcher._build_websearch_server()
                    agent_workdir = ""
                    agent_role = "architect"
                    agent_max_rounds = 6
                else:
                    specialist_prompt += self._build_tool_prompt()
                    extra = SmartDispatcher._build_extra_server(workdir)
                    agent_workdir = workdir
                    agent_role = "full"
                    agent_max_rounds = 15

                _agent_llm_fn = SmartDispatcher._make_agent_llm_fn(
                    agent_llm, max_tokens, temperature,
                )

                from .orchestrator import Orchestrator, is_complex_task
                if agent_mode == "full" and is_complex_task(user_input):
                    logger.info("[CHAT] complex task → Orchestrator pipeline")
                    orch = Orchestrator(
                        llm_fn=_agent_llm_fn,
                        workdir=agent_workdir,
                        max_rounds=agent_max_rounds,
                        model_name=specialist_spec.name,
                        extra_server=extra,
                        trace_id=trace.trace_id,
                    )
                    agent_result = _run_async(orch.run(user_input, specialist_prompt))
                    agent_content = agent_result.final_content
                    agent_tool_calls = []
                    for step in getattr(agent_result, "steps", []) or []:
                        agent_tool_calls.extend(step.tool_calls or [])
                    agent_rounds = agent_result.rounds
                else:
                    agent = Agent(
                        llm_fn=_agent_llm_fn,
                        workdir=agent_workdir,
                        max_rounds=agent_max_rounds,
                        model_name=getattr(agent_llm, "_current_model", None) or specialist_spec.name,
                        extra_server=extra,
                        role=agent_role,
                        trace_id=trace.trace_id,
                    )

                    task = user_input
                    if agent_mode == "full":
                        try:
                            import re as _re
                            from pathlib import Path
                            root = Path(workdir).expanduser().resolve()
                            if root.is_dir():
                                mentioned = set(_re.findall(
                                    r'[\w\-\.]+\.(?:py|js|ts|yaml|yml|json|toml|md|txt|sh|go|rs|java|c|cpp|h|hpp|rb|php|sql|html|css|vue|jsx|tsx|cfg|ini|env|lock)',
                                    user_input,
                                ))
                                for fname in mentioned:
                                    fpath = root / fname
                                    if fpath.is_file():
                                        try:
                                            content = fpath.read_text(errors="replace")
                                            if len(content) > 8000:
                                                content = content[:8000] + f"\n... (truncated)"
                                            task += f"\n\n### 文件内容: {fname}\n```\n{content}\n```\n"
                                            logger.info(f"[CHAT] auto-read file for agent: {fname}")
                                        except Exception:
                                            pass
                        except Exception:
                            pass

                    agent_result = agent.run_sync(task, system_prompt=specialist_prompt)
                    agent_content = agent_result.content
                    agent_tool_calls = agent_result.tool_calls or []
                    agent_rounds = agent_result.rounds

                if not skip_memory:
                    self._memory.add("user", user_input, target_key, {"model": target_key})
                    self._memory.add(
                        "assistant", agent_content, target_key, {"model": target_key}
                    )
                    if self.config.auto_save and self.config.memory_path:
                        self._memory.save(self.config.memory_path)

                for tc in agent_tool_calls:
                    if hasattr(tc, "name"):
                        logger.info(f"[AGENT] {tc.name}({tc.args}) → {'ok' if tc.success else 'fail'}")

                trace.model = getattr(agent_llm, "_current_model", None) or target_key
                trace.agent_mode = agent_mode or ""
                trace.tool_calls = len(agent_tool_calls)
                trace.rounds = agent_rounds or 0
                trace.phase("done")

                tc_payload = []
                for tc in agent_tool_calls:
                    if hasattr(tc, "name"):
                        tc_payload.append({
                            "name": tc.name,
                            "args": tc.args,
                            "result": (tc.result or "")[:500],
                            "success": tc.success,
                        })
                    elif isinstance(tc, dict):
                        tc_payload.append(tc)

                return DispatcherResult(
                    content=agent_content,
                    specialist_used=self._specialist_label(decision, specialist_spec),
                    confidence=decision.confidence,
                    reasoning=decision.reasoning,
                    success=True,
                    tool_calls=tc_payload or None,
                    trace_id=trace.trace_id,
                    trace=trace.to_dict(),
                    **self._expert_meta(decision),
                )

            else:
                messages = [
                    {"role": "system", "content": specialist_prompt},
                    {"role": "user", "content": full_input},
                ]
                response = self._specialist_llm.chat(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                trace.model = target_key
                trace.phase("chat_done")

            if not skip_memory:
                self._memory.add("user", user_input, target_key, {"model": target_key})
                self._memory.add(
                    "assistant", response.content, target_key, {"model": target_key}
                )

                if self.config.auto_save and self.config.memory_path:
                    self._memory.save(self.config.memory_path)

            trace.phase("done")
            return DispatcherResult(
                content=response.content,
                specialist_used=self._specialist_label(decision, specialist_spec),
                confidence=decision.confidence,
                reasoning=decision.reasoning,
                success=True,
                trace_id=trace.trace_id,
                trace=trace.to_dict(),
                **self._expert_meta(decision),
            )

        except Exception as e:
            logger.error(f"Dispatch error: {e}")
            return DispatcherResult(
                content=f"处理请求时出错: {str(e)}",
                specialist_used="",
                confidence=0.0,
                reasoning="执行出错",
                success=False,
                error=str(e),
            )

    @staticmethod
    def _build_websearch_server():
        try:
            from .mcp import create_websearch_server
            logger.info("[MCP] Created websearch-only server")
            return create_websearch_server()
        except Exception as e:
            logger.warning("[MCP] Failed to create websearch server: %s", e)
            return None

    def _resolve_agent_setup(
        self, decision: DispatchDecision, user_input: str, workdir: str,
    ) -> tuple[str | None, str]:
        """Return (agent_mode, log_reason). mode: 'web' | 'full' | None."""
        try:
            from pathlib import Path
            from .config import get_config
            from .experts import resolve_agent_mode, needs_web_search, is_workdir_query
            _cfg = get_config()
            if not _cfg.mcp.enabled:
                return None, "mcp disabled"

            if needs_web_search(user_input) and not is_workdir_query(user_input):
                return "web", "web search agent"

            wd = Path(workdir).expanduser() if workdir else None
            if wd and wd.is_dir():
                return "full", f"workdir+mcp full agent (expert={decision.target_expert or '—'})"

            mode = resolve_agent_mode(
                decision.target_expert or "", user_input, workdir,
            )
            if mode:
                return mode, f"expert={decision.target_expert or '—'} mode={mode}"
            return None, f"expert={decision.target_expert or '—'} (cloud chat only)"
        except Exception as e:
            logger.debug("agent setup: %s", e)
            return None, "agent setup failed"

    def _build_web_tool_prompt(self) -> str:
        return (
            "\n\n## 网络查询\n"
            "用户问题涉及实时信息（价格、天气、新闻等）。"
            "必须先调用 web_search 或 fetch_url 获取最新结果，禁止凭记忆编造数字。\n"
        )

    def _build_tool_prompt(self) -> str:
        """Minimal hint — tool schemas come from MCP function calling."""
        return (
            "\n\n## 工具使用\n"
            "你已通过 function calling 接入 MCP 工具（文件、终端、Git、LSP、代码图等）。\n"
            "需要操作文件或运行命令时直接调用工具，不要编造未读取的内容。\n"
            "写入前先 read_file 确认现状；修改后可用 lsp_diagnostics 自检。\n"
        )

    async def chat_stream(
        self,
        user_input: str,
        workdir: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        context: str = None,
        dispatch_context: str = None,
        forced_expert: str = "",
        trace=None,
    ):
        """Streaming version of chat() -- yields SSE event dicts via Agent.run_stream."""
        import json as _json
        from .run_trace import RunTrace

        trace = (trace or RunTrace()).bind()
        trace.phase("start")
        try:
            memory_context = context or self._memory.get_context_with_limit(max_tokens=1200)
            dispatch_ctx = dispatch_context or memory_context
            full_input = (
                f"对话历史:\n{memory_context}\n\n当前请求: {user_input}"
                if memory_context else user_input
            )
            yield {"event": "trace", "data": _json.dumps({"trace_id": trace.trace_id})}

            decision = self._route(
                user_input,
                workdir=workdir,
                forced_expert=forced_expert,
                dispatch_context=dispatch_ctx or "",
            )
            trace.expert = decision.target_expert or ""
            trace.route_source = decision.route_source or ""
            trace.phase("route")

            from .experts import dispatch_payload
            yield {"event": "dispatch", "data": _json.dumps(dispatch_payload(decision, trace.to_dict()))}

            if decision.action == DispatchAction.ANSWER and decision.can_answer:
                answer_text = decision.direct_answer or ""
                yield {"event": "token", "data": answer_text}
                trace.phase("done")
                done = dispatch_payload(decision, trace.to_dict())
                done["model"] = "dispatcher"
                done["content"] = answer_text
                yield {"event": "done", "data": _json.dumps(done)}
                return

            target_key = decision.target_model
            if not target_key or decision.action == DispatchAction.DECLINE:
                yield {"event": "token", "data": "抱歉，这个任务超出了我的能力范围。"}
                yield {"event": "done", "data": _json.dumps({"model": "none", "confidence": 0})}
                return

            target_key, specialist_spec, _chain = self._resolve_specialist_target(
                target_key,
                decision.alternative_models,
                role_hint=decision.target_role,
                expert_id=decision.target_expert,
            )

            if self._specialist_llm is None or getattr(self._specialist_llm, "_current_model", None) != target_key:
                self._specialist_llm = self._load_specialist(target_key)
                if self._specialist_llm is None:
                    yield {"event": "error", "data": f"无法加载模型: {target_key}"}
                    return

            agent_mode, agent_log = self._resolve_agent_setup(
                decision, user_input, workdir,
            )
            use_agent = agent_mode is not None
            if use_agent:
                logger.info("[MCP] agent loop: %s", agent_log)
            elif workdir:
                logger.info("[MCP] skip agent loop: %s", agent_log)

            specialist_prompt = self._append_workdir_context(
                self._build_specialist_prompt(
                    specialist_spec,
                    user_input if use_agent else full_input,
                ),
                workdir,
            )

            if use_agent:
                from .agent import Agent

                agent_llm = self._resolve_agent_llm(
                    self._specialist_llm,
                    role_hint=decision.target_role or "coding",
                )

                if agent_mode == "web":
                    specialist_prompt += self._build_web_tool_prompt()
                    extra = SmartDispatcher._build_websearch_server()
                    agent_workdir = ""
                    agent_role = "architect"
                    agent_max_rounds = 6
                else:
                    specialist_prompt += self._build_tool_prompt()
                    extra = SmartDispatcher._build_extra_server(workdir)
                    agent_workdir = workdir
                    agent_role = "full"
                    agent_max_rounds = 15

                _agent_llm_fn = SmartDispatcher._make_agent_llm_fn(
                    agent_llm, max_tokens, temperature,
                )

                from .orchestrator import Orchestrator, is_complex_task
                stream_agent = Agent(
                    llm_fn=_agent_llm_fn,
                    workdir=agent_workdir,
                    max_rounds=agent_max_rounds,
                    model_name=getattr(agent_llm, "_current_model", None) or specialist_spec.name,
                    extra_server=extra,
                    role=agent_role,
                    trace_id=trace.trace_id,
                )

                trace.model = getattr(agent_llm, "_current_model", None) or target_key
                trace.agent_mode = agent_mode or ""

                if agent_mode == "full" and is_complex_task(user_input):
                    logger.info("[CHAT-STREAM] complex task → Orchestrator pipeline")
                    orch = Orchestrator(
                        llm_fn=_agent_llm_fn,
                        workdir=agent_workdir,
                        max_rounds=agent_max_rounds,
                        model_name=specialist_spec.name,
                        extra_server=extra,
                        trace_id=trace.trace_id,
                    )
                    async for event_json in orch.run_stream(user_input, specialist_prompt):
                        try:
                            event = _json.loads(event_json) if isinstance(event_json, str) else event_json
                            etype = event.get("type", "")
                            if etype == "token":
                                yield {"event": "token", "data": event.get("content", "")}
                            elif etype == "tool_start":
                                yield {"event": "tool_start", "data": _json.dumps({"tool": event.get("tool", "")})}
                            elif etype == "tool_end":
                                yield {"event": "tool_end", "data": _json.dumps({
                                    "tool": event.get("tool", ""),
                                    "success": event.get("success", False),
                                    "preview": event.get("preview", "")[:200],
                                })}
                            elif etype == "done":
                                tc_list = event.get("tool_calls", [])
                                trace.tool_calls = len(tc_list)
                                trace.rounds = event.get("rounds", 0) or 0
                                trace.phase("done")
                                done = dispatch_payload(decision, trace.to_dict())
                                final_content = event.get("content", "")
                                done.update({
                                    "model": getattr(agent_llm, "_current_model", None) or specialist_spec.name,
                                    "content": final_content,
                                    "tool_calls": tc_list,
                                    "rounds": trace.rounds,
                                })
                                if final_content:
                                    yield {"event": "token", "data": final_content}
                                yield {"event": "done", "data": _json.dumps(done)}
                            elif etype == "error":
                                yield {"event": "error", "data": event.get("message", "")}
                        except Exception as e:
                            logger.warning("[CHAT-STREAM] orchestrator event parse: %s", e)
                else:
                    task = user_input
                    if agent_mode == "full":
                        try:
                            import re as _re
                            from pathlib import Path
                            root = Path(workdir).expanduser().resolve()
                            if root.is_dir():
                                mentioned = set(_re.findall(
                                    r'[\w\-\.]+\.(?:py|js|ts|yaml|yml|json|toml|md|txt|sh|go|rs)',
                                    user_input,
                                ))
                                for fname in mentioned:
                                    fpath = root / fname
                                    if fpath.is_file():
                                        try:
                                            fc = fpath.read_text(errors="replace")
                                            if len(fc) > 8000:
                                                fc = fc[:8000] + "\n... (truncated)"
                                            task += f"\n\n### 文件内容: {fname}\n```\n{fc}\n```\n"
                                        except Exception:
                                            pass
                        except Exception:
                            pass

                    agent_done = False
                    async for event_json in stream_agent.run_stream(task, system_prompt=specialist_prompt):
                        try:
                            event = _json.loads(event_json) if isinstance(event_json, str) else event_json
                            etype = event.get("type", "")
                            if etype == "token":
                                yield {"event": "token", "data": event.get("content", "")}
                            elif etype == "tool_start":
                                yield {"event": "tool_start", "data": _json.dumps({"tool": event.get("tool", "")})}
                            elif etype == "tool_end":
                                yield {"event": "tool_end", "data": _json.dumps({
                                    "tool": event.get("tool", ""),
                                    "success": event.get("success", False),
                                    "preview": event.get("preview", "")[:200],
                                })}
                            elif etype == "done":
                                agent_done = True
                                tc_list = event.get("tool_calls", [])
                                trace.tool_calls = len(tc_list)
                                trace.rounds = event.get("rounds", 0) or 0
                                trace.phase("done")
                                done = dispatch_payload(decision, trace.to_dict())
                                final_content = event.get("content", "")
                                done.update({
                                    "model": getattr(agent_llm, "_current_model", None) or specialist_spec.name,
                                    "content": final_content,
                                    "tool_calls": tc_list,
                                    "rounds": trace.rounds,
                                })
                                if final_content:
                                    yield {"event": "token", "data": final_content}
                                yield {"event": "done", "data": _json.dumps(done)}
                            elif etype == "error":
                                yield {"event": "error", "data": event.get("message", "")}
                        except Exception as e:
                            logger.warning("[CHAT-STREAM] agent event parse: %s", e)
                    else:
                        if not agent_done:
                            yield {"event": "error", "data": "Agent 未返回有效结果"}
            else:
                messages = [
                    {"role": "system", "content": specialist_prompt},
                    {"role": "user", "content": full_input},
                ]
                response = self._specialist_llm.chat(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                content = response.content if hasattr(response, "content") else str(response)
                model_label = (
                    getattr(self._specialist_llm, "_current_model", None)
                    or specialist_spec.name
                )
                trace.model = model_label
                trace.phase("chat_done")
                trace.phase("done")
                yield {"event": "token", "data": content}
                done = dispatch_payload(decision, trace.to_dict())
                done.update({"model": model_label, "content": content})
                yield {"event": "done", "data": _json.dumps(done)}

        except Exception as e:
            logger.error(f"Chat stream error: {e}")
            yield {"event": "error", "data": str(e)}

    def feedback(self, rating: int, notes: Optional[str] = None):
        if not self.config.enable_feedback:
            return
        last_model = None
        if self._memory.messages:
            for msg in reversed(self._memory.messages):
                if msg.role == "assistant" and msg.metadata:
                    last_model = msg.metadata.get("model")
                    break
        if last_model and last_model in self._performance:
            rec = self._performance[last_model]
            rec.total_uses += 1
            if rating >= 4:
                rec.success_count += 1
            else:
                rec.failure_count += 1
            self._save_performance()
            logger.info(f"Feedback recorded for {last_model}: rating={rating}")

    def clear_memory(self):
        self._memory.clear()
        logger.info("Memory cleared")

    def get_status(self) -> dict:
        from .model_pool import get_pool
        pool = get_pool()
        dispatcher_name = (
            get_dispatcher_model().name if self._dispatcher_llm else "not loaded"
        )
        from .experts import experts_status
        return {
            "dispatcher": dispatcher_name,
            "specialist": getattr(self._specialist_llm, "_current_model", "none"),
            "memory_size": len(self._memory.messages),
            "routing": pool.routing_status(),
            "experts": experts_status(pool),
            "performance": {
                key: {
                    "success_rate": rec.success_rate,
                    "total_uses": rec.total_uses,
                }
                for key, rec in self._performance.items()
            },
        }


Dispatcher = SmartDispatcher
