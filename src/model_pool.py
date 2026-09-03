"""Unified model pool — merges DB configs, config.yaml, and MODEL_REGISTRY for routing."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import get_config
from .models import MODEL_REGISTRY, ModelRole, ModelSpec

logger = logging.getLogger(__name__)

ROLE_FALLBACK_CHAIN = {
    "thinking": ("coding", "general"),
    "ocr": ("vision", "general"),
    "vision": ("general",),
    "documents": ("general",),
}

ROLE_ALIASES = {
    "dispatcher": "fast",
    "reasoning": "thinking",
    "thinking": "thinking",
    "ocr": "ocr",
    "vision": "vision",
    "documents": "documents",
    "coding": "coding",
    "general": "general",
    "fast": "fast",
}


def normalize_role(role: str) -> str:
    return ROLE_ALIASES.get((role or "general").lower(), role.lower())


def registry_role_to_pool(role: ModelRole) -> str:
    mapping = {
        ModelRole.DISPATCHER: "fast",
        ModelRole.FAST: "fast",
        ModelRole.CODING: "coding",
        ModelRole.GENERAL: "general",
        ModelRole.REASONING: "thinking",
        ModelRole.OCR: "ocr",
        ModelRole.VISION: "vision",
        ModelRole.DOCUMENTS: "documents",
    }
    return mapping.get(role, "general")


@dataclass
class ModelCandidate:
    key: str
    name: str
    role: str
    source: str  # local | cloud | registry
    priority: int = 10
    provider_type: str = "local"
    model_name: str = ""
    api_key: str = ""
    base_url: str = ""
    path: str = ""
    registry_key: str = ""
    extra_config: dict = field(default_factory=dict)
    rpm: int = 0
    rpd: int = 0
    available: bool = True
    db_id: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "role": self.role,
            "source": self.source,
            "priority": self.priority,
            "provider_type": self.provider_type,
            "model_name": self.model_name,
            "path": self.path,
            "registry_key": self.registry_key,
            "available": self.available,
            "rpm": self.rpm,
            "rpd": self.rpd,
        }


class ModelPool:
    _instance: Optional["ModelPool"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._candidates: list[ModelCandidate] = []
            cls._instance._by_key: dict[str, ModelCandidate] = {}
            cls._instance.reload()
        return cls._instance

    def reload(self) -> None:
        self._candidates = []
        self._by_key = {}
        seen_keys: set[str] = set()

        self._load_from_database(seen_keys)
        self._load_from_config_yaml(seen_keys)
        self._load_from_registry(seen_keys)

        self._candidates.sort(key=lambda c: (c.priority, c.key))
        self._by_key = {c.key: c for c in self._candidates}
        logger.info(
            "[POOL] loaded %d models: %s",
            len(self._candidates),
            ", ".join(f"{c.key}({c.role})" for c in self._candidates[:8]),
        )

    def _add(self, cand: ModelCandidate, seen: set[str]) -> None:
        if cand.key in seen:
            return
        seen.add(cand.key)
        self._candidates.append(cand)

    def _load_from_database(self, seen: set[str]) -> None:
        try:
            from .database import db
            rows = db.list_model_configs(is_active=True)
        except Exception as e:
            logger.debug("[POOL] DB configs unavailable: %s", e)
            return

        for row in rows:
            extra = row.get("extra_config") or "{}"
            if isinstance(extra, str):
                try:
                    extra = json.loads(extra)
                except json.JSONDecodeError:
                    extra = {}

            pt = row.get("provider_type", "openai")
            role = normalize_role(row.get("role", "general"))
            model_name = row.get("model_name", "")
            db_id = row.get("id")

            if pt == "local":
                path = model_name
                if not path or not os.path.isfile(path):
                    continue
                key = f"local_db_{db_id}"
                self._add(ModelCandidate(
                    key=key,
                    name=row.get("display_name") or Path(path).stem,
                    role=role,
                    source="local",
                    priority=row.get("priority", 10),
                    provider_type="local",
                    model_name=path,
                    path=path,
                    extra_config=extra if isinstance(extra, dict) else {},
                    db_id=db_id,
                ), seen)
            else:
                if not row.get("api_key") and pt not in ("openai_compat",):
                    continue
                key = f"cloud_db_{db_id}"
                self._add(ModelCandidate(
                    key=key,
                    name=row.get("display_name") or model_name,
                    role=role,
                    source="cloud",
                    priority=row.get("priority", 10),
                    provider_type=pt,
                    model_name=model_name,
                    api_key=row.get("api_key", ""),
                    base_url=row.get("base_url", ""),
                    rpm=row.get("rpm", 0),
                    rpd=row.get("rpd", 0),
                    extra_config=extra if isinstance(extra, dict) else {},
                    db_id=db_id,
                ), seen)

    def _load_from_config_yaml(self, seen: set[str]) -> None:
        try:
            cfg = get_config()
        except Exception:
            return

        local = cfg.models.local.model_dump(exclude_none=True)
        for key, data in local.items():
            if key == "default" or not isinstance(data, dict):
                continue
            path = data.get("path", "")
            if path and not os.path.isabs(path):
                path = os.path.join(cfg.paths.models_dir, path)
            if not path or not os.path.isfile(path):
                continue
            registry_key = _guess_registry_key(key)
            pool_key = f"yaml_{key}"
            self._add(ModelCandidate(
                key=pool_key,
                name=registry_key,
                role=normalize_role(data.get("role", "general")),
                source="local",
                priority=20,
                provider_type="local",
                model_name=path,
                path=path,
                registry_key=registry_key,
                extra_config={k: v for k, v in data.items() if k not in ("path", "role")},
            ), seen)

        proxy = cfg.proxy
        if proxy.enabled:
            from .experts import infer_cloud_roles, is_multimodal_cloud_model
            for pname, prov in proxy.providers.items():
                if not prov.api_key and pname != "openai_compat":
                    continue
                if prov.roles:
                    roles = [normalize_role(r) for r in prov.roles]
                else:
                    roles = infer_cloud_roles(
                        pname, prov.model, multimodal=getattr(prov, "multimodal", False),
                    )
                rl = proxy.rate_limits.get(pname)
                rpm = getattr(rl, "rpm", 0) if rl else 0
                rpd = getattr(rl, "rpd", 0) if rl else 0
                mm = getattr(prov, "multimodal", False) or is_multimodal_cloud_model(pname, prov.model)
                fallback_order = list(proxy.fallback_order or [])
                for role in roles:
                    pool_key = f"yaml_cloud_{pname}_{role}"
                    self._add(ModelCandidate(
                        key=pool_key,
                        name=f"{pname}/{prov.model}",
                        role=role,
                        source="cloud",
                        priority=_cloud_yaml_priority(pname, prov.model, role, fallback_order, mm),
                        provider_type=_map_provider_name(pname),
                        model_name=prov.model,
                        api_key=prov.api_key,
                        base_url=prov.base_url,
                        rpm=rpm,
                        rpd=rpd,
                        extra_config={"yaml_provider": pname, "multimodal": mm},
                    ), seen)

    def _load_from_registry(self, seen: set[str]) -> None:
        try:
            cfg = get_config()
            models_dir = cfg.paths.models_dir
        except Exception:
            models_dir = ""

        for reg_key, spec in MODEL_REGISTRY.items():
            path = spec.path
            if not os.path.isfile(path) and models_dir:
                alt = os.path.join(models_dir, os.path.basename(path))
                if os.path.isfile(alt):
                    path = alt
            available = os.path.isfile(path)
            pool_key = f"reg_{reg_key}"
            if pool_key in seen:
                continue
            self._add(ModelCandidate(
                key=pool_key,
                name=spec.name,
                role=registry_role_to_pool(spec.role),
                source="registry",
                priority=25 if available else 99,
                provider_type="local",
                model_name=path,
                path=path if available else spec.path,
                registry_key=reg_key,
                available=available,
            ), seen)

    def all(self) -> list[ModelCandidate]:
        return list(self._candidates)

    def get(self, key: str) -> Optional[ModelCandidate]:
        return self._by_key.get(key)

    def by_registry_key(self, registry_key: str) -> Optional[ModelCandidate]:
        for c in self._candidates:
            if c.registry_key == registry_key:
                return c
        return None

    def by_role(self, role: str, routing_mode: str | None = None) -> list[ModelCandidate]:
        role = normalize_role(role)
        items = [c for c in self._candidates if c.role == role and self._is_available(c, routing_mode)]
        return sorted(items, key=lambda c: c.priority)

    def dispatcher_candidates(self, routing_mode: str | None = None) -> list[ModelCandidate]:
        fast = self.by_role("fast", routing_mode)
        if fast:
            return fast
        return self.by_role("general", routing_mode)[:1]

    def resolve_chain(
        self,
        target_key: str | None = None,
        alternatives: list[str] | None = None,
        role_hint: str | None = None,
        routing_mode: str | None = None,
        expert_id: str | None = None,
        preferred_provider: str | None = None,
    ) -> list[ModelCandidate]:
        """Build ordered fallback chain for a dispatch decision."""
        routing_mode = routing_mode or self._routing_mode()
        chain: list[ModelCandidate] = []
        seen: set[str] = set()

        def _append(c: ModelCandidate | None):
            if c and c.key not in seen and self._is_available(c, routing_mode):
                seen.add(c.key)
                chain.append(c)

        role = normalize_role(role_hint) if role_hint else None

        if expert_id:
            try:
                bindings = get_config().routing.expert_bindings or {}
                provider_id = bindings.get(expert_id)
                if provider_id:
                    for c in self._candidates:
                        if (
                            c.source == "cloud"
                            and c.extra_config.get("yaml_provider") == provider_id
                            and self._is_available(c, routing_mode)
                            and (not role or c.role == role)
                        ):
                            _append(c)
            except Exception as e:
                logger.debug("[POOL] expert binding: %s", e)

        for key in ([target_key] if target_key else []) + (alternatives or []):
            if not key:
                continue
            c = self.by_registry_key(key) or self.get(key)
            if not c:
                spec = MODEL_REGISTRY.get(key)
                if spec:
                    role = role or registry_role_to_pool(spec.role)
            _append(c)

        if not role and target_key:
            spec = MODEL_REGISTRY.get(target_key)
            if spec:
                role = registry_role_to_pool(spec.role)

        if not role:
            role = "general"

        for c in self.by_role(role, routing_mode):
            _append(c)

        if not chain and role in ROLE_FALLBACK_CHAIN:
            for fb_role in ROLE_FALLBACK_CHAIN[role]:
                for c in self.by_role(fb_role, routing_mode):
                    _append(c)

        if not chain:
            for c in self.by_role("general", routing_mode):
                _append(c)

        if preferred_provider:
            bound = [
                c for c in chain
                if c.source == "cloud" and c.extra_config.get("yaml_provider") == preferred_provider
            ]
            rest = [c for c in chain if c not in bound]
            chain = bound + rest

        cloud = [c for c in chain if c.source == "cloud"]
        local = [c for c in chain if c.source in ("local", "registry")]
        if routing_mode == "auto":
            chain = _deprioritize_rate_limited(cloud) + local
        elif cloud:
            chain = _deprioritize_rate_limited(cloud) + local
        else:
            chain = cloud + local

        return chain

    def _routing_mode(self) -> str:
        try:
            return get_config().routing.mode
        except Exception:
            return "auto"

    def _is_available(self, cand: ModelCandidate, routing_mode: str | None) -> bool:
        routing_mode = routing_mode or self._routing_mode()
        if cand.source in ("local", "registry"):
            if routing_mode == "cloud_only":
                return False
            if cand.path and not os.path.isfile(cand.path):
                return False
            return cand.available
        if cand.source == "cloud":
            if routing_mode == "local_only":
                return False
            return bool(cand.api_key or cand.provider_type == "openai_compat")
        return False

    def routing_status(self) -> dict:
        mode = self._routing_mode()
        by_role: dict[str, list] = {}
        for c in self._candidates:
            if self._is_available(c, mode):
                by_role.setdefault(c.role, []).append(c.to_dict())
        return {
            "routing_mode": mode,
            "total": len(self._candidates),
            "available_by_role": by_role,
        }


def _deprioritize_rate_limited(candidates: list) -> list:
    """Move 429/cooldown providers to the tail so we skip them without a failed request."""
    try:
        from .api import _get_proxy
        rl = _get_proxy()._rate_limiter
    except Exception:
        return candidates

    def _exhausted(c) -> bool:
        pname = c.extra_config.get("yaml_provider") or c.provider_type or ""
        if pname and rl.is_exhausted(pname):
            return True
        pool_rl = f"pool_{c.key}"
        return rl.is_exhausted(pool_rl)

    available = [c for c in candidates if not _exhausted(c)]
    cooled = [c for c in candidates if _exhausted(c)]
    if cooled:
        names = [c.extra_config.get("yaml_provider") or c.provider_type for c in cooled]
        logger.info("[POOL] deprioritize rate-limited: %s", ", ".join(n for n in names if n))
    return available + cooled


def _cloud_yaml_priority(
    pname: str, model: str, role: str, fallback_order: list[str], multimodal: bool,
) -> int:
    """Lower = tried earlier. Respect fallback_order; :free gets a small boost, not global #1."""
    try:
        idx = fallback_order.index(pname)
    except ValueError:
        idx = len(fallback_order) + 3
    base = idx * 10
    if ":free" in (model or "").lower():
        base -= 2
    if role in ("vision", "ocr") and multimodal:
        base -= 1
    return max(0, base)


def _guess_registry_key(yaml_key: str) -> str:
    """Map config.yaml key (qwen2_5_3b) → MODEL_REGISTRY key (qwen2.5-3b)."""
    from .models import MODEL_REGISTRY

    if yaml_key in MODEL_REGISTRY:
        return yaml_key
    for reg_key in MODEL_REGISTRY:
        norm = reg_key.replace(".", "_").replace("-", "_")
        if norm == yaml_key:
            return reg_key
    return yaml_key.replace("_", "-")


def _map_provider_name(name: str) -> str:
    if name in ("google_ai", "gemma"):
        return "google_ai"
    if name == "openrouter":
        return "openrouter"
    return "openai_compat"


_pool: Optional[ModelPool] = None


def get_pool() -> ModelPool:
    global _pool
    if _pool is None:
        _pool = ModelPool()
    return _pool


def reload_pool() -> ModelPool:
    global _pool
    if _pool is None:
        _pool = ModelPool()
    else:
        _pool.reload()
    return _pool
