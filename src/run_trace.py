"""Request-scoped run trace: route → model → agent tools (observability)."""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_current: ContextVar[RunTrace | None] = ContextVar("run_trace", default=None)


@dataclass
class RunTrace:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    route_source: str = ""
    expert: str = ""
    model: str = ""
    agent_mode: str = ""
    tool_calls: int = 0
    rounds: int = 0
    phases: list[str] = field(default_factory=list)

    def phase(self, name: str) -> None:
        if name not in self.phases:
            self.phases.append(name)
        logger.info(
            "[TRACE:%s] %s expert=%s model=%s mode=%s",
            self.trace_id, name, self.expert or "—", self.model or "—", self.agent_mode or "—",
        )

    def bind(self) -> RunTrace:
        _current.set(self)
        return self

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "route_source": self.route_source,
            "expert": self.expert,
            "model": self.model,
            "agent_mode": self.agent_mode,
            "tool_calls": self.tool_calls,
            "rounds": self.rounds,
            "phases": list(self.phases),
        }


def get_trace() -> RunTrace | None:
    return _current.get()


def trace_payload() -> dict:
    t = get_trace()
    return t.to_dict() if t else {}


def merge_trace(payload: dict) -> dict:
    """Attach current trace metadata to an SSE/API payload."""
    tp = trace_payload()
    if tp:
        payload["trace"] = tp
    return payload
