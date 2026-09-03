import uuid
import json
import logging
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .memory import Message

logger = logging.getLogger(__name__)


@dataclass
class ChatSession:
    id: str
    name: str
    user_id: int = 0
    messages: list[Message] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    model: str = ""
    workdir: str = ""
    summary: str = ""
    summary_upto: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "user_id": self.user_id,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                    "model": m.model,
                    "metadata": m.metadata,
                }
                for m in self.messages
            ],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "model": self.model,
            "workdir": self.workdir,
            "summary": self.summary,
            "summary_upto": self.summary_upto,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChatSession":
        messages = []
        for m in data.get("messages", []):
            messages.append(
                Message(
                    role=m["role"],
                    content=m["content"],
                    timestamp=datetime.fromisoformat(m["timestamp"]),
                    model=m.get("model", ""),
                    metadata=m.get("metadata", {}),
                )
            )
        return cls(
            id=data["id"],
            name=data["name"],
            user_id=data.get("user_id", 0),
            messages=messages,
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            model=data.get("model", ""),
            workdir=data.get("workdir", ""),
            summary=data.get("summary", ""),
            summary_upto=int(data.get("summary_upto", 0) or 0),
        )


class SessionManager:
    _instance: Optional["SessionManager"] = None
    _sessions_dir: Path = Path.home() / ".opencode_helper" / "sessions"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._sessions: dict[str, ChatSession] = {}
            cls._instance._current_by_user: dict[int, str] = {}
            cls._instance._load_all()
        return cls._instance

    def _load_all(self):
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        for session_file in self._sessions_dir.glob("*.json"):
            try:
                with open(session_file) as f:
                    data = json.load(f)
                    session = ChatSession.from_dict(data)
                    self._sessions[session.id] = session
            except Exception as e:
                print(f"Failed to load session {session_file}: {e}")

    def _save(self, session: ChatSession):
        session.updated_at = datetime.now().isoformat()
        session_file = self._sessions_dir / f"{session.id}.json"
        with open(session_file, "w") as f:
            json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)

    def create_session(self, name: str = "", user_id: int = 0) -> ChatSession:
        session_id = str(uuid.uuid4())[:8]
        if not name:
            user_count = sum(1 for s in self._sessions.values() if s.user_id == user_id)
            name = f"会话 {user_count + 1}"
        session = ChatSession(id=session_id, name=name, user_id=user_id)
        self._sessions[session_id] = session
        self._current_by_user[user_id] = session_id
        self._save(session)
        return session

    def get_session(self, session_id: str, user_id: Optional[int] = None) -> Optional[ChatSession]:
        session = self._sessions.get(session_id)
        if not session:
            return None
        if user_id is not None and session.user_id != user_id:
            return None
        return session

    def list_sessions(self, user_id: int) -> list[dict]:
        return [
            {
                "id": s.id,
                "name": s.name,
                "message_count": len(s.messages),
                "created_at": s.created_at,
                "updated_at": s.updated_at,
                "model": s.model,
                "workdir": s.workdir or "",
            }
            for s in sorted(
                (s for s in self._sessions.values() if s.user_id == user_id),
                key=lambda x: x.updated_at,
                reverse=True,
            )
        ]

    def update_session(
        self,
        session_id: str,
        user_id: int,
        name: str = None,
        model: str = None,
        workdir: str = None,
    ) -> Optional[ChatSession]:
        session = self.get_session(session_id, user_id)
        if not session:
            return None
        if name is not None:
            session.name = name
        if model is not None:
            session.model = model
        if workdir is not None:
            session.workdir = workdir
        self._save(session)
        return session

    def set_workdir(self, session_id: str, user_id: int, path: str) -> Optional[ChatSession]:
        return self.update_session(session_id, user_id, workdir=path)

    def delete_session(self, session_id: str, user_id: int) -> bool:
        session = self.get_session(session_id, user_id)
        if not session:
            return False
        session_file = self._sessions_dir / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
        del self._sessions[session_id]
        try:
            from .session_memory import get_session_memory
            get_session_memory().clear_session(session_id)
        except Exception:
            pass
        if self._current_by_user.get(user_id) == session_id:
            remaining = [
                s.id for s in self._sessions.values() if s.user_id == user_id
            ]
            if remaining:
                self._current_by_user[user_id] = remaining[0]
            else:
                self._current_by_user.pop(user_id, None)
        return True

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        model: str = "",
        metadata: dict = None,
        user_id: Optional[int] = None,
    ) -> Optional[ChatSession]:
        session = self.get_session(session_id, user_id)
        if not session:
            return None
        session.messages.append(
            Message(role=role, content=content, model=model, metadata=metadata or {})
        )
        if model:
            session.model = model
        idx = len(session.messages) - 1
        try:
            from .session_memory import get_session_memory, maybe_summarize_session
            get_session_memory().index_message(session_id, idx, role, content)
        except Exception as e:
            logger.debug("session memory index: %s", e)
        self._save(session)
        try:
            import threading
            from .session_memory import maybe_summarize_session
            uid = user_id or session.user_id
            threading.Thread(
                target=maybe_summarize_session,
                args=(session_id, uid),
                daemon=True,
            ).start()
        except Exception as e:
            logger.debug("session summarize: %s", e)
        return session

    def clear_session(self, session_id: str, user_id: int) -> bool:
        session = self.get_session(session_id, user_id)
        if not session:
            return False
        session.messages = []
        session.summary = ""
        session.summary_upto = 0
        try:
            from .session_memory import get_session_memory
            get_session_memory().clear_session(session_id)
        except Exception:
            pass
        self._save(session)
        return True

    def get_current_session(self, user_id: int = 0) -> Optional[ChatSession]:
        session_id = self._current_by_user.get(user_id)
        if session_id:
            session = self.get_session(session_id, user_id)
            if session:
                return session
        user_sessions = [s for s in self._sessions.values() if s.user_id == user_id]
        if user_sessions:
            latest = max(user_sessions, key=lambda x: x.updated_at)
            self._current_by_user[user_id] = latest.id
            return latest
        return None

    def set_current_session(self, session_id: str, user_id: int = 0) -> bool:
        if not self.get_session(session_id, user_id):
            return False
        self._current_by_user[user_id] = session_id
        return True


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def build_session_context(
    messages: list[Message],
    max_tokens: int = 2500,
    summary: str = "",
    query: str = "",
    session_id: str = "",
) -> str:
    """Build LLM context: summary + vector recall + sliding recent window."""
    if not messages and not summary:
        return ""

    parts: list[str] = []
    budget = max_tokens

    if summary:
        line = f"[system] 会话摘要（较早对话）: {summary}"
        parts.append(line)
        budget -= _estimate_tokens(line)

    if session_id and query:
        try:
            from .session_memory import get_session_memory
            mem = get_session_memory()
            if messages:
                mem.ensure_indexed(session_id, messages)
            hits = mem.search(session_id, query, top_k=3)
            for hit in hits:
                line = f"[system] 相关历史: {hit}"
                t = _estimate_tokens(line)
                if t > budget:
                    break
                parts.append(line)
                budget -= t
        except Exception as e:
            logger.debug("session recall: %s", e)

    selected: list[str] = []
    total = 0
    omitted = 0

    for msg in reversed(messages):
        line = f"[{msg.role}] {msg.content}"
        tokens = _estimate_tokens(line)
        if total + tokens > budget:
            omitted += 1
            continue
        selected.insert(0, line)
        total += tokens

    if omitted:
        selected.insert(
            0,
            f"[system] （已压缩省略较早的 {omitted} 条消息，以下为最近对话）",
        )

    parts.extend(selected)
    return "\n".join(parts)


def build_dispatch_context(
    messages: list[Message],
    summary: str = "",
    max_tokens: int = 600,
) -> str:
    """Lightweight context for the local dispatcher (3B) — avoid context overflow."""
    parts: list[str] = []
    budget = max_tokens
    if summary:
        line = f"[system] 会话摘要: {summary[:400]}"
        parts.append(line)
        budget -= _estimate_tokens(line)
    recent = messages[-6:] if messages else []
    for msg in reversed(recent):
        line = f"[{msg.role}] {msg.content[:300]}"
        t = _estimate_tokens(line)
        if t > budget:
            break
        parts.insert(len(parts) - (1 if summary else 0), line)
        budget -= t
    return "\n".join(parts)


session_manager = SessionManager()
