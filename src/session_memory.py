"""Per-session message index (vector + keyword fallback) and LLM summarization."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from .memory import Message

logger = logging.getLogger(__name__)

SUMMARIZE_INTERVAL = 20
KEEP_RECENT_MESSAGES = 8
SUMMARY_MAX_CHARS = 600


class SessionMemoryIndex:
    """Index chat messages for semantic recall within a session."""

    _instance: Optional["SessionMemoryIndex"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ready = False
            cls._instance._client = None
            cls._instance._collection = None
            cls._instance._keyword_cache: dict[str, list[dict]] = {}
            cls._instance._init_store()
        return cls._instance

    def _init_store(self):
        storage = Path.home() / ".opencode_helper" / "session_memory"
        storage.mkdir(parents=True, exist_ok=True)
        try:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(storage / "chroma"))
            self._collection = self._client.get_or_create_collection(
                name="session_messages",
                metadata={"description": "CoolClaw per-session chat recall"},
            )
            self._ready = True
            logger.info("[SESSION_MEM] ChromaDB index ready")
        except Exception as e:
            logger.warning("[SESSION_MEM] ChromaDB unavailable, keyword fallback: %s", e)
            self._ready = False

    def _doc_id(self, session_id: str, index: int) -> str:
        return f"{session_id}_{index}"

    def index_message(
        self,
        session_id: str,
        index: int,
        role: str,
        content: str,
    ) -> None:
        text = (content or "").strip()
        if not text or not session_id:
            return

        entry = {"session_id": session_id, "index": index, "role": role, "content": text}
        cache = self._keyword_cache.setdefault(session_id, [])
        if index < len(cache):
            cache[index] = entry
        else:
            while len(cache) < index:
                cache.append({"session_id": session_id, "index": len(cache), "role": "", "content": ""})
            cache.append(entry)

        if not self._ready or self._collection is None:
            return
        try:
            self._collection.upsert(
                ids=[self._doc_id(session_id, index)],
                documents=[f"[{role}] {text[:2000]}"],
                metadatas=[{"session_id": session_id, "index": index, "role": role}],
            )
        except Exception as e:
            logger.debug("[SESSION_MEM] index upsert failed: %s", e)

    def search(self, session_id: str, query: str, top_k: int = 3) -> list[str]:
        query = (query or "").strip()
        if not query or not session_id:
            return []

        if self._ready and self._collection is not None:
            try:
                result = self._collection.query(
                    query_texts=[query],
                    n_results=min(top_k, 8),
                    where={"session_id": session_id},
                )
                docs = (result.get("documents") or [[]])[0]
                return [d for d in docs if d][:top_k]
            except Exception as e:
                logger.debug("[SESSION_MEM] vector search failed: %s", e)

        return self._keyword_search(session_id, query, top_k)

    def _keyword_search(self, session_id: str, query: str, top_k: int) -> list[str]:
        entries = self._keyword_cache.get(session_id, [])
        if not entries:
            return []
        terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 1]
        if not terms:
            return []

        scored: list[tuple[float, str]] = []
        for e in entries:
            text = e.get("content", "")
            if not text:
                continue
            lower = text.lower()
            score = sum(1 for t in terms if t in lower)
            if score > 0:
                scored.append((score, f"[{e.get('role', '?')}] {text[:300]}"))
        scored.sort(key=lambda x: -x[0])
        return [s[1] for s in scored[:top_k]]

    def clear_session(self, session_id: str) -> None:
        self._keyword_cache.pop(session_id, None)
        if not self._ready or self._collection is None:
            return
        try:
            self._collection.delete(where={"session_id": session_id})
        except Exception as e:
            logger.debug("[SESSION_MEM] clear failed: %s", e)

    def ensure_indexed(self, session_id: str, messages: list[Message]) -> None:
        if session_id in self._keyword_cache and len(self._keyword_cache[session_id]) >= len(messages):
            return
        self.reindex_session(session_id, messages)

    def reindex_session(self, session_id: str, messages: list[Message]) -> None:
        self.clear_session(session_id)
        for i, m in enumerate(messages):
            self.index_message(session_id, i, m.role, m.content)


_index = SessionMemoryIndex()


def get_session_memory() -> SessionMemoryIndex:
    return _index


def _extractive_summary(messages: list[Message], prior: str = "") -> str:
    lines: list[str] = []
    if prior:
        lines.append(prior.strip())
    for m in messages:
        snippet = m.content.strip().replace("\n", " ")[:120]
        if not snippet:
            continue
        prefix = "用户" if m.role == "user" else "助手"
        lines.append(f"{prefix}: {snippet}")
    text = " | ".join(lines)
    return text[:SUMMARY_MAX_CHARS]


async def summarize_messages_async(
    messages: list[Message],
    prior_summary: str = "",
) -> str:
    """Summarize a batch of messages; cloud LLM with extractive fallback."""
    if not messages:
        return prior_summary

    body_lines = []
    for m in messages:
        body_lines.append(f"[{m.role}] {m.content[:500]}")
    body = "\n".join(body_lines)
    if len(body) > 6000:
        body = body[:6000] + "\n...(truncated)"

    prompt = (
        "请将以下对话压缩为中文要点摘要（不超过200字）。"
        "保留：关键事实、数字、用户偏好、未完成事项、重要结论。"
    )
    if prior_summary:
        prompt += f"\n\n已有摘要（请合并更新）：\n{prior_summary}"

    try:
        from .config import get_config
        from .model_pool import get_pool
        from .proxy import Proxy, ProxyRequest

        cfg = get_config()
        if cfg.proxy.enabled:
            pool = get_pool()
            cloud = [c for c in pool.all() if c.source == "cloud"][:4]
            if cloud:
                proxy = Proxy()
                req = ProxyRequest(
                    model=cloud[0].model_name,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": body},
                    ],
                    max_tokens=400,
                    temperature=0.2,
                )
                resp = await proxy.chat_with_candidates(cloud, req)
                if resp.choices:
                    content = resp.choices[0].get("message", {}).get("content", "")
                    if content.strip():
                        return content.strip()[:SUMMARY_MAX_CHARS]
    except Exception as e:
        logger.warning("[SESSION_MEM] cloud summarize failed: %s", e)

    merged = _extractive_summary(messages, prior_summary)
    return merged[:SUMMARY_MAX_CHARS]


def maybe_summarize_session(session_id: str, user_id: int) -> None:
    """Fold older messages into session.summary when interval threshold is reached."""
    from .session import session_manager

    session = session_manager.get_session(session_id, user_id)
    if not session:
        return

    pending = len(session.messages) - session.summary_upto - KEEP_RECENT_MESSAGES
    if pending < SUMMARIZE_INTERVAL:
        return

    batch = session.messages[session.summary_upto : session.summary_upto + SUMMARIZE_INTERVAL]
    if not batch:
        return

    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            from .dispatcher import _run_async
            summary = _run_async(
                summarize_messages_async(batch, session.summary or "")
            )
        else:
            summary = loop.run_until_complete(
                summarize_messages_async(batch, session.summary or "")
            )
    except Exception:
        summary = _extractive_summary(batch, session.summary or "")

    session.summary = summary
    session.summary_upto += len(batch)
    session_manager._save(session)
    logger.info(
        "[SESSION_MEM] summarized session %s: upto=%d chars=%d",
        session_id, session.summary_upto, len(summary),
    )
