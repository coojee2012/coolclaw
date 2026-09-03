from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path


@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    model: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return len(self.content) // 4


@dataclass
class ConversationMemory:
    messages: list[Message] = field(default_factory=list)
    max_history: int = 20

    def add(self, role: str, content: str, model: str = "", metadata: dict = None):
        self.messages.append(
            Message(role=role, content=content, model=model, metadata=metadata or {})
        )
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history :]

    def get_context(self, include_last: int = 10) -> str:
        recent = self.messages[-include_last:] if self.messages else []
        return "\n".join(f"[{m.role}] {m.content}" for m in recent)

    def get_context_with_limit(self, max_tokens: int = 1500) -> str:
        result = []
        total_tokens = 0
        for msg in reversed(self.messages):
            msg_tokens = msg.tokens
            if total_tokens + msg_tokens > max_tokens:
                if total_tokens == 0:
                    result.insert(0, f"[{msg.role}] {msg.content[: max_tokens * 4]}...")
                break
            result.insert(0, f"[{msg.role}] {msg.content}")
            total_tokens += msg_tokens
        return "\n".join(result)

    def clear(self):
        self.messages = []

    def to_dict(self) -> dict:
        return {
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                    "model": m.model,
                    "metadata": m.metadata,
                }
                for m in self.messages
            ]
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ConversationMemory":
        mem = cls()
        if Path(path).exists():
            with open(path) as f:
                data = json.load(f)
            for m in data.get("messages", []):
                mem.messages.append(
                    Message(
                        role=m["role"],
                        content=m["content"],
                        timestamp=datetime.fromisoformat(m["timestamp"]),
                        model=m.get("model", ""),
                        metadata=m.get("metadata", {}),
                    )
                )
        return mem


class ProjectMemory:
    """跨 session 的项目级持久记忆：决策记录、错误学习、上下文摘要"""

    def __init__(self, project_root: str):
        self.root = Path(project_root) / ".coolclaw_memory"
        self.root.mkdir(parents=True, exist_ok=True)
        self._decisions_file = self.root / "decisions.json"
        self._errors_file = self.root / "errors.json"
        self._summary_file = self.root / "summary.json"

    def _load_json(self, path: Path) -> list:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save_json(self, path: Path, data: list):
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def record_decision(self, topic: str, decision: str, reason: str = "", context: str = ""):
        entries = self._load_json(self._decisions_file)
        entries.append({
            "topic": topic,
            "decision": decision,
            "reason": reason,
            "context": context,
            "timestamp": datetime.now().isoformat(),
        })
        self._save_json(self._decisions_file, entries)

    def get_decisions(self, topic: str = "", limit: int = 20) -> list[dict]:
        entries = self._load_json(self._decisions_file)
        if topic:
            entries = [e for e in entries if topic.lower() in e.get("topic", "").lower()]
        return entries[-limit:]

    def record_error(self, tool: str, error: str, fix: str = "", context: str = ""):
        entries = self._load_json(self._errors_file)
        entries.append({
            "tool": tool,
            "error": error[:500],
            "fix": fix[:500],
            "context": context[:300],
            "timestamp": datetime.now().isoformat(),
        })
        self._save_json(self._errors_file, entries[-100:])

    def get_error_suggestions(self, tool: str = "", error_pattern: str = "") -> list[dict]:
        entries = self._load_json(self._errors_file)
        if tool:
            entries = [e for e in entries if e.get("tool") == tool]
        if error_pattern:
            pattern_lower = error_pattern.lower()
            entries = [e for e in entries if pattern_lower in e.get("error", "").lower()]
        return entries[-10:]

    def save_summary(self, project_name: str, summary: str, files_touched: list[str] = None):
        summaries = self._load_json(self._summary_file)
        summaries.append({
            "project": project_name,
            "summary": summary[:2000],
            "files": files_touched or [],
            "timestamp": datetime.now().isoformat(),
        })
        self._save_json(self._summary_file, summaries[-50:])

    def get_recent_summaries(self, limit: int = 5) -> list[dict]:
        return self._load_json(self._summary_file)[-limit:]

    def get_preferences(self) -> dict:
        prefs_file = self.root / "preferences.json"
        if prefs_file.exists():
            try:
                return json.loads(prefs_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def save_preferences(self, preferences: dict):
        prefs_file = self.root / "preferences.json"
        prefs_file.write_text(json.dumps(preferences, indent=2, ensure_ascii=False), encoding="utf-8")

    def learn_preference(self, key: str, value: str):
        prefs = self.get_preferences()
        prefs[key] = value
        self.save_preferences(prefs)

    def _detect_coding_style(self, file_content: str) -> dict:
        import re as _re
        style = {}
        if _re.search(r'^    \S', file_content, _re.MULTILINE):
            style['indent'] = '4 spaces'
        elif _re.search(r'^  \S', file_content, _re.MULTILINE):
            style['indent'] = '2 spaces'
        if _re.search(r'\t\S', file_content):
            style['indent'] = 'tabs'

        if _re.search(r'\bdef [a-z_]+_[a-z_]+\b', file_content):
            style['naming'] = 'snake_case'
        elif _re.search(r'\bfunction [a-zA-Z]+[A-Z]', file_content):
            style['naming'] = 'camelCase'

        if '# TODO' in file_content or '# FIXME' in file_content:
            style['comments'] = 'inline'

        return style

    def record_file_style(self, file_path: str, content: str):
        if not file_path.endswith(('.py', '.js', '.ts')):
            return

        style = self._detect_coding_style(content)
        if style:
            existing = self.get_preferences()
            existing.setdefault('coding_style', {})
            for key, value in style.items():
                existing['coding_style'][key] = value
            self.save_preferences(existing)

    def get_context_block(self, max_tokens: int = 800) -> str:
        lines = []
        decisions = self.get_decisions(limit=5)
        if decisions:
            lines.append("## Recent Decisions")
            for d in decisions:
                lines.append(f"- [{d['topic']}] → {d['decision']}")
                if d.get("reason"):
                    lines.append(f"  原因: {d['reason']}")

        errors = self.get_error_suggestions(limit=3)
        if errors:
            lines.append("\n## Known Issues & Fixes")
            for e in errors:
                lines.append(f"- {e['tool']}: {e['error'][:80]}")
                if e.get("fix"):
                    lines.append(f"  修复: {e['fix'][:80]}")

        summaries = self.get_recent_summaries(limit=2)
        if summaries:
            lines.append("\n## Recent Work")
            for s in summaries:
                lines.append(f"- [{s['project']}] {s['summary'][:120]}")

        prefs = self.get_preferences()
        coding_style = prefs.get('coding_style', {})
        if coding_style:
            lines.append("\n## Coding Style Preferences")
            for k, v in coding_style.items():
                lines.append(f"- {k}: {v}")

        result = "\n".join(lines)
        estimated_tokens = len(result) // 4
        if estimated_tokens > max_tokens:
            result = result[:max_tokens * 4]
        return result
