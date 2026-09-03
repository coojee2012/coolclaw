"""Sandboxed file operations — per-session directory confinement."""

from __future__ import annotations

import base64
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 512 * 1024
MAX_LIST_ENTRIES = 500
IGNORE_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}
IGNORE_FILES = {".DS_Store", "Thumbs.db"}


def session_workdir(session_id: str, base: str = "") -> Path:
    if base:
        root = Path(base).expanduser().resolve()
    else:
        root = Path("data/sessions").resolve()
        safe_id = re.sub(r"[^\w\-]", "_", session_id)[:80] or "default"
        root = (root / safe_id).resolve()
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
    return root


def safe_resolve(root: Path, rel: str) -> Path:
    cleaned = rel.replace("\\", "/").lstrip("/")
    target = (root / cleaned).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes workspace: {rel}")
    return target


def _file_kind(p: Path) -> str:
    if p.is_symlink():
        return "symlink"
    if p.is_dir():
        return "dir"
    return "file"


@dataclass
class FileEntry:
    name: str
    kind: str
    size: int
    mtime: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "size": self.size,
            "mtime": self.mtime,
        }


def list_files(root: Path, rel: str = ".", limit: int = MAX_LIST_ENTRIES) -> dict:
    target = safe_resolve(root, rel)
    if not target.is_dir():
        return {"entries": [], "truncated": False, "path": rel}
    entries = []
    truncated = False
    try:
        for item in sorted(target.iterdir()):
            if item.name in IGNORE_FILES:
                continue
            if item.is_dir() and item.name in IGNORE_DIRS:
                continue
            try:
                st = item.stat(follow_symlinks=False)
                entries.append(
                    FileEntry(
                        name=item.name,
                        kind=_file_kind(item),
                        size=st.st_size,
                        mtime=st.st_mtime,
                    )
                )
            except (PermissionError, OSError):
                continue
            if len(entries) >= limit:
                truncated = True
                break
    except PermissionError:
        return {"entries": [], "truncated": False, "path": rel, "error": "permission denied"}
    return {"entries": [e.to_dict() for e in entries], "truncated": truncated, "path": rel}


def read_file(root: Path, rel: str, max_bytes: int = MAX_FILE_SIZE) -> dict:
    target = safe_resolve(root, rel)
    if not target.exists():
        raise FileNotFoundError(f"Not found: {rel}")
    if target.is_dir():
        raise IsADirectoryError(f"Is a directory: {rel}")
    size = target.stat().st_size
    if size > max_bytes:
        return {
            "content_b64": "",
            "truncated": True,
            "size": size,
            "encoding": "binary",
            "error": f"File too large ({size} bytes, max {max_bytes})",
        }
    try:
        text = target.read_text(encoding="utf-8")
        return {"content": text, "size": size, "encoding": "utf-8", "truncated": False}
    except UnicodeDecodeError:
        raw = target.read_bytes()
        return {
            "content_b64": base64.b64encode(raw).decode(),
            "size": size,
            "encoding": "binary",
            "truncated": False,
        }


def write_file(root: Path, rel: str, content: str) -> dict:
    target = safe_resolve(root, rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    size = target.stat().st_size
    logger.info(f"[FILE] wrote {rel} ({size} bytes)")
    return {"status": "ok", "size": size, "path": rel}


def edit_file(root: Path, rel: str, old_str: str, new_str: str) -> dict:
    target = safe_resolve(root, rel)
    if not target.exists():
        raise FileNotFoundError(f"Not found: {rel}")
    content = target.read_text(encoding="utf-8")
    occurrences = content.count(old_str)
    if occurrences == 0:
        return {"status": "error", "message": "old_str not found in file"}
    if occurrences > 1:
        lines = [i + 1 for i, line in enumerate(content.split("\n")) if old_str in line]
        return {"status": "error", "message": f"Ambiguous: {occurrences} matches at lines {lines}"}
    new_content = content.replace(old_str, new_str, 1)
    target.write_text(new_content, encoding="utf-8")
    logger.info(f"[FILE] edited {rel} (replaced 1 occurrence)")
    return {"status": "ok", "path": rel}


def delete_file(root: Path, rel: str, recursive: bool = False) -> dict:
    target = safe_resolve(root, rel)
    if not target.exists():
        raise FileNotFoundError(f"Not found: {rel}")
    if target.resolve() == root.resolve():
        return {"status": "error", "message": "Cannot delete workspace root"}
    if target.is_dir():
        if recursive:
            import shutil
            shutil.rmtree(target)
        else:
            target.rmdir()
    else:
        target.unlink()
    logger.info(f"[FILE] deleted {rel}")
    return {"status": "ok", "path": rel}
