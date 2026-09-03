"""SQLite database module for user management, auth, settings, and preferences.

Tables:
- users: user accounts with bcrypt-style password hashing
- auth_sessions: token-based session tracking
- settings: admin-configurable system settings (key-value)
- user_preferences: per-user config (user_id + key -> value)
- model_configs: cloud/local model configurations (provider, role, rate limits)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DB_DIR = Path.home() / ".opencode_helper"
_DB_PATH = _DB_DIR / "coolclaw.db"


def _hash_password(password: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    """Hash password with PBKDF2-HMAC-SHA256. Returns (hash, salt)."""
    if salt is None:
        salt = secrets.token_bytes(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return key, salt


def _verify_password(password: str, stored_hash: str) -> bool:
    """Verify password against stored hash (hex-encoded salt:hash)."""
    try:
        salt_hex, hash_hex = stored_hash.split(":")
        salt = bytes.fromhex(salt_hex)
        computed, _ = _hash_password(password, salt)
        return secrets.compare_digest(computed.hex(), hash_hex)
    except Exception:
        return False


def _hash_token(token: str) -> str:
    """Hash token for storage (tokens stored as hashes, not plaintext)."""
    return hashlib.sha256(token.encode()).hexdigest()


class Database:
    """Singleton SQLite database manager."""

    _instance: Optional["Database"] = None

    def __new__(cls) -> "Database":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._conn: Optional[sqlite3.Connection] = None
            cls._instance._init_done = False
        return cls._instance

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            _DB_DIR.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def init_db(self) -> None:
        """Create tables and default admin user."""
        if self._init_done:
            return
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                is_admin BOOLEAN DEFAULT 0,
                display_name TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS auth_sessions (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (user_id, key)
            );
        """)
        conn.commit()

        existing_tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}

        if "model_configs" not in existing_tables:
            conn.execute("""
                CREATE TABLE model_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider_type TEXT NOT NULL DEFAULT 'openai',
                    model_name TEXT NOT NULL,
                    display_name TEXT DEFAULT '',
                    api_key TEXT DEFAULT '',
                    base_url TEXT DEFAULT '',
                    role TEXT DEFAULT 'general',
                    priority INTEGER DEFAULT 10,
                    rpd INTEGER DEFAULT 0,
                    rpm INTEGER DEFAULT 0,
                    tpm INTEGER DEFAULT 0,
                    is_active BOOLEAN DEFAULT 1,
                    extra_config TEXT DEFAULT '{}',
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.commit()
            logger.info("Migrated: created model_configs table")
        conn.commit()

        # Create default admin if no users exist
        row = conn.execute("SELECT COUNT(*) as cnt FROM users").fetchone()
        if row["cnt"] == 0:
            self.create_user("admin", "admin123", is_admin=True)
            logger.info("Default admin user created (username=admin, password=admin123)")

        # Seed default system settings
        defaults = {
            "routing_mode": "cloud_only",
            "http_proxy": "",
            "https_proxy": "",
            "cloud_model": "gemini-3.5-flash-lite",
            "local_model": "qwen2.5-coder-7b",
            "rate_limit_rpm": "10",
            "rate_limit_burst": "5",
            "max_iterations": "20",
            "context_limit": "12000",
        }
        for key, value in defaults.items():
            existing = conn.execute("SELECT key FROM settings WHERE key = ?", (key,)).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                    (key, value),
                )
        conn.commit()

        self._init_done = True
        logger.info("Database initialized at %s", _DB_PATH)

    # ── Users ────────────────────────────────────────────────────────────

    def create_user(self, username: str, password: str, is_admin: bool = False, display_name: str = "") -> dict:
        """Create a new user. Returns user dict or raises on duplicate."""
        conn = self._get_conn()
        key, salt = _hash_password(password)
        password_hash = f"{salt.hex()}:{key.hex()}"
        now = datetime.utcnow().isoformat()
        cursor = conn.execute(
            "INSERT INTO users (username, password_hash, is_admin, display_name, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (username, password_hash, int(is_admin), display_name, now, now),
        )
        conn.commit()
        return {"id": cursor.lastrowid, "username": username, "is_admin": bool(is_admin), "display_name": display_name, "created_at": now}

    def authenticate_user(self, username: str, password: str) -> dict | None:
        """Authenticate user credentials. Returns user dict or None."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if not row:
            return None
        if not _verify_password(password, row["password_hash"]):
            return None
        return dict(row)

    def get_user(self, user_id: int) -> dict | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None

    def get_user_by_username(self, username: str) -> dict | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None

    def list_users(self) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute("SELECT id, username, is_admin, display_name, created_at FROM users ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def update_user(self, user_id: int, **kwargs) -> dict | None:
        conn = self._get_conn()
        allowed = {"username", "password", "is_admin", "display_name"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return self.get_user(user_id)
        if "password" in updates:
            key, salt = _hash_password(updates.pop("password"))
            updates["password_hash"] = f"{salt.hex()}:{key.hex()}"
        if "is_admin" in updates:
            updates["is_admin"] = int(updates["is_admin"])
        updates["updated_at"] = datetime.utcnow().isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [user_id]
        conn.execute(f"UPDATE users SET {set_clause} WHERE id = ?", values)
        conn.commit()
        return self.get_user(user_id)

    def delete_user(self, user_id: int) -> bool:
        conn = self._get_conn()
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        return conn.total_changes > 0

    # ── Auth Sessions ────────────────────────────────────────────────────

    def create_auth_session(self, user_id: int, token: str, expires_at: str) -> bool:
        conn = self._get_conn()
        token_hash = _hash_token(token)
        conn.execute(
            "INSERT OR REPLACE INTO auth_sessions (token_hash, user_id, expires_at, created_at) VALUES (?, ?, ?, datetime('now'))",
            (token_hash, user_id, expires_at),
        )
        conn.commit()
        return True

    def get_auth_session(self, token: str) -> dict | None:
        conn = self._get_conn()
        token_hash = _hash_token(token)
        row = conn.execute(
            "SELECT s.*, u.username, u.is_admin, u.display_name FROM auth_sessions s JOIN users u ON s.user_id = u.id WHERE s.token_hash = ? AND s.expires_at > datetime('now')",
            (token_hash,),
        ).fetchone()
        return dict(row) if row else None

    def delete_auth_session(self, token: str) -> bool:
        conn = self._get_conn()
        token_hash = _hash_token(token)
        conn.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (token_hash,))
        conn.commit()
        return True

    def cleanup_expired_sessions(self) -> int:
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM auth_sessions WHERE expires_at <= datetime('now')")
        conn.commit()
        return cursor.rowcount

    # ── Settings ─────────────────────────────────────────────────────────

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        conn = self._get_conn()
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> bool:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, datetime('now'))",
            (key, str(value)),
        )
        conn.commit()
        return True

    def get_all_settings(self) -> dict[str, str]:
        conn = self._get_conn()
        rows = conn.execute("SELECT key, value FROM settings ORDER BY key").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def delete_setting(self, key: str) -> bool:
        conn = self._get_conn()
        conn.execute("DELETE FROM settings WHERE key = ?", (key,))
        conn.commit()
        return True

    # ── User Preferences ─────────────────────────────────────────────────

    def get_user_preference(self, user_id: int, key: str, default: str | None = None) -> str | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT value FROM user_preferences WHERE user_id = ? AND key = ?",
            (user_id, key),
        ).fetchone()
        return row["value"] if row else default

    def set_user_preference(self, user_id: int, key: str, value: str) -> bool:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO user_preferences (user_id, key, value) VALUES (?, ?, ?)",
            (user_id, key, str(value)),
        )
        conn.commit()
        return True

    def get_all_user_preferences(self, user_id: int) -> dict[str, str]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT key, value FROM user_preferences WHERE user_id = ? ORDER BY key",
            (user_id,),
        ).fetchall()
        return {r["key"]: r["value"] for r in rows}

    def delete_user_preference(self, user_id: int, key: str) -> bool:
        conn = self._get_conn()
        conn.execute("DELETE FROM user_preferences WHERE user_id = ? AND key = ?", (user_id, key))
        conn.commit()
        return True

    # ── Model Configs ────────────────────────────────────────────────────

    def list_model_configs(self, provider_type: str | None = None, is_active: bool | None = None) -> list[dict]:
        conn = self._get_conn()
        query = "SELECT * FROM model_configs"
        conditions = []
        params: list[Any] = []
        if provider_type:
            conditions.append("provider_type = ?")
            params.append(provider_type)
        if is_active is not None:
            conditions.append("is_active = ?")
            params.append(int(is_active))
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY priority ASC, id ASC"
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_model_config(self, model_id: int) -> dict | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM model_configs WHERE id = ?", (model_id,)).fetchone()
        return dict(row) if row else None

    def get_model_configs_by_role(self, role: str) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM model_configs WHERE role = ? AND is_active = 1 ORDER BY priority ASC",
            (role,),
        ).fetchall()
        return [dict(r) for r in rows]

    def create_model_config(
        self,
        provider_type: str,
        model_name: str,
        display_name: str = "",
        api_key: str = "",
        base_url: str = "",
        role: str = "general",
        priority: int = 10,
        rpd: int = 0,
        rpm: int = 0,
        tpm: int = 0,
        is_active: bool = True,
        extra_config: dict | None = None,
    ) -> dict:
        conn = self._get_conn()
        now = datetime.utcnow().isoformat()
        cursor = conn.execute(
            """INSERT INTO model_configs
               (provider_type, model_name, display_name, api_key, base_url,
                role, priority, rpd, rpm, tpm, is_active, extra_config, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                provider_type, model_name, display_name, api_key, base_url,
                role, priority, rpd, rpm, tpm, int(is_active),
                json.dumps(extra_config or {}), now, now,
            ),
        )
        conn.commit()
        return self.get_model_config(cursor.lastrowid)  # type: ignore[return-value]

    def update_model_config(self, model_id: int, **kwargs: Any) -> dict | None:
        conn = self._get_conn()
        allowed = {
            "provider_type", "model_name", "display_name", "api_key", "base_url",
            "role", "priority", "rpd", "rpm", "tpm", "is_active", "extra_config",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return self.get_model_config(model_id)
        if "is_active" in updates:
            updates["is_active"] = int(updates["is_active"])
        if "extra_config" in updates and isinstance(updates["extra_config"], dict):
            updates["extra_config"] = json.dumps(updates["extra_config"])
        updates["updated_at"] = datetime.utcnow().isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [model_id]
        conn.execute(f"UPDATE model_configs SET {set_clause} WHERE id = ?", values)
        conn.commit()
        return self.get_model_config(model_id)

    def delete_model_config(self, model_id: int) -> bool:
        conn = self._get_conn()
        conn.execute("DELETE FROM model_configs WHERE id = ?", (model_id,))
        conn.commit()
        return True

    def seed_model_configs_from_yaml(self, config_data: dict) -> int:
        conn = self._get_conn()
        existing = conn.execute("SELECT COUNT(*) as cnt FROM model_configs").fetchone()
        if existing and existing["cnt"] > 0:
            return 0

        imported = 0
        proxy_providers = config_data.get("proxy", {}).get("providers", {})
        rate_limits = config_data.get("proxy", {}).get("rate_limits", {})

        for name, provider in proxy_providers.items():
            rl = rate_limits.get(name, {})
            self.create_model_config(
                provider_type="cloud",
                model_name=provider.get("model", ""),
                display_name=name,
                api_key=provider.get("api_key", ""),
                base_url=provider.get("base_url", ""),
                role="general",
                priority=10 + imported,
                rpd=rl.get("rpd", 0),
                rpm=rl.get("rpm", 0),
                tpm=0,
                is_active=True,
                extra_config={"timeout": provider.get("timeout", 180)},
            )
            imported += 1

        local_models = config_data.get("models", {}).get("local", {})
        for key, model in local_models.items():
            if key == "default" or not isinstance(model, dict):
                continue
            self.create_model_config(
                provider_type="local",
                model_name=model.get("path", ""),
                display_name=key.replace("_", "-"),
                role=model.get("role", "general"),
                priority=10 + imported,
                is_active=True,
                extra_config={
                    k: v for k, v in model.items()
                    if k not in ("path", "role")
                },
            )
            imported += 1

        return imported


db = Database()
