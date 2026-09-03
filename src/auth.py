"""Authentication middleware and helpers for FastAPI.

Cookie-based auth with opaque tokens stored in SQLite.
"""

from __future__ import annotations

import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Request, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.database import db

logger = logging.getLogger(__name__)

COOKIE_NAME = "coolclaw_session"
COOKIE_MAX_DAYS = 7

# Paths that don't require authentication
_EXEMPT_PREFIXES = (
    "/login",
    "/register",
    "/health",
    "/static/",
    "/favicon",
)
_EXEMPT_EXACT = {
    "/login.html",
    "/api/auth/login",
    "/api/auth/register",
    "/api/auth/logout",
}


def _is_exempt(path: str) -> bool:
    """Check if a path is exempt from authentication."""
    if path in _EXEMPT_EXACT:
        return True
    for prefix in _EXEMPT_PREFIXES:
        if path.startswith(prefix):
            return True
    # Allow static asset file extensions
    if any(path.endswith(ext) for ext in (".html", ".css", ".js", ".ico", ".png", ".svg", ".jpg")):
        return True
    return False


class AuthMiddleware(BaseHTTPMiddleware):
    """Cookie-based authentication middleware."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip auth for exempt paths
        if _is_exempt(path):
            response = await call_next(request)
            return response

        # Check cookie
        token = request.cookies.get(COOKIE_NAME)
        if token:
            session_data = db.get_auth_session(token)
            if session_data:
                # Attach user info to request state
                request.state.user = {
                    "id": session_data["user_id"],
                    "username": session_data["username"],
                    "is_admin": bool(session_data["is_admin"]),
                    "display_name": session_data.get("display_name", ""),
                }
                response = await call_next(request)
                return response

        # Not authenticated — redirect to login for HTML requests
        accept = request.headers.get("accept", "")
        if "text/html" in accept or path.startswith("/api/"):
            if path.startswith("/api/"):
                return JSONResponse(status_code=401, content={"detail": "Not authenticated"})
            return RedirectResponse(url="/login.html", status_code=302)

        return JSONResponse(status_code=401, content={"detail": "Not authenticated"})


def create_session_token(user_id: int) -> str:
    """Generate and store an auth session token. Returns the token."""
    token = secrets.token_urlsafe(32)
    expires_at = (datetime.utcnow() + timedelta(days=COOKIE_MAX_DAYS)).isoformat()
    db.create_auth_session(user_id, token, expires_at)
    return token


def destroy_session_token(token: str) -> bool:
    """Delete an auth session token."""
    return db.delete_auth_session(token)


def get_current_user(request: Request) -> dict | None:
    """Get current user from request state (set by AuthMiddleware)."""
    return getattr(request.state, "user", None)


def require_auth(request: Request) -> dict:
    """Require authenticated user. Raises 401 if not."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def require_admin(request: Request) -> dict:
    """Require admin user. Raises 403 if not admin."""
    user = require_auth(request)
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user
