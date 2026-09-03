"""Runtime reload and optional process restart."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_restart_argv: Optional[list[str]] = None
_restart_scheduled = False
_reload_hook: Optional[Callable[[], dict[str, Any]]] = None


def set_restart_argv(argv: Optional[list[str]] = None) -> None:
    global _restart_argv
    _restart_argv = list(argv) if argv else list(sys.argv)


def register_reload_hook(fn: Callable[[], dict[str, Any]]) -> None:
    """Register in-process config reload (called from create_app)."""
    global _reload_hook
    _reload_hook = fn


def reload_runtime() -> dict[str, Any]:
    """Hot-reload config.yaml into the running server (no process exit)."""
    from .config import reset_config, get_config
    from .model_pool import reload_pool

    reset_config()
    pool = reload_pool()
    cfg = get_config()

    extra: dict[str, Any] = {}
    if _reload_hook is not None:
        try:
            extra = _reload_hook()
        except Exception as e:
            logger.warning("Runtime reload hook failed: %s", e)
            extra = {"hook_error": str(e)}

    logger.info("Runtime config reloaded (mode=%s, pool=%d)", cfg.routing.mode, len(pool.all()))
    return {
        "reloaded": True,
        "mode": "hot_reload",
        "routing_mode": cfg.routing.mode,
        "pool_size": len(pool.all()),
        **extra,
    }


def _build_exec_args() -> list[str]:
    """Build argv for os.execv — must include script path (e.g. main.py)."""
    argv = list(_restart_argv or sys.argv)
    if not argv:
        return [sys.executable, "main.py"]
    # python3 main.py → argv=['main.py']; python -m src.cli → argv=['-m', 'src.cli', ...]
    if os.path.isabs(argv[0]) or argv[0] == sys.executable:
        return argv
    return [sys.executable, *argv]


def schedule_process_restart(delay: float = 1.5) -> bool:
    """Full process restart via os.execv (rarely needed)."""
    global _restart_scheduled
    if _restart_scheduled:
        return False
    _restart_scheduled = True
    exec_args = _build_exec_args()

    def _do_restart() -> None:
        logger.info("CoolClaw process restart: %s", " ".join(exec_args))
        os.execv(exec_args[0], exec_args)

    try:
        loop = asyncio.get_running_loop()
        loop.call_later(delay, _do_restart)
    except RuntimeError:
        threading.Timer(delay, _do_restart).start()

    logger.info("Process restart scheduled in %.1fs", delay)
    return True


# Backward-compatible alias
def schedule_restart(delay: float = 1.5) -> bool:
    return schedule_process_restart(delay)
