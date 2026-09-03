"""Skill runner — execute skills safely with timeout and error handling.

The runner imports the skill's main.py, calls its run() function,
and returns structured results.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from src.skills.registry import SkillInfo, SkillRegistry

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30  # seconds


def run_skill(
    name: str,
    params: dict[str, Any],
    *,
    registry: SkillRegistry | None = None,
    workdir: str = "",
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Execute a skill by name.

    Args:
        name: Skill name (e.g., "weather")
        params: Parameters to pass to the skill's run() function
        registry: Optional pre-built registry (will create one if not provided)
        workdir: Working directory (used if registry is not provided)
        timeout: Max execution time in seconds

    Returns:
        {
            "success": True/False,
            "skill": "skill_name",
            "result": <skill output>,
            "error": "error message if failed",
            "duration_ms": 1234
        }
    """
    # Get or create registry
    if registry is None:
        registry = SkillRegistry(workdir=workdir)
        registry.scan()

    # Find skill
    skill = registry.get_skill(name)
    if skill is None:
        available = list(registry.skills.keys())
        return {
            "success": False,
            "skill": name,
            "result": None,
            "error": f"Skill '{name}' not found. Available: {available}",
            "duration_ms": 0,
        }

    # Check dependencies
    missing = _check_dependencies(skill.dependencies)
    if missing:
        return {
            "success": False,
            "skill": name,
            "result": None,
            "error": f"Missing dependencies: {', '.join(missing)}. Install with: pip install {' '.join(missing)}",
            "duration_ms": 0,
        }

    # Validate parameters
    validation_error = registry.validate_params(skill, params)
    if validation_error:
        return {
            "success": False,
            "skill": name,
            "result": None,
            "error": validation_error,
            "duration_ms": 0,
        }

    # Merge defaults
    merged_params = _apply_defaults(skill, params)

    # Execute
    start = time.monotonic()
    try:
        result = _execute_skill(skill, merged_params, timeout)
        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "success": True,
            "skill": name,
            "result": result,
            "error": None,
            "duration_ms": duration_ms,
        }
    except TimeoutError:
        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "success": False,
            "skill": name,
            "result": None,
            "error": f"Skill timed out after {timeout}s",
            "duration_ms": duration_ms,
        }
    except PermissionError as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "success": False,
            "skill": name,
            "result": None,
            "error": f"Permission denied: {exc}",
            "duration_ms": duration_ms,
        }
    except Exception as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.exception("Skill '%s' failed", name)
        return {
            "success": False,
            "skill": name,
            "result": None,
            "error": f"{type(exc).__name__}: {exc}",
            "duration_ms": duration_ms,
        }


def _check_dependencies(deps: list[str]) -> list[str]:
    """Check if Python packages are installed."""
    missing = []
    for dep in deps:
        # Normalize: "Pillow" -> "pillow", "beautifulsoup4" -> "beautifulsoup4"
        try:
            __import__(dep.replace("-", "_").lower())
        except ImportError:
            # Try importlib
            try:
                __import__(dep.replace("-", "_"))
            except ImportError:
                missing.append(dep)
    return missing


def _apply_defaults(skill: SkillInfo, params: dict[str, Any]) -> dict[str, Any]:
    """Merge parameter defaults into params."""
    merged = dict(params)
    for pname, param in skill.parameters.items():
        if pname not in merged and param.default is not None:
            merged[pname] = param.default
    return merged


def _execute_skill(skill: SkillInfo, params: dict[str, Any], timeout: int) -> Any:
    """Execute a skill's run() function.

    Passes params via stdin to avoid shell-escaping issues with
    control characters, quotes, etc.
    """
    entry_path = skill.path / skill.entry
    if not entry_path.exists():
        raise FileNotFoundError(f"Skill entry not found: {entry_path}")

    params_json = json.dumps(params, ensure_ascii=False, default=str)
    runner_script = """\
import sys
import json
from pathlib import Path

# Add skill directory to path for local imports
sys.path.insert(0, sys.argv[1])

# Import and run
import importlib.util
spec = importlib.util.spec_from_file_location("skill_main", sys.argv[2])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

params = json.loads(sys.stdin.read())
result = module.run(**params)

# Output as JSON
print(json.dumps(result, ensure_ascii=False, default=str))
"""
    # Run in subprocess for isolation; params passed via stdin
    result = subprocess.run(
        [sys.executable, "-c", runner_script, str(skill.path), str(entry_path)],
        input=params_json,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(skill.path),
    )

    if result.returncode != 0:
        error_msg = result.stderr.strip() if result.stderr else "Unknown error"
        raise RuntimeError(f"Skill execution failed: {error_msg}")

    if not result.stdout.strip():
        return {"message": "(no output)"}

    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return {"output": result.stdout.strip()}


def format_skill_result(result: dict[str, Any]) -> str:
    """Format a skill result as human-readable text."""
    if not result.get("success"):
        return f"Error: {result.get('error', 'Unknown error')}"

    skill_result = result.get("result", {})
    duration = result.get("duration_ms", 0)

    if isinstance(skill_result, dict):
        lines = []
        for key, value in skill_result.items():
            if isinstance(value, list):
                lines.append(f"**{key}:**")
                for item in value:
                    if isinstance(item, dict):
                        parts = [f"{k}: {v}" for k, v in item.items()]
                        lines.append(f"  - {', '.join(parts)}")
                    else:
                        lines.append(f"  - {item}")
            else:
                lines.append(f"**{key}:** {value}")
        text = "\n".join(lines)
    elif isinstance(skill_result, list):
        text = "\n".join(f"- {item}" for item in skill_result)
    else:
        text = str(skill_result)

    return f"{text}\n\n_(completed in {duration}ms)_"
