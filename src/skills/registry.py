"""Skill registry — auto-discover and register skills from directories.

Scans two locations:
1. Built-in skills: src/skills/builtin/
2. User skills: skills/ (in project workdir)

Each skill is a directory with:
- manifest.json: metadata (name, description, parameters, dependencies)
- main.py: execution entry point with a run() function
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default search paths (relative to project root)
_BUILTIN_DIR = Path(__file__).parent / "builtin"
_USER_DIR_NAME = "skills"


@dataclass(frozen=True)
class SkillParam:
    """Parameter definition for a skill."""

    name: str
    type: str  # "string", "integer", "float", "boolean", "array", "object"
    description: str = ""
    required: bool = False
    default: Any = None


@dataclass(frozen=True)
class SkillInfo:
    """Metadata about a discovered skill."""

    name: str
    description: str
    version: str = "1.0.0"
    parameters: dict[str, SkillParam] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    entry: str = "main.py"
    path: Path = field(default_factory=lambda: Path("."))
    source: str = "builtin"  # "builtin" or "user"

    @property
    def params_summary(self) -> str:
        """Human-readable parameter list."""
        if not self.parameters:
            return "(no parameters)"
        parts = []
        for p in self.parameters.values():
            req = " [required]" if p.required else ""
            default = f" (default: {p.default})" if p.default is not None else ""
            parts.append(f"  - {p.name} ({p.type}): {p.description}{req}{default}")
        return "\n".join(parts)


class SkillRegistry:
    """Discovers, registers, and manages skills."""

    def __init__(self, workdir: str = "") -> None:
        self.workdir = Path(workdir).expanduser().resolve() if workdir else Path.cwd()
        self.skills: dict[str, SkillInfo] = {}
        self._modules: dict[str, Any] = {}  # cached imported modules

    def scan(self) -> dict[str, SkillInfo]:
        """Scan all skill directories and register discovered skills."""
        self.skills.clear()
        self._modules.clear()

        # 1. Built-in skills
        if _BUILTIN_DIR.is_dir():
            self._scan_directory(_BUILTIN_DIR, source="builtin")

        # 2. User skills (in project workdir)
        user_dir = self.workdir / _USER_DIR_NAME
        if user_dir.is_dir():
            self._scan_directory(user_dir, source="user")

        logger.info("Discovered %d skills: %s", len(self.skills), list(self.skills.keys()))
        return self.skills

    def _scan_directory(self, directory: Path, source: str) -> None:
        """Scan a directory for skill subdirectories."""
        for child in sorted(directory.iterdir()):
            if not child.is_dir():
                continue
            manifest_path = child / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                skill = self._load_skill(child, source)
                if skill:
                    self.skills[skill.name] = skill
            except Exception as exc:
                logger.warning("Failed to load skill from %s: %s", child, exc)

    def _load_skill(self, skill_dir: Path, source: str) -> SkillInfo | None:
        """Load a skill from its directory."""
        manifest_path = skill_dir / "manifest.json"
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Invalid manifest %s: %s", manifest_path, exc)
            return None

        name = raw.get("name", skill_dir.name)
        if not name:
            return None

        # Parse parameters
        params: dict[str, SkillParam] = {}
        for pname, pdef in raw.get("parameters", {}).items():
            if isinstance(pdef, dict):
                params[pname] = SkillParam(
                    name=pname,
                    type=pdef.get("type", "string"),
                    description=pdef.get("description", ""),
                    required=pdef.get("required", False),
                    default=pdef.get("default"),
                )

        return SkillInfo(
            name=name,
            description=raw.get("description", ""),
            version=raw.get("version", "1.0.0"),
            parameters=params,
            dependencies=raw.get("dependencies", []),
            entry=raw.get("entry", "main.py"),
            path=skill_dir,
            source=source,
        )

    def get_skill(self, name: str) -> SkillInfo | None:
        """Get a skill by name."""
        return self.skills.get(name)

    def get_module(self, skill: SkillInfo) -> Any:
        """Import and cache a skill's main.py module."""
        if skill.name in self._modules:
            return self._modules[skill.name]

        entry_path = skill.path / skill.entry
        if not entry_path.exists():
            raise FileNotFoundError(f"Skill entry point not found: {entry_path}")

        spec = importlib.util.spec_from_file_location(
            f"coolclaw_skill_{skill.name}", str(entry_path)
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load skill module: {entry_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        self._modules[skill.name] = module
        return module

    def validate_params(self, skill: SkillInfo, params: dict[str, Any]) -> str | None:
        """Validate parameters against skill definition.

        Returns error message or None if valid.
        """
        for pname, param in skill.parameters.items():
            if param.required and pname not in params:
                if param.default is not None:
                    continue  # has default, ok
                return f"Missing required parameter: {pname}"
            if pname in params:
                value = params[pname]
                expected = param.type
                if expected == "string" and not isinstance(value, str):
                    return f"Parameter '{pname}' must be a string, got {type(value).__name__}"
                if expected == "integer" and not isinstance(value, int):
                    return f"Parameter '{pname}' must be an integer, got {type(value).__name__}"
                if expected == "float" and not isinstance(value, (int, float)):
                    return f"Parameter '{pname}' must be a number, got {type(value).__name__}"
                if expected == "boolean" and not isinstance(value, bool):
                    return f"Parameter '{pname}' must be a boolean, got {type(value).__name__}"
        return None
