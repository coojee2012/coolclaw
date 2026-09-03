"""CoolClaw Skill System — discover, register, and execute user-defined skills.

Usage:
    from src.skills import SkillRegistry, run_skill

    registry = SkillRegistry(workdir="/path/to/project")
    registry.scan()

    # List available skills
    for name, skill in registry.skills.items():
        print(f"{name}: {skill.description}")

    # Run a skill
    result = run_skill("weather", {"city": "北京"})
"""

from src.skills.registry import SkillRegistry, SkillInfo
from src.skills.runner import run_skill

__all__ = ["SkillRegistry", "SkillInfo", "run_skill"]
