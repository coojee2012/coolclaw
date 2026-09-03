from typing import Optional, Any
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio


class CapabilityCategory(str, Enum):
    NETWORK = "network"
    FILE = "file"
    NOTIFICATION = "notification"
    SCHEDULE = "schedule"
    SYSTEM = "system"
    AI = "ai"


class CapabilityInput(BaseModel):
    class Config:
        extra = "allow"


class CapabilityOutput(BaseModel):
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


@dataclass
class Capability:
    name: str
    description: str
    category: CapabilityCategory
    input_schema: dict
    output_schema: dict
    memory_mb: float = 10.0
    examples: list[str] = field(default_factory=list)
    requires_auth: bool = False


class BaseCapability:
    capability: Capability = None

    def __init__(self):
        if self.capability is None:
            raise NotImplementedError("Capability not defined")

    async def execute(self, params: dict) -> CapabilityOutput:
        raise NotImplementedError("execute not implemented")

    def validate_params(self, params: dict) -> bool:
        return True


class CapabilityRegistry:
    _capabilities: dict[str, Capability] = {}
    _executors: dict[str, type[BaseCapability]] = {}

    @classmethod
    def register(cls, capability: Capability, executor: type[BaseCapability]):
        cls._capabilities[capability.name] = capability
        cls._executors[capability.name] = executor

    @classmethod
    def get(cls, name: str) -> Optional[Capability]:
        return cls._capabilities.get(name)

    @classmethod
    def get_executor(cls, name: str) -> Optional[type[BaseCapability]]:
        return cls._executors.get(name)

    @classmethod
    def list_all(cls) -> list[Capability]:
        return list(cls._capabilities.values())

    @classmethod
    def list_by_category(cls, category: CapabilityCategory) -> list[Capability]:
        return [c for c in cls._capabilities.values() if c.category == category]

    @classmethod
    async def execute(cls, name: str, params: dict) -> CapabilityOutput:
        executor_cls = cls._executors.get(name)
        if not executor_cls:
            return CapabilityOutput(
                success=False, error=f"Capability '{name}' not found"
            )
        try:
            executor = executor_cls()
            return await executor.execute(params)
        except Exception as e:
            return CapabilityOutput(success=False, error=str(e))


def capability(
    name: str,
    description: str,
    category: CapabilityCategory,
    input_schema: dict,
    output_schema: dict,
    memory_mb: float = 10.0,
    examples: list[str] = None,
    requires_auth: bool = False,
):
    def decorator(cls: type[BaseCapability]):
        cap = Capability(
            name=name,
            description=description,
            category=category,
            input_schema=input_schema,
            output_schema=output_schema,
            memory_mb=memory_mb,
            examples=examples or [],
            requires_auth=requires_auth,
        )
        CapabilityRegistry.register(cap, cls)
        cls.capability = cap
        return cls

    return decorator
