import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel
from pydantic import Field


class ModelConfig(BaseModel):
    path: str
    n_ctx: int = 8192
    n_gpu_layers: int = 35
    n_threads: int = 6
    n_batch: int = 512
    low_vram: bool = False
    use_turboquant: bool = False


class CloudModelConfig(BaseModel):
    api_key: str = Field(default="")
    model: str = "gemini-2.5-flash"
    safety_threshold: str = "BLOCK_MEDIUM_AND_ABOVE"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 8192


class LocalModelsConfig(BaseModel):
    default: str = "qwen2.5-coder-7b"
    qwen2_5_coder_7b: Optional[ModelConfig] = None
    qwen2_5_coder_14b: Optional[ModelConfig] = None
    qwen2_5_coder_7b_turbo: Optional[ModelConfig] = None


class LocalConfig(BaseModel):
    default: str = "qwen2.5-coder-7b"
    qwen2_5_coder_7b: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            path="models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
        )
    )


class CloudConfig(BaseModel):
    default: str = "gemini-2.5-flash"
    gemini: CloudModelConfig = Field(default_factory=CloudModelConfig)


class ModelsConfig(BaseModel):
    local: LocalConfig = Field(default_factory=LocalConfig)
    cloud: CloudConfig = Field(default_factory=CloudConfig)


class RoutingConfig(BaseModel):
    mode: str = "auto"
    auto_threshold_tokens: int = 2000
    local_timeout: int = 300
    cloud_timeout: int = 60


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080
    api_key: str = "sk-local-dev"
    openai_endpoint: str = "/v1/chat/completions"


class PathsConfig(BaseModel):
    models_dir: str = "./models"
    cache_dir: str = "./cache"
    logs_dir: str = "./logs"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class Config(BaseModel):
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def resolve_model_path(self, model_key: str) -> str:
        models = self.models.local.model_dump(exclude_none=True)
        if model_key in models:
            model = models[model_key]
            if model and isinstance(model, dict):
                path = model.get("path", "")
                if path and not os.path.isabs(path):
                    return os.path.join(self.paths.models_dir, path)
                return path
        return ""


_config: Optional[Config] = None


def get_config(config_path: str = "config.yaml") -> Config:
    global _config
    if _config is None:
        config_file = Path(config_path)
        if config_file.exists():
            _config = Config.from_yaml(str(config_file))
        else:
            _config = Config()
    return _config


def reset_config():
    global _config
    _config = None
