import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class ModelConfig(BaseModel):
    path: str
    role: str = "general"
    n_ctx: int = 131072
    n_gpu_layers: int = -1
    n_threads: int = 6
    n_batch: int = 512
    low_vram: bool = False
    flash_attn: bool = True
    cache_type_k: str = "q4_0"
    cache_type_v: str = "q4_0"
    use_mmap: bool = True


class CloudModelConfig(BaseModel):
    api_key: str = Field(default="")
    model: str = "gemini-3.5-flash-lite"
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
    model_config = ConfigDict(extra="allow")

    default: str = "qwen2.5-coder-7b"
    qwen2_5_coder_7b: Optional[ModelConfig] = Field(
        default=None,
    )


class CloudConfig(BaseModel):
    default: str = "gemma-4-31b-it"
    google_ai: CloudModelConfig = Field(default_factory=CloudModelConfig)


class ModelsConfig(BaseModel):
    local: LocalConfig = Field(default_factory=LocalConfig)
    cloud: CloudConfig = Field(default_factory=CloudConfig)


class RoutingConfig(BaseModel):
    mode: str = "auto"
    auto_threshold_tokens: int = 2000
    local_timeout: int = 300
    cloud_timeout: int = 60
    expert_bindings: dict[str, str] = Field(default_factory=dict)


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


class NetworkConfig(BaseModel):
    http_proxy: str = ""
    https_proxy: str = ""


class SMTPConfig(BaseModel):
    host: str = "smtp.gmail.com"
    port: int = 587
    user: str = ""
    password: str = ""
    from_email: str = ""
    use_tls: bool = True


class NotificationConfig(BaseModel):
    default_webhook_url: str = ""
    dingtalk_webhook: str = ""
    dingtalk_secret: str = ""
    feishu_webhook: str = ""
    smtp: SMTPConfig = Field(default_factory=SMTPConfig)


class RateLimitConfig(BaseModel):
    rpm: int = 10
    burst: int = 5
    queue_size: int = 50
    timeout: int = 30
    rpd: int = 0
    cooldown_seconds: float = 60.0


class ProviderEndpointConfig(BaseModel):
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    timeout: int = 180
    roles: list[str] = Field(default_factory=list)
    multimodal: bool = False


class ProxyConfig(BaseModel):
    enabled: bool = False
    default_provider: str = "google_ai"
    fallback_order: list[str] = Field(default_factory=list)
    rate_limits: dict[str, RateLimitConfig] = Field(default_factory=dict)
    providers: dict[str, ProviderEndpointConfig] = Field(default_factory=dict)
    model_routes: dict[str, str] = Field(default_factory=dict)


class McpConfig(BaseModel):
    """MCP Agent 配置"""
    enabled: bool = False
    max_rounds: int = 10
    agent_model: str = ""  # 留空则使用 default_provider


class Config(BaseModel):
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    notification: NotificationConfig = Field(default_factory=NotificationConfig)
    smtp: SMTPConfig = Field(default_factory=SMTPConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    mcp: McpConfig = Field(default_factory=McpConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def resolve_model_path(self, model_key: str) -> str:
        models = self.models.local.model_dump(exclude_none=True)

        normalized_key = model_key.replace("-", "_").replace(".", "_")
        key_variants = [normalized_key, model_key]

        for key in key_variants:
            if key in models:
                model = models[key]
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


def get_httpx_proxy(config: Config | None = None) -> dict[str, str] | None:
    """Return httpx proxies dict from config or env vars."""
    import os
    cfg = config or get_config()
    http_p = cfg.network.http_proxy or os.environ.get("HTTP_PROXY", "")
    https_p = cfg.network.https_proxy or os.environ.get("HTTPS_PROXY", "")
    if https_p:
        return {"https://": https_p, "http://": http_p or https_p}
    if http_p:
        return {"http://": http_p, "https://": http_p}
    return None
