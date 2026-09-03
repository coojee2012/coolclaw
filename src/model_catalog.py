"""Curated catalogs for local GGUF models and cloud API providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LocalModelVariant:
  quant: str
  filename: str
  size_gb: float
  recommended: bool = False


@dataclass(frozen=True)
class LocalModelEntry:
  id: str
  name: str
  family: str
  description: str
  role: str  # coding | general | fast | thinking
  tags: tuple[str, ...]
  min_ram_gb: int
  recommended_vram_gb: int
  huggingface_repo: str
  variants: tuple[LocalModelVariant, ...]
  llama_params: dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> dict:
    return {
      "id": self.id,
      "name": self.name,
      "family": self.family,
      "description": self.description,
      "role": self.role,
      "tags": list(self.tags),
      "min_ram_gb": self.min_ram_gb,
      "recommended_vram_gb": self.recommended_vram_gb,
      "huggingface_repo": self.huggingface_repo,
      "variants": [
        {"quant": v.quant, "filename": v.filename, "size_gb": v.size_gb, "recommended": v.recommended}
        for v in self.variants
      ],
      "llama_params": self.llama_params,
    }


@dataclass(frozen=True)
class CloudModelEntry:
  id: str
  name: str
  description: str
  free_quota: str
  rpm: int = 0
  rpd: int = 0
  recommended: bool = False

  def to_dict(self) -> dict:
    return {
      "id": self.id,
      "name": self.name,
      "description": self.description,
      "free_quota": self.free_quota,
      "rpm": self.rpm,
      "rpd": self.rpd,
      "recommended": self.recommended,
    }


@dataclass(frozen=True)
class CloudProviderEntry:
  id: str
  name: str
  region: str  # intl | cn | both
  description: str
  free_tier: str
  signup_url: str
  docs_url: str
  api_key_hint: str
  default_base_url: str
  provider_type: str  # google_ai | openai | openrouter | deepseek | zhipu | siliconflow | groq
  models: tuple[CloudModelEntry, ...]

  def to_dict(self) -> dict:
    return {
      "id": self.id,
      "name": self.name,
      "region": self.region,
      "description": self.description,
      "free_tier": self.free_tier,
      "signup_url": self.signup_url,
      "docs_url": self.docs_url,
      "api_key_hint": self.api_key_hint,
      "default_base_url": self.default_base_url,
      "provider_type": self.provider_type,
      "models": [m.to_dict() for m in self.models],
    }


_LOW_VRAM_DEFAULTS = {
  "n_ctx": 4096,
  "n_gpu_layers": 20,
  "n_threads": 4,
  "n_batch": 256,
  "flash_attn": True,
  "low_vram": True,
  "cache_type_k": "q4_0",
  "cache_type_v": "q4_0",
}

LOCAL_MODEL_CATALOG: list[LocalModelEntry] = [
  LocalModelEntry(
    id="qwen2.5-coder-7b",
    name="Qwen2.5-Coder-7B-Instruct",
    family="Qwen",
    description="阿里代码专用模型，工具调用与代码生成表现优秀，适合 Agent 编码任务。",
    role="coding",
    tags=("coding", "agent", "recommended"),
    min_ram_gb=8,
    recommended_vram_gb=4,
    huggingface_repo="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
    variants=(
      LocalModelVariant("Q4_K_M", "qwen2.5-coder-7b-instruct-q4_k_m.gguf", 4.7, True),
      LocalModelVariant("Q5_K_M", "qwen2.5-coder-7b-instruct-q5_k_m.gguf", 5.4),
      LocalModelVariant("Q3_K_M", "qwen2.5-coder-7b-instruct-q3_k_m.gguf", 3.8),
    ),
    llama_params=_LOW_VRAM_DEFAULTS,
  ),
  LocalModelEntry(
    id="qwen2.5-3b",
    name="Qwen2.5-3B-Instruct",
    family="Qwen",
    description="轻量通用模型，适合调度器、快速问答和低配机器日常对话。",
    role="fast",
    tags=("fast", "dispatcher", "low-vram"),
    min_ram_gb=4,
    recommended_vram_gb=2,
    huggingface_repo="Qwen/Qwen2.5-3B-Instruct-GGUF",
    variants=(
      LocalModelVariant("Q4_K_M", "qwen2.5-3b-instruct-q4_k_m.gguf", 2.0, True),
      LocalModelVariant("Q5_K_M", "qwen2.5-3b-instruct-q5_k_m.gguf", 2.3),
    ),
    llama_params={**_LOW_VRAM_DEFAULTS, "n_gpu_layers": 35},
  ),
  LocalModelEntry(
    id="qwen2.5-1.5b",
    name="Qwen2.5-1.5B-Instruct",
    family="Qwen",
    description="超轻量模型，极低配设备可用，适合意图识别和简单对话。",
    role="fast",
    tags=("ultra-light", "dispatcher"),
    min_ram_gb=3,
    recommended_vram_gb=0,
    huggingface_repo="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    variants=(
      LocalModelVariant("Q4_K_M", "qwen2.5-1.5b-instruct-q4_k_m.gguf", 1.0, True),
    ),
    llama_params={**_LOW_VRAM_DEFAULTS, "n_gpu_layers": 0, "n_ctx": 2048},
  ),
  LocalModelEntry(
    id="llama-3.2-3b",
    name="Llama 3.2 3B Instruct",
    family="Meta",
    description="Meta 小模型，英文优秀，多语言尚可，社区生态成熟。",
    role="general",
    tags=("general", "meta"),
    min_ram_gb=4,
    recommended_vram_gb=2,
    huggingface_repo="bartowski/Llama-3.2-3B-Instruct-GGUF",
    variants=(
      LocalModelVariant("Q4_K_M", "Llama-3.2-3B-Instruct-Q4_K_M.gguf", 2.0, True),
      LocalModelVariant("Q5_K_M", "Llama-3.2-3B-Instruct-Q5_K_M.gguf", 2.3),
    ),
    llama_params=_LOW_VRAM_DEFAULTS,
  ),
  LocalModelEntry(
    id="phi-3-mini",
    name="Phi-3 Mini 4K Instruct",
    family="Microsoft",
    description="微软小模型，推理能力强，适合逻辑与代码辅助。",
    role="general",
    tags=("reasoning", "microsoft"),
    min_ram_gb=4,
    recommended_vram_gb=2,
    huggingface_repo="microsoft/Phi-3-mini-4k-instruct-gguf",
    variants=(
      LocalModelVariant("Q4_K_M", "Phi-3-mini-4k-instruct-q4.gguf", 2.3, True),
    ),
    llama_params=_LOW_VRAM_DEFAULTS,
  ),
  LocalModelEntry(
    id="gemma-2-2b",
    name="Gemma 2 2B IT",
    family="Google",
    description="Google 轻量指令模型，响应快，适合日常助手场景。",
    role="general",
    tags=("google", "fast"),
    min_ram_gb=4,
    recommended_vram_gb=2,
    huggingface_repo="bartowski/gemma-2-2b-it-GGUF",
    variants=(
      LocalModelVariant("Q4_K_M", "gemma-2-2b-it-Q4_K_M.gguf", 1.6, True),
    ),
    llama_params=_LOW_VRAM_DEFAULTS,
  ),
  LocalModelEntry(
    id="deepseek-coder-6.7b",
    name="DeepSeek Coder 6.7B Instruct",
    family="DeepSeek",
    description="深度求索代码模型，中英文代码能力强。",
    role="coding",
    tags=("coding", "deepseek"),
    min_ram_gb=8,
    recommended_vram_gb=4,
    huggingface_repo="bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
    variants=(
      LocalModelVariant("Q4_K_M", "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf", 4.1, True),
    ),
    llama_params=_LOW_VRAM_DEFAULTS,
  ),
  LocalModelEntry(
    id="qwen2.5-coder-14b",
    name="Qwen2.5-Coder-14B-Instruct",
    family="Qwen",
    description="更大代码模型，质量更高，需要 16GB+ 内存。",
    role="coding",
    tags=("coding", "high-quality"),
    min_ram_gb=16,
    recommended_vram_gb=8,
    huggingface_repo="Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
    variants=(
      LocalModelVariant("Q4_K_M", "qwen2.5-coder-14b-instruct-q4_k_m.gguf", 8.9, True),
      LocalModelVariant("Q3_K_M", "qwen2.5-coder-14b-instruct-q3_k_m.gguf", 7.0),
    ),
    llama_params={**_LOW_VRAM_DEFAULTS, "n_ctx": 8192, "n_gpu_layers": 30},
  ),
]

CLOUD_PROVIDER_CATALOG: list[CloudProviderEntry] = [
  CloudProviderEntry(
    id="google_ai",
    name="Google AI (Gemini)",
    region="intl",
    description="Google AI Studio 官方 API，Gemini Flash 系列免费额度稳定，推荐首选。",
    free_tier="Flash 系列每日免费额度；需科学上网",
    signup_url="https://aistudio.google.com/apikey",
    docs_url="https://ai.google.dev/gemini-api/docs",
    api_key_hint="AIza...",
    default_base_url="https://generativelanguage.googleapis.com/v1beta",
    provider_type="google_ai",
    models=(
      CloudModelEntry("gemini-2.0-flash", "Gemini 2.0 Flash", "速度快、免费额度高", "每日免费", rpm=15, rpd=1500, recommended=True),
      CloudModelEntry("gemini-2.0-flash-lite", "Gemini 2.0 Flash Lite", "更轻量，适合高频调用", "每日免费", rpm=30, rpd=1500, recommended=True),
      CloudModelEntry("gemini-1.5-flash", "Gemini 1.5 Flash", "成熟稳定", "每日免费", rpm=15, rpd=1500),
      CloudModelEntry("gemini-1.5-pro", "Gemini 1.5 Pro", "更强推理，免费额度较少", "有限免费", rpm=2, rpd=50),
    ),
  ),
  CloudProviderEntry(
    id="openrouter",
    name="OpenRouter",
    region="both",
    description="聚合多家模型，有免费模型通道（:free 后缀），适合备用和模型对比。",
    free_tier="多个 :free 模型每日限额",
    signup_url="https://openrouter.ai/keys",
    docs_url="https://openrouter.ai/docs",
    api_key_hint="sk-or-...",
    default_base_url="https://openrouter.ai/api/v1",
    provider_type="openrouter",
    models=(
      CloudModelEntry("google/gemma-3-12b-it:free", "Gemma 3 12B (Free)", "Google 开源模型免费通道", "每日免费", recommended=True),
      CloudModelEntry("meta-llama/llama-3.3-70b-instruct:free", "Llama 3.3 70B (Free)", "大模型免费试用", "每日免费"),
      CloudModelEntry("qwen/qwen-2.5-coder-32b-instruct", "Qwen2.5 Coder 32B", "付费但便宜", "按量计费"),
    ),
  ),
  CloudProviderEntry(
    id="groq",
    name="Groq",
    region="intl",
    description="超快推理，Llama/Mixtral 免费额度，延迟极低。",
    free_tier="每日免费请求配额",
    signup_url="https://console.groq.com/keys",
    docs_url="https://console.groq.com/docs",
    api_key_hint="gsk_...",
    default_base_url="https://api.groq.com/openai/v1",
    provider_type="openai",
    models=(
      CloudModelEntry("llama-3.3-70b-versatile", "Llama 3.3 70B", "高质量通用", "每日免费", rpm=30, recommended=True),
      CloudModelEntry("llama-3.1-8b-instant", "Llama 3.1 8B Instant", "极速响应", "每日免费", rpm=30),
      CloudModelEntry("mixtral-8x7b-32768", "Mixtral 8x7B", "长上下文", "每日免费"),
    ),
  ),
  CloudProviderEntry(
    id="deepseek",
    name="DeepSeek",
    region="cn",
    description="国产高性价比 API，新用户赠额度，代码能力强。",
    free_tier="新用户赠送额度",
    signup_url="https://platform.deepseek.com/",
    docs_url="https://platform.deepseek.com/api-docs/",
    api_key_hint="sk-...",
    default_base_url="https://api.deepseek.com/v1",
    provider_type="openai",
    models=(
      CloudModelEntry("deepseek-chat", "DeepSeek Chat", "通用对话", "赠额度", recommended=True),
      CloudModelEntry("deepseek-coder", "DeepSeek Coder", "代码专用", "赠额度", recommended=True),
    ),
  ),
  CloudProviderEntry(
    id="siliconflow",
    name="硅基流动 SiliconFlow",
    region="cn",
    description="国内聚合平台，多款开源模型免费额度，无需翻墙。",
    free_tier="多款模型每日免费额度",
    signup_url="https://cloud.siliconflow.cn/",
    docs_url="https://docs.siliconflow.cn/",
    api_key_hint="sk-...",
    default_base_url="https://api.siliconflow.cn/v1",
    provider_type="openai",
    models=(
      CloudModelEntry("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5 7B", "免费额度稳定", "每日免费", recommended=True),
      CloudModelEntry("deepseek-ai/DeepSeek-V2.5", "DeepSeek V2.5", "高质量", "每日免费"),
      CloudModelEntry("THUDM/glm-4-9b-chat", "GLM-4 9B", "智谱开源", "每日免费"),
    ),
  ),
  CloudProviderEntry(
    id="zhipu",
    name="智谱 AI (GLM)",
    region="cn",
    description="清华系大模型，GLM-4-Flash 免费额度适合日常。",
    free_tier="GLM-4-Flash 免费",
    signup_url="https://open.bigmodel.cn/",
    docs_url="https://open.bigmodel.cn/dev/api",
    api_key_hint="...",
    default_base_url="https://open.bigmodel.cn/api/paas/v4",
    provider_type="openai",
    models=(
      CloudModelEntry("glm-4-flash", "GLM-4 Flash", "免费快速模型", "每日免费", recommended=True),
      CloudModelEntry("glm-4-air", "GLM-4 Air", "均衡性能", "按量计费"),
      CloudModelEntry("glm-4-plus", "GLM-4 Plus", "旗舰模型", "按量计费"),
    ),
  ),
  CloudProviderEntry(
    id="moonshot",
    name="Moonshot (Kimi)",
    region="cn",
    description="长上下文能力强，新用户赠额度。",
    free_tier="新用户赠额度",
    signup_url="https://platform.moonshot.cn/",
    docs_url="https://platform.moonshot.cn/docs",
    api_key_hint="sk-...",
    default_base_url="https://api.moonshot.cn/v1",
    provider_type="openai",
    models=(
      CloudModelEntry("moonshot-v1-8k", "Moonshot 8K", "标准版", "赠额度"),
      CloudModelEntry("moonshot-v1-32k", "Moonshot 32K", "长上下文", "赠额度", recommended=True),
    ),
  ),
  CloudProviderEntry(
    id="dashscope",
    name="阿里百炼 (DashScope)",
    region="cn",
    description="阿里云模型服务，Qwen 系列，新用户有免费额度。",
    free_tier="新用户免费额度",
    signup_url="https://dashscope.console.aliyun.com/",
    docs_url="https://help.aliyun.com/zh/dashscope/",
    api_key_hint="sk-...",
    default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    provider_type="openai",
    models=(
      CloudModelEntry("qwen-turbo", "Qwen Turbo", "快速便宜", "免费额度", recommended=True),
      CloudModelEntry("qwen-plus", "Qwen Plus", "更强能力", "免费额度"),
      CloudModelEntry("qwen-coder-turbo", "Qwen Coder Turbo", "代码专用", "免费额度"),
    ),
  ),
  CloudProviderEntry(
    id="mistral",
    name="Mistral AI",
    region="intl",
    description="欧洲模型厂商，有免费实验额度。",
    free_tier="实验性免费额度",
    signup_url="https://console.mistral.ai/",
    docs_url="https://docs.mistral.ai/",
    api_key_hint="...",
    default_base_url="https://api.mistral.ai/v1",
    provider_type="openai",
    models=(
      CloudModelEntry("mistral-small-latest", "Mistral Small", "均衡", "有限免费"),
      CloudModelEntry("open-mistral-nemo", "Mistral Nemo", "开源系", "有限免费", recommended=True),
    ),
  ),
]


def get_local_catalog() -> list[dict]:
  return [m.to_dict() for m in LOCAL_MODEL_CATALOG]


def get_cloud_catalog() -> list[dict]:
  return [p.to_dict() for p in CLOUD_PROVIDER_CATALOG]


def find_local_model(model_id: str) -> LocalModelEntry | None:
  for m in LOCAL_MODEL_CATALOG:
    if m.id == model_id:
      return m
  return None


def find_cloud_provider(provider_id: str) -> CloudProviderEntry | None:
  for p in CLOUD_PROVIDER_CATALOG:
    if p.id == provider_id:
      return p
  return None
