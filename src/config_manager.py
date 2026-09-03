"""Read/write config.yaml for admin UI with secret masking."""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config.yaml")
SENSITIVE_SUFFIXES = ("_key", "_password", "_secret", "_token")
SENSITIVE_EXACT = {"api_key", "password", "from_email"}


def _is_sensitive(key: str) -> bool:
  k = key.lower()
  if k in SENSITIVE_EXACT:
    return True
  return any(k.endswith(s) for s in SENSITIVE_SUFFIXES)


def _mask_value(key: str, value: Any) -> Any:
  if isinstance(value, dict):
    return {k: _mask_value(k, v) for k, v in value.items()}
  if isinstance(value, list):
    return [_mask_value(key, v) for v in value]
  if _is_sensitive(key) and isinstance(value, str) and value:
    return value[:4] + "****" + value[-2:] if len(value) > 8 else "****"
  return value


def _deep_merge(base: dict, updates: dict) -> dict:
  result = copy.deepcopy(base)
  for key, value in updates.items():
    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
      result[key] = _deep_merge(result[key], value)
    else:
      result[key] = value
  return result


def _preserve_secrets(original: dict, updates: dict) -> dict:
  """Keep original secrets when UI sends masked placeholders."""
  result = copy.deepcopy(updates)
  for key, value in updates.items():
    if isinstance(value, dict) and key in original and isinstance(original[key], dict):
      result[key] = _preserve_secrets(original[key], value)
    elif isinstance(value, str) and "****" in value and key in original:
      result[key] = original[key]
  return result


def load_yaml_raw(path: Path | None = None) -> dict:
  cfg_path = path or DEFAULT_CONFIG_PATH
  if not cfg_path.exists():
    return {}
  with open(cfg_path) as f:
    data = yaml.safe_load(f) or {}
  return data


def save_yaml_raw(data: dict, path: Path | None = None) -> None:
  cfg_path = path or DEFAULT_CONFIG_PATH
  with open(cfg_path, "w") as f:
    yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def get_admin_config(path: Path | None = None) -> dict:
  raw = load_yaml_raw(path)
  masked = _mask_value("", raw)
  return {
    "config": masked,
    "config_path": str(path or DEFAULT_CONFIG_PATH.resolve()),
    "sections": _config_schema(),
  }


def update_admin_config(updates: dict, path: Path | None = None) -> dict:
  from .config import reset_config

  cfg_path = path or DEFAULT_CONFIG_PATH
  original = load_yaml_raw(cfg_path)
  cleaned = _preserve_secrets(original, updates)
  merged = _deep_merge(original, cleaned)
  save_yaml_raw(merged, cfg_path)
  reset_config()
  try:
    from .model_pool import reload_pool
    reload_pool()
  except Exception:
    pass
  logger.info("Config updated via admin UI: %s", list(updates.keys()))
  return get_admin_config(cfg_path)


def _config_schema() -> list[dict]:
  return [
    {
      "id": "paths",
      "icon": "📁",
      "title": "存储路径",
      "description": "指定模型文件、缓存和日志的存放位置。首次使用请先设置「模型存放目录」。",
      "fields": [
        {"key": "paths.models_dir", "label": "模型存放目录", "type": "path", "required": True,
         "placeholder": "/Volumes/LynnData/myclaw_models",
         "hint": "本机 .gguf 模型文件目录，下载模型前必填"},
        {"key": "paths.cache_dir", "label": "缓存目录", "type": "path", "placeholder": "./cache"},
        {"key": "paths.logs_dir", "label": "日志目录", "type": "path", "placeholder": "./logs"},
      ],
    },
    {
      "id": "routing",
      "icon": "🔀",
      "title": "模型路由",
      "description": "决定优先使用本地模型还是云端 API。大多数用户选择「智能自动」即可。",
      "fields": [
        {"key": "routing.mode", "label": "路由模式", "type": "select", "simple": True,
         "options": [
           {"value": "auto", "label": "智能自动（推荐）",
            "hint": "OMO 能力路由 + 云端免费优先，本地兜底"},
           {"value": "cloud_first", "label": "云端优先", "hint": "优先使用云端 API"},
           {"value": "local_only", "label": "仅本地", "hint": "完全离线，不调用云端"},
           {"value": "cloud_only", "label": "仅云端", "hint": "不使用本地 GGUF"},
         ]},
        {"key": "routing.auto_threshold_tokens", "label": "自动切换阈值 (tokens)", "type": "number", "simple": False},
        {"key": "routing.local_timeout", "label": "本地模型超时 (秒)", "type": "number", "simple": False},
        {"key": "routing.cloud_timeout", "label": "云端 API 超时 (秒)", "type": "number", "simple": False},
        {"key": "routing.expert_bindings", "label": "专家 Provider 绑定", "type": "json", "simple": False,
         "hint": "请在「模型配置 → 专家路由」中可视化设置"},
      ],
    },
    {
      "id": "network",
      "icon": "🌐",
      "title": "网络代理",
      "description": "访问 Google / OpenRouter 等国际 API 时，如无法直连请配置代理。",
      "fields": [
        {"key": "network.http_proxy", "label": "HTTP 代理", "type": "text",
         "placeholder": "http://127.0.0.1:20171"},
        {"key": "network.https_proxy", "label": "HTTPS 代理", "type": "text",
         "placeholder": "http://127.0.0.1:20171"},
      ],
    },
    {
      "id": "mcp",
      "icon": "🤖",
      "title": "Agent 工具",
      "description": "启用后 AI 可读写项目文件、运行命令、联网搜索。对话时需设置工作目录。",
      "fields": [
        {"key": "mcp.enabled", "label": "启用 MCP 工具循环", "type": "boolean", "simple": True},
        {"key": "mcp.agent_model", "label": "Agent 使用的模型", "type": "select", "simple": True,
         "dynamic_options": "mcp_providers",
         "options": [
           {"value": "", "label": "自动（跟随专家路由，云端优先）"},
           {"value": "local", "label": "本地 Coder 7B"},
         ],
         "hint": "一般保持「自动」即可"},
        {"key": "mcp.max_rounds", "label": "最大工具调用轮数", "type": "number", "simple": False},
      ],
    },
    {
      "id": "logging",
      "icon": "📋",
      "title": "日志",
      "description": "调整服务端日志详细程度，排查问题时可用 DEBUG。",
      "fields": [
        {"key": "logging.level", "label": "日志级别", "type": "select",
         "options": [
           {"value": "INFO", "label": "INFO（默认）"},
           {"value": "DEBUG", "label": "DEBUG（详细）"},
           {"value": "WARNING", "label": "WARNING"},
           {"value": "ERROR", "label": "ERROR"},
         ]},
      ],
    },
    {
      "id": "proxy",
      "icon": "⚡",
      "title": "云端代理层",
      "description": "多 Provider 自动回退、速率限制与冷却（高级选项）。",
      "advanced": True,
      "fields": [
        {"key": "proxy.enabled", "label": "启用代理层", "type": "boolean"},
        {"key": "proxy.default_provider", "label": "默认 Provider", "type": "text"},
      ],
    },
  ]


def get_nested(data: dict, dotted_key: str) -> Any:
  parts = dotted_key.split(".")
  cur = data
  for p in parts:
    if not isinstance(cur, dict) or p not in cur:
      return None
    cur = cur[p]
  return cur


def set_nested(data: dict, dotted_key: str, value: Any) -> None:
  parts = dotted_key.split(".")
  cur = data
  for p in parts[:-1]:
    cur = cur.setdefault(p, {})
  cur[parts[-1]] = value


def scan_installed_models(models_dir: str, include_config: bool = True) -> list[dict]:
  """Scan models_dir for GGUF files and merge config.yaml local model paths."""
  results: list[dict] = []
  seen_paths: set[str] = set()

  def _add(path: Path, source: str, in_config: bool = False, config_key: str = "", role: str = ""):
    p = str(path.expanduser().resolve())
    if p in seen_paths or not path.is_file():
      return
    seen_paths.add(p)
    try:
      stat = path.stat()
      results.append({
        "filename": path.name,
        "path": p,
        "relative_path": path.name,
        "size_gb": round(stat.st_size / (1024 ** 3), 2),
        "mtime": stat.st_mtime,
        "source": source,
        "in_config": in_config,
        "config_key": config_key,
        "role": role,
      })
    except Exception:
      pass

  root = Path(models_dir).expanduser().resolve() if models_dir else None
  if root and root.is_dir():
    for f in sorted(root.iterdir()):
      if f.is_file() and f.suffix.lower() == ".gguf":
        _add(f, "disk")
    # Also recurse one level for subdirs
    for sub in sorted(root.iterdir()):
      if sub.is_dir() and sub.name not in (".cache", ".git"):
        for f in sorted(sub.rglob("*.gguf")):
          rel = str(f.relative_to(root))
          p = str(f.resolve())
          if p in seen_paths:
            continue
          seen_paths.add(p)
          try:
            stat = f.stat()
            results.append({
              "filename": f.name,
              "path": p,
              "relative_path": rel,
              "size_gb": round(stat.st_size / (1024 ** 3), 2),
              "mtime": stat.st_mtime,
              "source": "disk",
              "in_config": False,
              "config_key": "",
              "role": "",
            })
          except Exception:
            pass

  if include_config:
    raw = load_yaml_raw()
    local = raw.get("models", {}).get("local", {})
    default_name = local.get("default", "")
    for key, model in local.items():
      if key == "default" or not isinstance(model, dict):
        continue
      path_str = model.get("path", "")
      if not path_str:
        continue
      p = Path(path_str).expanduser().resolve()
      # Update existing entry or add
      existing = next((r for r in results if r["path"] == str(p)), None)
      if existing:
        existing["in_config"] = True
        existing["config_key"] = key
        existing["role"] = model.get("role", "")
        existing["is_default"] = (default_name.replace("-", "_") == key or default_name == key.replace("_", "-"))
      else:
        _add(p, "config", in_config=True, config_key=key, role=model.get("role", ""))
        if results and results[-1]["path"] == str(p):
          results[-1]["is_default"] = (default_name.replace("-", "_") == key)

  # Mark default for disk-only matches
  default_local = load_yaml_raw().get("models", {}).get("local", {}).get("default", "")
  for r in results:
    if r.get("is_default"):
      continue
    fname = r["filename"].lower()
    if default_local and default_local.lower().replace("-", "") in fname.replace("-", "").replace("_", ""):
      r["is_default"] = True

  results.sort(key=lambda x: (not x.get("in_config", False), not x.get("is_default", False), x["filename"]))
  return results


def register_local_model_in_config(
  model_id: str,
  file_path: str,
  set_default: bool = False,
  path: Path | None = None,
) -> dict:
  from .model_catalog import find_local_model
  from .config import reset_config

  entry = find_local_model(model_id)
  if not entry:
    raise ValueError(f"Unknown catalog model: {model_id}")

  cfg_path = path or DEFAULT_CONFIG_PATH
  raw = load_yaml_raw(cfg_path)
  key = model_id.replace("-", "_").replace(".", "_")

  model_block = {"path": file_path, "role": entry.role, **entry.llama_params}
  raw.setdefault("models", {}).setdefault("local", {})[key] = model_block
  if set_default:
    raw["models"]["local"]["default"] = model_id

  save_yaml_raw(raw, cfg_path)
  reset_config()
  try:
    from .model_pool import reload_pool
    reload_pool()
  except Exception:
    pass
  return {"config_key": key, "path": file_path, "default": set_default}


def register_local_model_by_path(
  file_path: str,
  role: str = "general",
  set_default: bool = False,
  config_key: str = "",
  path: Path | None = None,
) -> dict:
  """Register any local GGUF file into config.yaml without catalog entry."""
  from .config import reset_config

  p = Path(file_path).expanduser().resolve()
  if not p.is_file():
    raise ValueError(f"File not found: {file_path}")

  key = config_key or p.stem.lower().replace("-", "_").replace(".", "_")[:40]
  cfg_path = path or DEFAULT_CONFIG_PATH
  raw = load_yaml_raw(cfg_path)
  raw.setdefault("models", {}).setdefault("local", {})[key] = {
    "path": str(p),
    "role": role,
    **_LOW_VRAM_DEFAULTS,
  }
  if set_default:
    raw["models"]["local"]["default"] = key.replace("_", "-")

  save_yaml_raw(raw, cfg_path)
  reset_config()
  try:
    from .model_pool import reload_pool
    reload_pool()
  except Exception:
    pass
  return {"config_key": key, "path": str(p), "default": set_default}


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


def _config_path(path: Path | None = None) -> Path:
  return path or DEFAULT_CONFIG_PATH


def read_config_yaml_text(path: Path | None = None) -> dict:
  """Return raw config.yaml text (preserves comments and formatting)."""
  cfg_path = _config_path(path)
  if not cfg_path.exists():
    return {
      "content": "",
      "path": str(cfg_path.resolve()),
      "size": 0,
      "exists": False,
    }
  text = cfg_path.read_text(encoding="utf-8")
  stat = cfg_path.stat()
  return {
    "content": text,
    "path": str(cfg_path.resolve()),
    "size": len(text),
    "exists": True,
    "modified": stat.st_mtime,
    "backup_path": str(cfg_path.with_suffix(".yaml.bak").resolve()),
  }


def validate_yaml_text(content: str) -> dict:
  """Parse YAML and return validation metadata."""
  if not content.strip():
    return {"valid": False, "error": "配置文件为空"}
  try:
    data = yaml.safe_load(content)
  except yaml.YAMLError as e:
    return {"valid": False, "error": str(e)}
  if data is not None and not isinstance(data, dict):
    return {"valid": False, "error": "根节点必须是 YAML 映射 (key: value)"}
  top_keys = list((data or {}).keys())
  return {
    "valid": True,
    "top_level_keys": top_keys,
    "line_count": content.count("\n") + (1 if content and not content.endswith("\n") else 0),
  }


def normalize_yaml_text(content: str) -> str:
  """Normalize line endings; ensure single trailing newline."""
  text = content.replace("\r\n", "\n").replace("\r", "\n")
  if text and not text.endswith("\n"):
    text += "\n"
  return text


def format_yaml_text(content: str) -> str:
  """Re-format YAML with ruamel (preserves comments when possible)."""
  try:
    from ruamel.yaml import YAML
  except ImportError as e:
    raise RuntimeError("ruamel.yaml 未安装，无法格式化") from e

  y = YAML()
  y.preserve_quotes = True
  y.indent(mapping=2, sequence=4, offset=2)
  y.width = 120
  y.default_flow_style = False

  from io import StringIO
  data = y.load(content)
  stream = StringIO()
  y.dump(data, stream)
  return normalize_yaml_text(stream.getvalue())


def save_config_yaml_text(
  content: str,
  path: Path | None = None,
  *,
  backup: bool = True,
  apply_runtime: bool = True,
) -> dict:
  """Validate and write raw YAML to disk."""
  from .config import reset_config

  validation = validate_yaml_text(content)
  if not validation["valid"]:
    raise ValueError(validation["error"])

  cfg_path = _config_path(path)
  normalized = normalize_yaml_text(content)

  backup_written = None
  if backup and cfg_path.exists():
    backup_path = cfg_path.with_suffix(".yaml.bak")
    backup_path.write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")
    backup_written = str(backup_path.resolve())

  cfg_path.write_text(normalized, encoding="utf-8")

  if apply_runtime:
    reset_config()
    try:
      from .model_pool import reload_pool
      reload_pool()
    except Exception:
      pass

  logger.info("config.yaml saved via raw editor (%d bytes)", len(normalized))
  return {
    "success": True,
    "path": str(cfg_path.resolve()),
    "backup_path": backup_written,
    "validation": validation,
    "size": len(normalized),
  }

