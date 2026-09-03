# OpenCode 对接本地 AI Helper 调度中心

本文档介绍如何将 OpenCode AI 助手配置为使用本项目的 AI 调度中心。

## 功能概述

OpenCode Helper 提供以下能力：

- **智能调度**：自动选择合适的模型（本地 Ollama/llama.cpp 或云端 Gemini）
- **多会话**：支持多会话管理与历史记录
- **任务自动化**：支持定时任务执行
- **通知**：支持钉钉、飞书、邮件通知
- **知识库**：RAG 向量检索

## API 端点

项目提供 OpenAI 兼容接口：

| 端点 | 说明 |
|------|------|
| `POST /v1/chat/completions` | 聊天补全 |
| `POST /v1/completions` | 文本补全 |
| `GET /v1/models` | 列出可用模型 |

## 在 OpenCode 中配置

### 方式一：配置文件

编辑 OpenCode 配置文件 `~/.config/opencode/opencode.jsonc`：

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "local-ai-helper": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "本地 AI 调度中心",
      "options": {
        "baseURL": "http://127.0.0.1:8484/v1",
        "apiKey": "sk-local-dev"
      },
      "models": {
        "qwen2.5-coder-7b": {
          "name": "Qwen2.5 Coder 7B",
          "limit": {
            "context": 4096,
            "output": 2048
          }
        },
        "qwen2.5-3b": {
          "name": "Qwen2.5 3B 快速模式",
          "limit": {
            "context": 4096,
            "output": 1024
          }
        },
        "gemma-4-31b-it": {
          "name": "Gemma 4 31B 云端",
          "limit": {
            "context": 256000,
            "output": 8192
          }
        }
      }
    }
  }
}
```

### 方式二：使用 /connect 命令

```bash
$ /connect

┌ 添加凭证
│
◇ 输入提供商 ID
│ local-ai-helper
└

$ /connect local-ai-helper

┌ 添加凭证
│
◇ 输入 API Key
│ sk-local-dev
└

# Base URL 会自动使用 http://127.0.0.1:8484/v1
```

### 凭证配置

在 `~/.local/share/opencode/auth.json` 中添加：

```json
{
  "local-ai-helper": {
    "type": "api",
    "key": "sk-local-dev"
  }
}
```

## 使用示例

### 选择模型

```bash
$ /model local-ai-helper/qwen2.5-coder-7b
已切换到 qwen2.5-coder-7b
```

### 查看模型列表

```bash
$ /models
```

应该能看到配置的模型以及其他可用模型。

## 可用模型

项目支持的模型（在 config.yaml 中配置）：

| 模型 | 说明 | 场景 |
|------|------|------|
| `qwen2.5-coder-7b` | Qwen2.5 编码专家 7B | 代码生成、优化 |
| `qwen2.5-3b` | Qwen2.5 3B | 快速响应、轻量任务 |
| `gemma-4-31b-it` | Gemma 4 31B 云端 | 复杂推理、长文本 |

## 本地模型配置

如果需要使用本地 Ollama 模型：

```json
{
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama 本地",
      "options": {
        "baseURL": "http://localhost:11434/v1"
      },
      "models": {
        "qwen2.5:7b": {},
        "codellama:7b": {}
      }
    }
  }
}
```

## 高级配置

### 自定义超时

```json
{
  "provider": {
    "local-ai-helper": {
      "options": {
        "baseURL": "http://127.0.0.1:8484/v1",
        "timeout": 300000
      }
    }
  }
}
```

### 添加请求头

```json
{
  "provider": {
    "local-ai-helper": {
      "options": {
        "baseURL": "http://127.0.0.1:8484/v1",
        "headers": {
          "X-Custom-Header": "value"
        }
      }
    }
  }
}
```

## 故障排除

### 连接被拒绝

确保服务已启动：

```bash
cd /path/to/opencode_helper
python -m src.cli serve
# 或
python -m uvicorn src.api:create_app --reload --port 8484
```

### 401 认证错误

检查 API Key 是否正确，默认值为 `sk-local-dev`（在 config.yaml 中配置）。

### 模型未找到

确保模型名称与提供商支持的名称完全匹配。

## 环境变量

项目支持以下环境变量：

| 变量 | 说明 |
|------|------|
| `GOOGLE_AI_API_KEY` | Google AI API Key (用于 Gemini) |
| `GEMINI_API_KEY` | Gemini API Key (别名) |
| `OPENCODE_CONFIG` | 自定义配置文件路径 |

## 更多信息

- 项目地址：[GitHub 仓库]
- 文档：[docs/](.)
- 配置：[config.yaml](../config.yaml)