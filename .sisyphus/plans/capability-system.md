# 能力库系统设计计划

## 概述

为 OpenCode Helper 调度中心设计一个模块化的能力系统，让用户可以通过自然语言或可视化界面组合各种能力来创建自动化任务。

---

## 一、能力清单 (Capability Inventory)

### 📊 能力分类总览

| 类别 | 能力数量 | 内存占用 | 实现难度 |
|------|----------|----------|----------|
| 网络类 | 4 | ~50MB | 低 |
| 文件类 | 4 | ~30MB | 极低 |
| 通知类 | 3 | ~25MB | 低 |
| 定时类 | 2 | ~15MB | 低 |
| 系统类 | 3 | ~10MB | 极低 |
| AI处理类 | 5 | (使用已有模型) | 中 |
| **合计** | **21** | **~130MB** | - |

---

## 二、能力详细清单

### 🌐 A. 网络类能力

| 能力 | 功能 | 推荐库 | 内存 | 状态 |
|------|------|--------|------|------|
| `web_search` | DuckDuckGo 搜索，返回标题/摘要/链接 | `duckduckgo-search-api` | ~30MB | ⭐ 优先 |
| `web_fetch` | 获取网页内容，支持静态页面 | `requests` + `BeautifulSoup` | ~50MB | ⭐ 优先 |
| `api_client` | 通用 HTTP 客户端，支持 GET/POST | `httpx` | ~15MB | ⭐ 优先 |
| `webhook_trigger` | 发送 webhook 通知 | `httpx` | ~10MB | 中 |

### 📁 B. 文件类能力

| 能力 | 功能 | 推荐库 | 内存 | 状态 |
|------|------|--------|------|------|
| `file_read` | 读取文本/JSON/CSV文件 | 内置 | ~5MB | ✅ 已有 |
| `file_write` | 写入文本/JSON/CSV/Markdown | 内置 | ~5MB | ✅ 已有 |
| `file_watch` | 监控目录变化 | `watchdog` | ~20MB | 中 |
| `file_download` | 下载网络文件到本地 | `requests` | ~10MB | ⭐ 优先 |

### 🔔 C. 通知类能力

| 能力 | 功能 | 推荐库 | 内存 | 状态 |
|------|------|--------|------|------|
| `notify_telegram` | 发送 Telegram 消息 | `pingram` | ~5MB | 中 |
| `notify_email` | 发送邮件 | `apprise` | ~20MB | 中 |
| `notify_webhook` | 发送通用 webhook | `httpx` | ~5MB | ⭐ 优先 |

### ⏰ D. 定时类能力

| 能力 | 功能 | 推荐库 | 内存 | 状态 |
|------|------|--------|------|------|
| `schedule_cron` | Cron 表达式定时任务 | `APScheduler` | ~10MB | ⭐ 优先 |
| `schedule_interval` | 间隔执行任务 | `APScheduler` | ~10MB | ⭐ 优先 |

### 💻 E. 系统类能力

| 能力 | 功能 | 推荐库 | 内存 | 状态 |
|------|------|--------|------|------|
| `clipboard_copy` | 复制内容到剪贴板 | `pyperclip` | ~5MB | ⭐ 优先 |
| `clipboard_paste` | 从剪贴板读取 | `pyperclip` | ~5MB | ⭐ 优先 |
| `shell_execute` | 执行 Shell 命令 | 内置 `subprocess` | ~5MB | 中 |

### 🤖 F. AI处理类能力（复用已有模型）

| 能力 | 功能 | 使用模型 | 内存 | 状态 |
|------|------|----------|------|------|
| `ai_code` | 代码生成/审查 | Qwen2.5-Coder-7B | ~6GB | ✅ 已有 |
| `ai_ocr` | 图片/扫描件文字识别 | LightOnOCR-2-1B | ~2GB | ✅ 已有 |
| `ai_vision` | 图片理解分析 | SmolVLM-256M | ~1GB | ✅ 已有 |
| `ai_document` | 文档结构化 | Granite-Docling | ~1GB | ✅ 已有 |
| `ai_summarize` | 文本摘要生成 | Qwen2.5-3B | ~2GB | ✅ 已有 |

---

## 三、系统架构设计

### 3.1 能力注册机制

```python
# src/capabilities/__init__.py
from typing import Callable, Any
from dataclasses import dataclass

@dataclass
class Capability:
    name: str                           # 唯一标识: "web_search"
    description: str                    # 能力描述
    category: str                       # 分类: "network"
    input_schema: dict                  # 输入参数定义
    output_schema: dict                # 输出结果定义
    memory_usage: int                  # 内存占用(MB)
    async_func: Callable               # 异步执行函数
    examples: list[str]                 # 使用示例

class CapabilityRegistry:
    _capabilities: dict[str, Capability] = {}
    
    @classmethod
    def register(cls, name: str, **kwargs):
        """装饰器注册能力"""
        def decorator(func: Callable):
            cls._capabilities[name] = Capability(
                name=name,
                async_func=func,
                **kwargs
            )
            return func
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Capability:
        return cls._capabilities.get(name)
    
    @classmethod
    def list_by_category(cls, category: str) -> list[Capability]:
        return [c for c in cls._capabilities.values() 
                if c.category == category]
```

### 3.2 能力执行流程

```
用户输入: "帮我每天早上8点搜索科技新闻，生成摘要发到Telegram"
                ↓
┌─────────────────────────────────────────────────────┐
│              任务解析器 (Intent Parser)              │
│  提取: 关键词=科技新闻, 时间=8:00, 动作=搜索+摘要+发送 │
└─────────────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────┐
│           能力规划器 (Capability Planner)           │
│  能力链: web_search → ai_summarize → notify_telegram │
│  执行顺序: 串行 (节省内存)                            │
└─────────────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────┐
│             能力执行器 (Capability Executor)          │
│  1. web_search("科技新闻") → results[]               │
│  2. ai_summarize(results) → summary               │
│  3. notify_telegram(summary) → sent ✓               │
└─────────────────────────────────────────────────────┘
                ↓
              结果输出
```

### 3.3 任务配置格式

```python
# 用户可以用自然语言创建任务
user_input = "每天早上8点搜索科技新闻，生成摘要发到Telegram"

# 系统解析后生成结构化配置
task_config = {
    "name": "每日科技新闻摘要",
    "trigger": {
        "type": "cron",
        "expression": "0 8 * * *",  # 每天8点
    },
    "steps": [
        {
            "capability": "web_search",
            "params": {"query": "科技新闻"},
        },
        {
            "capability": "ai_summarize",
            "params": {"length": "short"},
        },
        {
            "capability": "notify_telegram",
            "params": {},
        },
    ],
    "output": {
        "format": "telegram_message",
    }
}
```

---

## 四、实际任务示例

### 任务1: 每日新闻摘要
```
用户: "帮我每天早上8点搜索国际军事动态，整理成今日头条风格的文案，复制到剪贴板"
能力链: web_search → ai_summarize → clipboard_copy
```

### 任务2: 价格监控
```
用户: "监控某宝某商品价格，低于100元时发Telegram通知"
能力链: web_fetch → ai_parse → notify_telegram
```

### 任务3: 文档自动处理
```
用户: "当下载文件夹有新PDF时，自动OCR提取文字并保存到文档目录"
能力链: file_watch → ai_ocr → file_write
```

### 任务4: 内容创作助手
```
用户: "帮我收集本周AI领域的5篇重要文章，生成阅读清单"
能力链: web_search × 5 → ai_summarize × 5 → clipboard_copy
```

### 任务5: 自动备份报告
```
用户: "每周日晚上把项目文件夹打包，上传到指定目录"
能力链: shell_execute → file_write
```

### 任务6: 热点追踪
```
用户: "追踪某个话题的微博热搜，每小时报告一次变化"
能力链: web_fetch → ai_parse → notify_telegram
```

### 任务7: 图片内容分析
```
用户: "当截图文件夹有新图片时，自动描述图片内容"
能力链: file_watch → ai_vision → clipboard_copy
```

### 任务8: 邮件摘要
```
用户: "每天早上9点读取邮箱未读邮件，生成摘要"
能力链: api_client → ai_summarize → notify_telegram
```

### 任务9: 代码审查提醒
```
用户: "检测到代码仓库有新提交时，自动审查代码"
能力链: webhooks → ai_code → notify_webhook
```

### 任务10: 会议纪要生成
```
用户: "把会议录音转成文字，提取关键决策和待办事项"
能力链: clipboard_paste → ai_summarize → clipboard_copy
```

---

## 五、实现优先级

### Phase 1: 基础能力 (1-2周)
| 能力 | 依赖 | 优先级 |
|------|------|--------|
| `web_search` | duckduckgo-search-api | P0 |
| `clipboard_copy/paste` | pyperclip | P0 |
| `schedule_cron` | APScheduler | P0 |
| `notify_webhook` | httpx | P1 |

### Phase 2: 文件和通知 (2-3周)
| 能力 | 依赖 | 优先级 |
|------|------|--------|
| `file_watch` | watchdog | P1 |
| `file_write` | 内置 | P1 |
| `notify_telegram` | pingram | P2 |
| `web_fetch` | requests + bs4 | P2 |

### Phase 3: 高级集成 (3-4周)
| 能力 | 依赖 | 优先级 |
|------|------|--------|
| `file_download` | requests | P2 |
| `schedule_interval` | APScheduler | P2 |
| 任务持久化存储 | SQLite/JSON | P2 |
| 任务历史记录 | 内置 | P2 |

### Phase 4: 智能增强 (4-6周)
| 能力 | 依赖 | 优先级 |
|------|------|--------|
| 自然语言任务解析 | Qwen2.5-Coder | P2 |
| 任务模板市场 | Web UI | P3 |
| 执行历史可视化 | Web UI | P3 |

---

## 六、技术依赖

```txt
# requirements.txt 扩展
duckduckgo-search-api>=0.1.5
requests>=2.28.0
beautifulsoup4>=4.9.0
httpx>=0.24.0
APScheduler>=3.10.0
pyperclip>=1.8.0
watchdog>=3.0.0
pingram>=0.3.0
apprise>=1.6.0
```

---

## 七、内存管理策略

```
总能力内存预算: ~200MB (远低于你的16GB)

策略:
1. 按需加载: 能力在执行时才加载
2. 顺序执行: 任务步骤串行执行，不并行
3. 释放机制: 能力执行完后释放内存
4. 模型复用: AI能力共用已有模型实例
```

---

## 八、扩展计划

未来可扩展方向:
- 🌐 云存储集成 (iCloud, Dropbox)
- 📊 数据可视化 (图表生成)
- 🎨 图片生成 (Stable Diffusion)
- 🌏 多语言翻译 (Google Translate API)
- 📅 日历集成 (Google Calendar API)
