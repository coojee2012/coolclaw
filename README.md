# CoolClaw — 本地 AI Agent 平台

> 自主编程 · 智能调度 · 31 个 MCP 工具 · Skill 自动化 · 本地推理 + 云端回退

## 定位

CoolClaw 是一个**本地 AI Agent 平台**，具备完整的代码开发能力和自动化 Skill 系统：读写文件、代码搜索、类型诊断、Git 操作、网络搜索、测试执行，以及可扩展的 Skill 插件（天气查询、Word 文档生成等），全部通过 MCP (Model Context Protocol) 协议接入，由本地 7B 模型驱动 + Gemini 云端回退。

```
用户："帮我给 Dispatcher 添加重试机制"
           |
           v
+------------------------------------------------------+
|                  Agent (LLM 循环)                     |
|                                                      |
|  1. _plan_task  -> 拆解为编号步骤                      |
|  2. codegraph_search("Dispatcher")                   |
|  3. read_file("src/dispatcher.py")                   |
|  4. lsp_diagnostics("src/dispatcher.py")             |
|  5. edit_file(添加重试逻辑)                            |
|  6. [自动] lsp_diagnostics(验证无报错)                 |
|  7. run_test(运行测试)                                |
|  8. git_status -> git_commit                         |
+------------------------------------------------------+
           |
           v
  31 个 MCP 工具（文件/代码/LSP/Git/网络/测试/Skill）
```

## 系统架构

```
+----------------------------------------------------------+
|              Web UI (HTMX + Alpine.js + TailwindCSS)      |
|   index.html | login.html | settings.html | tasks/logs    |
|             http://localhost:8484                          |
+-------------------+-------------------+------------------+
                    |                   |
              Cookie Auth (coolclaw_session)
                    |                   |
+-------------------v-------------------v------------------+
|           AuthMiddleware (src/auth.py)                     |
|   Cookie 鉴权 · 不透明 Token · PBKDF2 密码哈希              |
|   未登录 → 重定向 /login.html · API 未认证 → 401            |
+-------------------+-------------------+------------------+
                    |                   |
             /api/chat            /api/chat/stream
             (同步)               (SSE 流式)
                    |                   |
+-------------------v-------------------v------------------+
|                  FastAPI (src/api.py)                      |
|  OpenAI 兼容 API · SSE 流式 · 多会话 · 用户管理 · 管理后台  |
+-------------------+-------------------+------------------+
                    |                   |
+-------------------v---------+  +------v-----------------+
|   SmartDispatcher (3B)      |  |   会话 & 记忆           |
|  意图识别 -> 能力路由        |  |  session.py            |
|  -> 性能追踪 -> 智能回退     |  |  memory.py             |
+-----------+-----------------+  +-----------------------+
            |
+-----------v--------------------------------------------+
|                 Agent (LLM 工具调用循环)                  |
|                                                         |
|  _plan_task      : 复杂任务先拆解为编号步骤再执行          |
|  asyncio.gather  : 只读工具并行执行，写入工具串行          |
|  自我审查        : write/edit 后自动 lsp_diagnostics      |
|  _compress_msgs  : 自动裁剪早期工具结果控制 context       |
+-----------+--------------------------------------------+
            |
+-----------v--------------------------------------------+
|              MCP Combined Server (31 tools)               |
|                                                          |
|  +----------+ +----------+ +----------+ +----------+    |
|  |文件操作   | |代码搜索   | |LSP 诊断  | |Git 操作  |    |
|  |5 tools   | |3 tools   | |5 tools   | |8 tools   |    |
|  +----------+ +----------+ +----------+ +----------+    |
|  +----------+ +----------+ +----------+ +----------+    |
|  |网络      | |测试      | |文档查询   | |Skill     |    |
|  |2 tools   | |1 tool    | |2 tools   | |3 tools   |    |
|  +----------+ +----------+ +----------+ +----------+    |
+---------------------------------------------------------+
            |
+-----------v--------------------------------------------+
|              本地推理 + 云端回退                           |
|  Qwen2.5-Coder-7B (llama.cpp)  <->  Gemini API          |
+---------------------------------------------------------+
```

### 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 前端 | HTMX + Alpine.js + TailwindCSS | 响应式 Web UI，零构建 |
| 后端 | Python 3.12 + FastAPI | 异步 API，SSE 流式 |
| 认证 | Cookie + 不透明 Token (SQLite) | PBKDF2 密码哈希，会话管理 |
| Agent | LLM + MCP 工具循环 | prompt-based 工具调用 |
| 本地推理 | llama.cpp (llama-cpp-python) | Mac M4 优化 |
| 云端回退 | Gemini API (Google AI Studio) | 免费额度 |
| 代码索引 | Codegraph (SQLite FTS5) | 符号搜索 + 调用图 |
| 类型诊断 | basedpyright | Python 类型检查 |
| 知识库 | ChromaDB | 文档语义检索 |
| 任务调度 | APScheduler | 定时/Cron 任务 |
| 桌面版 | Tauri v2 + PyInstaller | 原生应用打包 |

### 目录结构

```
coolclaw/
+-- main.py                    # 入口
+-- config.yaml                # 配置（模型路径、代理、日志）
+-- web/                       # 静态 Web UI
|   +-- index.html             # AI 对话（需登录）
|   +-- login.html             # 登录 / 注册页
|   +-- tasks.html             # 任务管理
|   +-- logs.html              # 执行日志
|   +-- settings.html          # 管理后台（用户管理、系统设置）
+-- src/
|   +-- api.py                 # FastAPI 端点（含鉴权路由、管理员 CRUD）
|   +-- auth.py                # Cookie 鉴权中间件 + Token 管理
|   +-- database.py            # SQLite 模块（users/sessions/settings/preferences）
|   +-- agent.py               # Agent 核心（LLM + MCP 工具调用循环）
|   +-- dispatcher.py          # SmartDispatcher（意图路由 + 系统提示）
|   +-- session.py             # 会话管理（多会话、历史记录）
|   +-- memory.py              # 跨会话记忆（ProjectMemory）
|   +-- models.py              # 模型注册表
|   +-- config.py              # 配置解析
|   +-- router.py              # OpenAI 兼容 API 路由
|   +-- local_llm.py           # llama.cpp 本地推理
|   +-- gemini_client.py       # Gemini API 客户端
|   +-- proxy.py               # HTTP/SOCKS5 代理支持
|   +-- rate_limiter.py        # 速率限制 + 429 退避
|   +-- sandbox.py             # 跨平台沙箱（Seatbelt/Landlock/bwrap）
|   +-- mcp_server.py          # 文件操作 MCP Server
|   +-- mcp/
|   |   +-- combined.py        # 合并 MCP Server（31 个工具）
|   |   +-- codegraph.py       # 代码图谱（SQLite 直查）
|   |   +-- lsp.py             # LSP 诊断（basedpyright）
|   |   +-- context7.py        # Context7 文档查询
|   |   +-- websearch.py       # 网络搜索（DDG + httpx）
|   +-- skills/                # Skill 系统
|   |   +-- __init__.py        # 公共 API 导出
|   |   +-- registry.py        # 自动发现 + 注册
|   |   +-- runner.py          # 子进程执行 + 超时控制
|   |   +-- builtin/           # 内置 Skill
|   |       +-- weather/       # 天气查询（wttr.in）
|   |       +-- docx_template/ # Word 文档生成/填充
|   +-- capabilities/          # 调度能力模块
|   +-- providers/             # 云 API Provider
|   +-- knowledge_base.py      # RAG 知识库
|   +-- task_manager.py        # 定时任务管理
|   +-- storage.py             # 加密存储
+-- skills/                    # 用户自定义 Skill 目录
+-- desktop/                   # 桌面版 (Tauri v2)
|   +-- backend/
|   |   +-- build.py           # PyInstaller 构建脚本
|   |   +-- main_wrapper.py    # 冻结模式入口
|   |   +-- dist/              # 构建产出 (git-ignored)
|   +-- src-tauri/
|   |   +-- src/lib.rs         # Rust 主进程
|   |   +-- tauri.conf.json    # Tauri 配置
|   |   +-- capabilities/      # Shell 权限
|   +-- dist/                  # 前端静态文件
|   +-- package.json           # Node.js 依赖
+-- .codegraph/                # Codegraph 索引（SQLite）
```

## MCP 工具清单 (31 个)

### 文件操作 (5)

| 工具 | 说明 | 读/写 |
|------|------|-------|
| `list_files` | 列出目录内容（递归可选） | 只读 |
| `read_file` | 读取文件内容（支持行范围） | 只读 |
| `write_file` | 创建/覆盖写入文件 | 写入 |
| `edit_file` | 精确文本替换（old_string -> new_string） | 写入 |
| `run_command` | 执行 shell 命令（allowlist + Seatbelt 沙箱） | 写入 |

### 代码搜索 — Codegraph (3)

基于 SQLite FTS5 全文搜索 + 调用图数据库，直接查询 `.codegraph/codegraph.db`，无需外部 CLI。

| 工具 | 说明 |
|------|------|
| `codegraph_search` | FTS5 全文搜索符号名/签名/文档 |
| `codegraph_explore` | 搜索 + 源码片段 + 调用路径 + 子节点 |
| `codegraph_list_symbols` | 列出文件内所有符号 |

### LSP 诊断 (5)

基于 basedpyright CLI + Codegraph 回退。

| 工具 | 说明 |
|------|------|
| `lsp_diagnostics` | basedpyright 类型诊断（JSON 格式） |
| `lsp_goto_definition` | 跳转到定义 |
| `lsp_find_references` | 查找引用 |
| `lsp_rename` | 重命名建议（仅建议，不自动修改） |
| `lsp_document_symbols` | 文档符号列表 |

### Git 操作 (8)

| 工具 | 说明 | 读/写 |
|------|------|-------|
| `git_status` | 工作区状态（--short） | 只读 |
| `git_log` | 提交历史（可指定条数） | 只读 |
| `git_diff` | 文件差异（工作区 vs HEAD） | 只读 |
| `git_diff_staged` | 已暂存差异 | 只读 |
| `git_blame` | 逐行注释（支持行范围） | 只读 |
| `git_branch` | 分支列表 | 只读 |
| `git_commit` | 暂存 + 提交（可指定文件） | 写入 |
| `git_checkout` | 切换分支 | 写入 |

### 网络 (2)

| 工具 | 说明 |
|------|------|
| `web_search` | DuckDuckGo 搜索 |
| `fetch_url` | HTTP 抓取 URL 内容 |

### 测试 (1)

| 工具 | 说明 |
|------|------|
| `run_test` | 自动检测 pytest/unittest，返回结构化 pass/fail 结果 |

### 文档查询 (2)

| 工具 | 说明 | 状态 |
|------|------|------|
| `resolve_library_id` | Context7 库 ID 解析 | GFW 封锁 |
| `query_docs` | Context7 文档查询 | GFW 封锁 |

### 代码库理解 (1)

| 工具 | 说明 |
|------|------|
| `clone_and_index` | 克隆 GitHub 仓库并自动索引（codegraph） |

### Skill 系统 (3)

| 工具 | 说明 |
|------|------|
| `list_skills` | 列出所有可用 Skill（内置 + 用户自定义） |
| `run_skill` | 执行指定 Skill（子进程隔离，超时控制） |
| `create_skill` | 创建新 Skill（自动写入 skills/ 目录） |

## Skill 系统

CoolClaw 的 Skill 系统允许用户定义可复用的自动化任务，通过 MCP 工具调用执行。

### 架构

```
src/skills/
+-- __init__.py        # 公共 API：SkillRegistry, SkillInfo, run_skill
+-- registry.py        # 自动发现 + manifest.json 解析
+-- runner.py          # 子进程执行（隔离、超时、JSON 输出）
+-- builtin/           # 内置 Skill
|   +-- weather/       # 天气查询
|   +-- docx_template/ # Word 文档生成/填充

skills/                # 用户自定义 Skill 目录（热加载）
+-- my_skill/
|   +-- manifest.json  # 元数据（名称、参数、依赖）
|   +-- main.py        # 执行入口（定义 run() 函数）
```

### 内置 Skill

#### 天气查询 (`weather`)

查询全球城市天气预报，数据来源 wttr.in（免费，无需 API Key）。

```
run_skill("weather", {"city": "Shanghai", "days": 2})
```

**参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `city` | string | ✅ | 城市名称（支持中英文） |
| `days` | integer | ❌ | 预报天数（1-3，默认 2） |

**返回示例：**

```json
{
  "location": "Pootung, China",
  "current": {
    "temperature": "28",
    "feels_like": "30",
    "humidity": "81",
    "wind_speed": "19",
    "description": "Partly Cloudy"
  },
  "forecast": [...]
}
```

#### Word 文档生成 (`docx_template`)

基于 python-docx 生成或填充 Word 文档，支持 `{{variable}}` 模板变量。

```
# 从内容创建
run_skill("docx_template", {
  "content": "Hello World\n\nThis is a test.",
  "output": "output.docx"
})

# 填充模板
run_skill("docx_template", {
  "template": "template.docx",
  "data": {"name": "张三", "date": "2026-08-23"},
  "output": "filled.docx"
})
```

**参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `output` | string | ✅ | 输出文件路径 |
| `content` | string | ❌ | 文本内容（无模板时使用） |
| `template` | string | ❌ | 模板文件路径 |
| `data` | object | ❌ | 模板变量键值对 |

### 创建自定义 Skill

#### 1. 目录结构

```
skills/my_skill/
+-- manifest.json
+-- main.py
```

#### 2. manifest.json

```json
{
  "name": "my_skill",
  "description": "我的自定义 Skill",
  "version": "1.0.0",
  "parameters": {
    "input": {
      "type": "string",
      "description": "输入参数",
      "required": true
    },
    "count": {
      "type": "integer",
      "description": "数量（默认 1）",
      "default": 1,
      "required": false
    }
  },
  "dependencies": [],
  "entry": "main.py"
}
```

#### 3. main.py

```python
"""My custom skill."""


def run(input: str, count: int = 1) -> dict:
    """执行 Skill。

    Args:
        input: 输入参数
        count: 数量

    Returns:
        结果字典
    """
    result = [f"Item {i}: {input}" for i in range(count)]

    return {
        "input": input,
        "count": count,
        "items": result,
    }
```

#### 4. MCP 工具调用

```
# 列出所有 Skill
list_skills()

# 执行
run_skill("my_skill", {"input": "hello", "count": 3})

# 创建（通过 MCP 动态创建）
create_skill(
  name="dynamic_skill",
  description="动态创建的 Skill",
  parameters={"text": {"type": "string", "description": "文本"}},
  entry_code='def run(text): return {"echo": text.upper()}'
)
```

### 执行机制

```
run_skill("weather", {"city": "Beijing"})
    |
    +-- 1. SkillRegistry.scan()      # 发现 skill 目录
    +-- 2. _check_dependencies()     # 检查 Python 包（httpx, docx 等）
    +-- 3. validate_params()         # 验证参数类型和必填项
    +-- 4. _apply_defaults()         # 合并默认值
    +-- 5. subprocess.run()          # 子进程执行（stdin 传参，超时 30s）
    +-- 6. json.loads(stdout)        # 解析 JSON 输出
```

**安全特性：**
- **子进程隔离**：每个 Skill 在独立进程中运行，崩溃不影响主进程
- **依赖检查**：执行前自动检查 Python 包是否安装
- **超时控制**：默认 30 秒超时，防止无限阻塞
- **参数校验**：类型检查 + 必填项验证

## Agent 核心能力

### 1. 任务规划 (`_plan_task`)

复杂任务先让 LLM 拆解为编号步骤，再逐步执行：

```
用户：帮我给 agent 添加重试机制和超时控制

规划输出：
1. [读取] 读取 src/agent.py 了解当前结构
2. [搜索] codegraph_search 找到相关类和方法
3. [诊断] lsp_diagnostics 检查当前类型错误
4. [编辑] edit_file 添加重试装饰器
5. [编辑] edit_file 添加超时参数
6. [诊断] lsp_diagnostics 验证修改无报错
7. [测试] run_test 运行相关测试
8. [Git] git_commit 提交修改
```

### 2. 并行工具执行

只读工具（read_file, codegraph_search, lsp_diagnostics 等）通过 `asyncio.gather` 并行执行，写入工具（write_file, edit_file）保持串行，最大化吞吐：

```
并行批次: [codegraph_search("Agent"), read_file("src/agent.py"), lsp_diagnostics("src/agent.py")]
    -> 3 个工具同时执行

串行批次: [edit_file(修改), edit_file(修改)]  (写入操作必须顺序执行)
```

### 3. 自我审查

每次 write/edit 操作后，自动对修改的 `.py` 文件运行 `lsp_diagnostics`，如有类型错误会反馈给 LLM 修复：

```
edit_file("src/agent.py", ...) -> 成功
  -> 自动: lsp_diagnostics("src/agent.py")
  -> 发现 3 个 error
  -> 注入消息: "[Self-Review] src/agent.py 存在类型/语法错误: ... 请修复"
  -> LLM 自动修复
```

### 4. 上下文管理 (`_compress_old_messages`)

当消息历史过长（>12000 字符）时，自动裁剪早期工具结果（只保留前 6 行 + 摘要），防止 context window 溢出。

### 5. 跨会话记忆 (`ProjectMemory`)

持久化存储到 `.coolclaw_memory/` 目录：

| 功能 | 方法 | 说明 |
|------|------|------|
| 项目决策 | `record_decision()` | 记录技术选型和原因 |
| 错误学习 | `record_error()` | 记录错误和修复方案 |
| 工作摘要 | `save_summary()` | 记录每次工作内容 |
| 上下文注入 | `get_context_block()` | 生成摘要注入新对话 |

## 快速开始

### 1. 安装依赖

```bash
cd coolclaw
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置

编辑 `config.yaml`：

```yaml
# 模型路径
models:
  local:
    qwen2_5_coder_7b:
      path: "/path/to/qwen2.5-coder-7b-instruct-q4_k_m.gguf"

# 代理（中国用户需要）
network:
  http_proxy: "http://127.0.0.1:20171"
  https_proxy: "http://127.0.0.1:20171"

# 路由模式
routing:
  mode: "cloud_only"    # cloud_only | local_only | auto
```

### 3. 构建 Codegraph 索引（可选，用于代码搜索）

```bash
# 在项目根目录运行
codegraph init
```

### 4. 启动服务

```bash
python main.py
# 或指定端口
python main.py --port 8484
```

### 5. 访问

- Web UI: http://localhost:8484
- 默认管理员: 用户名 `admin`，密码 `admin123`
- 首次访问自动跳转登录页，登录后进入 AI 对话界面
- 管理后台: http://localhost:8484/settings.html（管理员可管理用户、系统设置）
- 同步 API: POST http://localhost:8484/api/chat
- 流式 API: POST http://localhost:8484/api/chat/stream (SSE)
- WebSocket: ws://localhost:8484/ws/chat

### API 使用示例

```bash
# 同步调用
curl -X POST http://localhost:8484/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "帮我看看 src/agent.py 有什么问题", "session_id": "test"}'

# 流式调用 (SSE)
curl -X POST http://localhost:8484/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "创建一个 hello.py 文件", "session_id": "test"}'
```

### 鉴权 API

```bash
# 登录（返回 Set-Cookie: coolclaw_session=...）
curl -X POST http://localhost:8484/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# 注册新用户
curl -X POST http://localhost:8484/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "user1", "password": "pass1234"}'

# 查看当前用户（需带 Cookie）
curl -b "coolclaw_session=<token>" http://localhost:8484/api/auth/me

# 登出
curl -b "coolclaw_session=<token>" -X POST http://localhost:8484/api/auth/logout
```

### 管理员 API（需管理员权限）

```bash
# 查看所有用户
curl -b "coolclaw_session=<token>" http://localhost:8484/api/admin/users

# 创建用户
curl -b "coolclaw_session=<token>" -X POST http://localhost:8484/api/admin/users \
  -H "Content-Type: application/json" \
  -d '{"username": "dev", "password": "dev12345", "is_admin": false}'

# 更新用户
curl -b "coolclaw_session=<token>" -X PUT http://localhost:8484/api/admin/users/2 \
  -H "Content-Type: application/json" \
  -d '{"display_name": "开发者", "is_admin": false}'

# 删除用户
curl -b "coolclaw_session=<token>" -X DELETE http://localhost:8484/api/admin/users/2

# 查看系统设置
curl -b "coolclaw_session=<token>" http://localhost:8484/api/admin/settings

# 更新系统设置
curl -b "coolclaw_session=<token>" -X PUT http://localhost:8484/api/admin/settings \
  -H "Content-Type: application/json" \
  -d '{"key": "rate_limit_rpm", "value": "20"}'
```

## 桌面版打包与发布

CoolClaw 使用 Tauri v2 + PyInstaller 构建原生桌面应用，Python 后端以 sidecar 形式嵌入。

### 架构概览

```
CoolClaw.app (macOS)
├── Contents/
│   ├── MacOS/
│   │   ├── coolclaw-desktop    # Tauri Rust 主进程 (12MB)
│   │   └── backend             # PyInstaller --onefile (43MB)
│   ├── Resources/
│   │   ├── icon.icns
│   │   └── _up_/               # Tauri 资源映射
│   └── Info.plist
```

**启动流程**: `coolclaw-desktop` → spawn `backend --port 8484` → FastAPI 就绪 → 加载 WebView UI

### 前置条件

| 依赖 | 版本 | 用途 |
|------|------|------|
| Rust | ≥ 1.77 | Tauri 编译 |
| Node.js | ≥ 18 | Tauri CLI |
| Python | 3.12 | 后端构建 |
| PyInstaller | ≥ 6.0 | Python → 单文件二进制 |
| Tauri CLI | v2.x | `npx tauri build` |

### 构建步骤

#### 1. 构建 Python 后端 (PyInstaller)

```bash
# 从项目根目录
source .venv/bin/activate
pip install pyinstaller

# 构建 --onefile 二进制
.venv/bin/python desktop/backend/build.py

# 产出: desktop/backend/dist/backend (43MB)
```

**关键配置** (`desktop/backend/build.py`):
- `--onefile`: 单文件模式，运行时解压到 `sys._MEIPASS`
- `--add-data web:web`: 打包 Web UI 静态文件
- `--add-data config.yaml:.`: 打包默认配置
- `--add-binary llama_cpp/lib/*.dylib:llama_cpp/lib`: 打包 llama.cpp 原生库
- `--hidden-import`: 显式声明所有动态导入模块

#### 2. 准备 Tauri Sidecar

Tauri 的 `externalBin` 要求 sidecar 命名格式为 `{name}-{target-triple}`：

```bash
# macOS ARM64
cp desktop/backend/dist/backend desktop/backend/dist/backend-aarch64-apple-darwin
chmod +x desktop/backend/dist/backend-aarch64-apple-darwin
```

**已知限制**: Tauri v2 的 `bundle.resources` 无法正确打包 `--onedir` 目录（`_internal/` 含数千文件导致资源遍历失败），因此使用 `--onefile` 模式，冷启动约 20 秒。

#### 3. 构建 Tauri 应用

```bash
cd desktop

# 开发模式 (连接已运行的后端)
npx tauri dev

# 生产构建
npx tauri build
```

**产出**:
- `src-tauri/target/release/bundle/macos/CoolClaw.app`
- `src-tauri/target/release/bundle/dmg/CoolClaw_1.0.0_aarch64.dmg`

#### 4. 测试构建产物

```bash
# 启动应用
open src-tauri/target/release/bundle/macos/CoolClaw.app

# 验证后端启动 (~20s 冷启动)
sleep 25
curl http://localhost:8484/api/chat -X POST \
  -H "Content-Type: application/json" \
  -d '{"message": "hi", "session_id": "test"}'
```

### 应用行为

| 行为 | 说明 |
|------|------|
| **自动启动后端** | App 启动时自动 spawn `backend --port 8484` |
| **关闭窗口** | 最小化到系统托盘，不退出 |
| **托盘菜单** | Show Window / Hide Window / Quit CoolClaw |
| **退出应用** | 系统托盘 → Quit，自动 kill 后端进程 |
| **开发模式** | Sidecar 失败时自动回退到 `.venv/bin/python main.py` |

### 项目结构

```
desktop/
├── backend/
│   ├── build.py              # PyInstaller 构建脚本
│   ├── main_wrapper.py       # 冻结模式入口 (路径解析 + llama_cpp stub)
│   ├── dist/                 # 构建产出 (git-ignored)
│   │   ├── backend           # --onefile 二进制
│   │   └── backend-aarch64-apple-darwin  # Tauri sidecar 命名
│   └── build/                # PyInstaller 临时文件 (git-ignored)
├── src-tauri/
│   ├── src/lib.rs            # Rust 主进程 (sidecar 管理 + 系统托盘)
│   ├── tauri.conf.json       # Tauri 配置
│   ├── capabilities/         # Shell 权限配置
│   └── icons/                # 应用图标
├── dist/                     # 前端静态文件 (WebView 加载)
│   └── index.html            # 桌面启动器 (iframe → localhost:8484)
└── package.json              # Node.js 依赖 (Tauri CLI)
```

## 开发者手册

### 环境搭建

```bash
# 1. 克隆仓库
git clone https://github.com/your-repo/coolclaw.git
cd coolclaw

# 2. Python 环境
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Rust 环境 (桌面版开发)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 4. Node.js 环境 (Tauri CLI)
cd desktop && npm install

# 5. 可选: 构建 Codegraph 索引
codegraph init
```

### 开发模式

#### Web 后端开发

```bash
# 启动后端 (热重载需手动重启)
python main.py --port 8484

# 或使用 uvicorn 热重载
uvicorn src.api:create_app --factory --reload --port 8484
```

#### Skill 开发

```bash
# 查看已注册 Skill
python -c "
from src.skills.registry import SkillRegistry
reg = SkillRegistry()
reg.scan()
for name, s in reg.skills.items():
    print(f'{name}: {s.description} (source={s.source})')
"

# 测试执行
python -c "
from src.skills.runner import run_skill
result = run_skill('weather', {'city': 'Shanghai'})
print(result)
"

# 创建新 Skill 目录
mkdir -p skills/my_skill
# 然后创建 manifest.json 和 main.py
```

#### 桌面版开发

```bash
cd desktop

# 开发模式: 自动连接 http://localhost:8484
npx tauri dev

# 需要先启动后端:
python ../main.py --port 8484 &
npx tauri dev
```

### 调试技巧

#### Rust 代码调试

```bash
# 编译检查
cargo check --manifest-path desktop/src-tauri/Cargo.toml

# 查看日志
RUST_LOG=debug npx tauri dev
```

#### Python 后端调试

```bash
# 冻结模式路径解析
python -c "
import sys
sys.frozen = True
sys._MEIPASS = '/tmp/test'
exec(open('desktop/backend/main_wrapper.py').read())
"

# 验证 llama_cpp stub
python -c "
from desktop.backend.main_wrapper import _stub_heavy_imports
_stub_heavy_imports()
import llama_cpp
print('Stub loaded:', llama_cpp.Llama)
"
```

#### Tauri Sidecar 调试

```bash
# 直接运行 sidecar
./desktop/backend/dist/backend-aarch64-apple-darwin --port 8484

# 查看 sidecar 输出
RUST_LOG=tauri_plugin_shell=debug npx tauri dev
```

### 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| Sidecar 启动失败 | 二进制不存在或命名错误 | 检查 `dist/backend-aarch64-apple-darwin` 是否存在 |
| 后端 503 Service Unavailable | FastAPI 尚未就绪 | 等待 ~20s (冷启动解压时间) |
| `llama_cpp` 导入失败 | 原生库未打包 | 运行 `build.py` 时自动处理 |
| WebView 白屏 | 前端文件未打包 | 检查 `desktop/dist/index.html` |
| 端口冲突 | 8484 已被占用 | `lsof -i :8484` 查看并 kill |

### 贡献流程

1. Fork 仓库
2. 创建功能分支: `git checkout -b feature/amazing-feature`
3. 提交更改: `git commit -m 'Add amazing feature'`
4. 推送分支: `git push origin feature/amazing-feature`
5. 创建 Pull Request

### 代码规范

- **Python**: 遵循 PEP 8，使用 `ruff` 格式化
- **Rust**: 使用 `cargo fmt` + `cargo clippy`
- **提交信息**: 使用 [Conventional Commits](https://www.conventionalcommits.org/)

## 运维手册

### 部署架构

```
┌─────────────────────────────────────────────────────┐
│                    用户机器                           │
├─────────────────────────────────────────────────────┤
│  CoolClaw.app                                       │
│  ├── coolclaw-desktop (Rust 进程)                    │
│  │   ├── WebView 渲染 UI                             │
│  │   └── 管理 backend 子进程                         │
│  └── backend (Python 进程)                           │
│      ├── FastAPI 服务 (port 8484)                   │
│      ├── Agent 循环 (LLM + MCP)                     │
│      └── 本地推理 / 云端回退                          │
└─────────────────────────────────────────────────────┘
```

### 端口与进程

| 端口 | 进程 | 说明 |
|------|------|------|
| 8484 | backend | FastAPI HTTP 服务 |
| - | coolclaw-desktop | Tauri 主进程 (无网络端口) |

### 日志管理

#### 应用日志

```bash
# 查看后端日志 (stdout/stderr)
# Tauri 应用日志输出到终端，可通过 Console.app 查看

# macOS Console.app
open -a Console
# 搜索 "CoolClaw" 或 "backend"
```

#### 日志级别配置

编辑 `config.yaml`:

```yaml
logging:
  level: "INFO"       # DEBUG | INFO | WARNING | ERROR
  format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
```

### 进程管理

#### 启动/停止

```bash
# 启动应用
open /Applications/CoolClaw.app

# 停止应用 (优雅退出)
pkill -f "coolclaw-desktop"

# 强制停止
pkill -9 -f "coolclaw-desktop"
pkill -9 -f "Contents/MacOS/backend"
```

#### 进程监控

```bash
# 查看进程状态
ps aux | grep -E "coolclaw|backend" | grep -v grep

# 查看端口占用
lsof -i :8484

# 查看内存占用
ps -p $(pgrep -f "Contents/MacOS/backend") -o pid,rss,vsz,comm
```

### 故障排查

#### 后端无法启动

```bash
# 1. 检查端口占用
lsof -i :8484
# 如果有进程占用: kill <PID>

# 2. 手动测试 sidecar
./desktop/backend/dist/backend-aarch64-apple-darwin --port 8485
# 观察输出日志

# 3. 检查配置文件
cat config.yaml | grep -E "proxy|model|routing"
```

#### 内存占用过高

```bash
# 查看内存
ps -eo pid,rss,comm | grep backend | awk '{print $1, $2/1024 "MB", $3}'

# 优化建议:
# - 使用 cloud_only 模式 (不加载本地模型)
# - 减少 context window 大小
# - 定期重启应用
```

#### 网络代理问题

```bash
# 验证代理配置
curl -x http://127.0.0.1:20171 https://api.google.com

# 检查 config.yaml
network:
  http_proxy: "http://127.0.0.1:20171"
  https_proxy: "http://127.0.0.1:20171"
```

### 更新与回滚

#### 更新应用

```bash
# 1. 备份配置
cp config.yaml config.yaml.bak

# 2. 拉取最新代码
git pull origin main

# 3. 重新构建
.venv/bin/python desktop/backend/build.py
cd desktop && npx tauri build

# 4. 替换应用
cp -r src-tauri/target/release/bundle/macos/CoolClaw.app /Applications/
```

#### 回滚版本

```bash
# 查看提交历史
git log --oneline -10

# 回滚到指定版本
git checkout <commit-hash>

# 重新构建
.venv/bin/python desktop/backend/build.py
cd desktop && npx tauri build
```

### 安全注意事项

- **沙箱**: 命令执行受 allowlist + OS 原生沙箱保护（macOS Seatbelt / Linux Landlock），禁止 `rm -rf`、`kill *` 等危险操作
- **端口**: 仅监听 localhost，不对外暴露
- **配置**: `config.yaml` 包含 API 密钥，勿提交到公开仓库
- **日志**: 避免在日志中输出敏感信息 (API Key、密码等)

### 性能调优

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `routing.mode` | `cloud_only` | 本地推理需 >8GB 内存 |
| `logging.level` | `INFO` | 生产环境建议 `WARNING` |
| `agent.max_iterations` | 20 | Agent 循环最大次数 |
| `agent.context_limit` | 12000 | Context window 字符限制 |

## 性能数据 (Mac M4 16GB)

| 模型 | 速度 | 内存占用 | 用途 |
|------|------|----------|------|
| Qwen2.5-3B | 31 t/s | ~2 GB | 调度路由（意图识别） |
| Qwen2.5-Coder-7B | 14 t/s | ~5 GB | 代码开发 Agent |
| LightOnOCR-2-1B | 45 t/s | ~1 GB | 文档 OCR |

## 配置说明

### 路由模式

| 模式 | 说明 |
|------|------|
| `cloud_only` | 仅使用云端 API（Gemini） |
| `local_only` | 仅使用本地模型（llama.cpp） |
| `auto` | 优先本地，失败回退云端 |

### 代理设置

```yaml
network:
  http_proxy: "http://127.0.0.1:20171"
  socks5_proxy: "socks5://127.0.0.1:20170"
```

### 速率限制

- `rate_limiter.py`: Token bucket + 429 退避
- `peer_marking`: 标记失败 Provider，自动冷却
- `fallback_chain`: 自动回退到下一个 Provider

## 开发计划

### Phase 1: 基础架构

- [x] 调度中心架构
- [x] 专家模型系统
- [x] 基础能力（搜索、剪贴板）
- [x] 定时任务
- [x] Web UI

### Phase 2: 能力扩展

- [x] 钉钉/飞书/邮件/Telegram 通知
- [x] ChromaDB 知识库 + RAG
- [x] 调度优化（重试、回退、速率限制）
- [x] 代理支持（HTTP/SOCKS5）

### Phase 3: AI Agent 核心

- [x] MCP 工具系统（27 个工具）
- [x] Agent 循环（LLM 思考 + 工具调用）
- [x] 任务规划 (`_plan_task`)
- [x] 并行工具执行 (`asyncio.gather`)
- [x] 自我审查（写完自动诊断）
- [x] 上下文管理（自动裁剪）
- [x] 跨会话记忆 (`ProjectMemory`)
- [x] SSE 流式输出
- [x] Git 集成（8 个工具）
- [x] 测试运行器
- [x] 代码搜索（Codegraph SQLite 直查）
- [x] LSP 诊断（basedpyright）

### Phase 4: 高级能力

- [x] 并行子 Agent（多 Agent 协作）→ Orchestrator Pipeline
- [x] 持久化记忆（向量化错误学习）→ ProjectMemory + User Preferences
- [x] 工作区沙箱（命令拦截）→ allowlist + Seatbelt/Landlock
- [x] WebSocket 实时通信 → /ws/chat
- [x] 多语言支持（TypeScript/Go/Rust/C/C++/Java）→ LSP auto-detect
- [x] 代码库克隆理解 → clone_and_index()
- [x] 自我反思与回滚 → auto git stash on edit failures
- [x] 桌面版打包 (Tauri)
- [x] Skill 系统 → 自动发现 + 子进程执行 + MCP 工具
- [x] 用户管理 → 登录/注册、Cookie 鉴权、管理员 CRUD、用户偏好
- [x] 管理后台 → 用户管理、系统设置、通知渠道、知识库配置

## 已知限制

| 限制 | 说明 | 状态 |
|------|------|------|
| Context7 | `api.context7.com` 被 GFW 封锁 | 不可修复（需 VPN） |
| 本地模型 | 7B 模型工具调用准确率有限 | 升级更大模型可改善 |

## License

MIT
