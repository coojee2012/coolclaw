"""
Tool Schema Registry — 27 tools in OpenAI Function Calling format.

Static schemas for:
- Documentation and validation
- Fallback when MCP server is unavailable
- Schema-based LLM output validation
- Role-based tool filtering (B2)
"""
from typing import Any

# ── Tool metadata (context tags, read/write classification) ──

TOOL_CONTEXT: dict[str, dict[str, Any]] = {
    # Analysis context (B2: architect role)
    "codegraph_search":     {"context": "analysis", "read_only": True},
    "codegraph_explore":    {"context": "analysis", "read_only": True},
    "codegraph_list_symbols": {"context": "analysis", "read_only": True},
    "lsp_diagnostics":      {"context": "analysis", "read_only": True},
    "lsp_goto_definition":  {"context": "analysis", "read_only": True},
    "lsp_find_references":  {"context": "analysis", "read_only": True},
    "lsp_document_symbols": {"context": "analysis", "read_only": True},
    "read_file":            {"context": "analysis", "read_only": True},
    "list_files":           {"context": "analysis", "read_only": True},

    # Implementation context (B2: coder role)
    "write_file":           {"context": "implementation", "read_only": False},
    "edit_file":            {"context": "implementation", "read_only": False},
    "run_command":          {"context": "implementation", "read_only": False},
    "lsp_rename":           {"context": "implementation", "read_only": False},

    # Verification context (B2: reviewer role)
    "run_test":             {"context": "verification", "read_only": False},
    "git_diff":             {"context": "verification", "read_only": True},
    "git_diff_staged":      {"context": "verification", "read_only": True},
    "git_blame":            {"context": "verification", "read_only": True},

    # Version control context
    "git_status":           {"context": "version_control", "read_only": True},
    "git_log":              {"context": "version_control", "read_only": True},
    "git_branch":           {"context": "version_control", "read_only": True},
    "git_commit":           {"context": "version_control", "read_only": False},
    "git_checkout":         {"context": "version_control", "read_only": False},

    # Network context
    "web_search":           {"context": "network", "read_only": True},
    "fetch_url":            {"context": "network", "read_only": True},

    # Documentation context
    "resolve_library_id":   {"context": "documentation", "read_only": True},
    "query_docs":           {"context": "documentation", "read_only": True},
}

# Role → allowed context tags (B2 multi-agent)
ROLE_TOOL_CONTEXTS: dict[str, set[str]] = {
    "architect":  {"analysis", "documentation", "network"},
    "coder":      {"implementation", "analysis", "version_control"},
    "reviewer":   {"verification", "analysis", "version_control"},
    "full":       set(TOOL_CONTEXT.keys()),  # all tools
}


def get_tools_for_role(role: str) -> set[str]:
    """Return allowed tool names for a given agent role."""
    contexts = ROLE_TOOL_CONTEXTS.get(role, ROLE_TOOL_CONTEXTS["full"])
    return {
        name for name, meta in TOOL_CONTEXT.items()
        if meta["context"] in contexts or role == "full"
    }


# ── OpenAI Function Calling Schemas ──

TOOL_SCHEMAS: list[dict[str, Any]] = [
    # === File Operations ===
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出目录内容（递归可选）",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对于工作目录的路径，默认 '.'"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文件内容（支持行范围）",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对于工作目录的文件路径"},
                    "offset": {"type": "integer", "description": "起始行号（从1开始）", "minimum": 1, "default": 1},
                    "limit": {"type": "integer", "description": "最大行数", "minimum": 1, "maximum": 2000, "default": 500},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "创建或覆盖写入文件。写入后会自动进行 LSP 诊断检查。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对于工作目录的文件路径"},
                    "content": {"type": "string", "description": "要写入的文件内容"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "精确文本替换（old_string → new_string）",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对于工作目录的文件路径"},
                    "old_str": {"type": "string", "description": "要替换的原始文本"},
                    "new_str": {"type": "string", "description": "替换后的新文本"},
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "执行终端命令。危险命令（rm -rf, sudo, chmod 777 等）会被沙箱拦截。",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的终端命令"},
                    "timeout": {"type": "integer", "description": "超时时间（秒）", "minimum": 1, "maximum": 120, "default": 30},
                },
                "required": ["command"],
            },
        },
    },

    # === Context7 Documentation ===
    {
        "type": "function",
        "function": {
            "name": "resolve_library_id",
            "description": "解析库名称为 Context7 兼容的 library ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "library_name": {"type": "string", "description": "库名称，如 'react', 'express'"},
                },
                "required": ["library_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_docs",
            "description": "查询库的官方文档",
            "parameters": {
                "type": "object",
                "properties": {
                    "library_id": {"type": "string", "description": "Context7 library ID（格式: /org/project）"},
                    "query": {"type": "string", "description": "查询内容"},
                },
                "required": ["library_id", "query"],
            },
        },
    },

    # === Codegraph ===
    {
        "type": "function",
        "function": {
            "name": "codegraph_search",
            "description": "FTS5 全文搜索符号（名称、签名、docstring）",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "limit": {"type": "integer", "description": "最大返回数量", "default": 20},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "codegraph_explore",
            "description": "探索代码符号：搜索 + 源码片段 + 调用路径 + 子节点",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "符号名称或自然语言问题"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "codegraph_list_symbols",
            "description": "列出指定文件中的所有符号",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件路径（相对于工作目录）"},
                },
                "required": ["file_path"],
            },
        },
    },

    # === Websearch ===
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索网络（DuckDuckGo）",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索查询"},
                    "num_results": {"type": "integer", "description": "返回结果数量", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "获取 URL 内容（HTTP GET）",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "要获取的 URL"},
                },
                "required": ["url"],
            },
        },
    },

    # === Git ===
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "查看工作区状态（修改/新增/删除的文件）",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_log",
            "description": "查看提交历史",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "description": "显示最近 N 条提交", "minimum": 1, "maximum": 50, "default": 10},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "查看文件差异（工作区 vs HEAD）",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件路径（可选，不传则显示所有变更）", "default": ""},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff_staged",
            "description": "查看已暂存（staged）的差异",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_blame",
            "description": "查看文件的 git blame（逐行注释）",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件路径"},
                    "start_line": {"type": "integer", "description": "起始行号", "minimum": 1, "default": 0},
                    "end_line": {"type": "integer", "description": "结束行号", "minimum": 0, "default": 0},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_commit",
            "description": "暂存文件并提交",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "提交信息"},
                    "files": {"type": "string", "description": "要暂存的文件（空格分隔），默认全部"},
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_branch",
            "description": "列出所有分支",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_checkout",
            "description": "切换分支",
            "parameters": {
                "type": "object",
                "properties": {
                    "branch": {"type": "string", "description": "分支名称或 commit hash"},
                },
                "required": ["branch"],
            },
        },
    },

    # === LSP ===
    {
        "type": "function",
        "function": {
            "name": "lsp_diagnostics",
            "description": "获取文件的类型诊断（错误、警告）",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件路径"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lsp_goto_definition",
            "description": "跳转到符号定义",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件路径"},
                    "line": {"type": "integer", "description": "行号"},
                    "character": {"type": "integer", "description": "列号"},
                },
                "required": ["file_path", "line", "character"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lsp_find_references",
            "description": "查找符号的所有引用",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件路径"},
                    "line": {"type": "integer", "description": "行号"},
                    "character": {"type": "integer", "description": "列号"},
                },
                "required": ["file_path", "line", "character"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lsp_rename",
            "description": "重命名符号（返回需要修改的位置列表，不自动修改）",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件路径"},
                    "line": {"type": "integer", "description": "行号"},
                    "character": {"type": "integer", "description": "列号"},
                    "new_name": {"type": "string", "description": "新名称"},
                },
                "required": ["file_path", "line", "character", "new_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lsp_document_symbols",
            "description": "列出文档中的所有符号",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件路径"},
                },
                "required": ["file_path"],
            },
        },
    },

    # === Test Runner ===
    {
        "type": "function",
        "function": {
            "name": "run_test",
            "description": "运行测试并返回结构化结果（自动检测 pytest/unittest）",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "测试文件或目录路径（空=全部）", "default": ""},
                    "marker": {"type": "string", "description": "pytest marker 表达式，如 'not slow'", "default": ""},
                    "extra_args": {"type": "string", "description": "额外参数，如 '-v --tb=short'", "default": ""},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clone_and_index",
            "description": "Clone GitHub repo and index with codegraph for codebase understanding",
            "parameters": {
                "type": "object",
                "properties": {
                    "github_url": {"type": "string", "description": "GitHub repo URL (e.g., https://github.com/user/repo)"},
                    "target_dir": {"type": "string", "description": "Target directory name (optional)", "default": ""},
                },
                "required": ["github_url"],
            },
        },
    },
]


# ── Lookup helpers ──

_SCHEMA_BY_NAME: dict[str, dict[str, Any]] = {
    s["function"]["name"]: s for s in TOOL_SCHEMAS
}


def get_tool_schema(name: str) -> dict[str, Any] | None:
    """Get a single tool's OpenAI FC schema by name."""
    return _SCHEMA_BY_NAME.get(name)


def get_all_tool_schemas() -> list[dict[str, Any]]:
    """Get all 27 tool schemas."""
    return TOOL_SCHEMAS


def get_tool_names() -> list[str]:
    """Get all tool names."""
    return [s["function"]["name"] for s in TOOL_SCHEMAS]


def get_tools_for_model(tool_names: set[str] | None = None) -> list[dict[str, Any]]:
    """Filter schemas to only the given tool names. None = all."""
    if tool_names is None:
        return TOOL_SCHEMAS
    return [s for s in TOOL_SCHEMAS if s["function"]["name"] in tool_names]
