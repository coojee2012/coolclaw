"""
MCP 工具模块 — 为 Agent 提供扩展能力

包含以下 MCP Server:
- coolclaw-tools: 文件操作（list_files, read_file, write_file, edit_file, run_command）
- context7: 实时文档查询
- codegraph: 代码图谱探索
- websearch: 网络搜索
- lsp: 语言服务（诊断、定义跳转、引用查找、重命名）
"""

from .context7 import create_context7_server
from .codegraph import create_codegraph_server
from .websearch import create_websearch_server
from .lsp import create_lsp_server
from .combined import create_combined_server

__all__ = [
    "create_context7_server",
    "create_codegraph_server",
    "create_websearch_server",
    "create_lsp_server",
    "create_combined_server",
]
