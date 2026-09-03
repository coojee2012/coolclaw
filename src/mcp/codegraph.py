"""
Codegraph MCP Server — 代码图谱探索

提供代码符号搜索、调用链分析、依赖关系查询等能力。
通过 codegraph CLI 工具与 SQLite 知识图谱交互。
"""

import json
import os
import subprocess
from typing import Annotated

from pydantic import Field

from mcp.server import MCPServer


def create_codegraph_server(project_path: str = "") -> MCPServer:
    """创建 Codegraph MCP Server"""

    server = MCPServer("codegraph")

    def _run_codegraph(args, timeout=30):
        """执行 codegraph CLI 命令"""
        try:
            cmd = ["codegraph"] + args
            if project_path:
                cmd.extend(["--project", project_path])
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0:
                return "Error: " + (result.stderr.strip() or "codegraph command failed")
            return result.stdout.strip() or "(no output)"
        except FileNotFoundError:
            return "Error: codegraph CLI not found. Install with: npm install -g codegraph"
        except subprocess.TimeoutExpired:
            return "Error: command timed out after " + str(timeout) + "s"
        except Exception as e:
            return "Error: " + str(e)

    @server.tool(
        title="Explore code symbols",
        annotations={"read_only_hint": True, "open_world_hint": False},
    )
    def explore(
        query: Annotated[str, Field(description="Symbol names or natural language question about code")],
        max_files: Annotated[int, Field(description="Maximum files to return", ge=1, le=30)] = 12,
    ) -> str:
        """Explore codebase symbols, call paths, and blast radius"""
        return _run_codegraph(["explore", query, "--max-files", str(max_files)])

    @server.tool(
        title="Search symbols",
        annotations={"read_only_hint": True, "open_world_hint": False},
    )
    def search_symbols(
        query: Annotated[str, Field(description="Symbol name or pattern to search")],
        scope: Annotated[str, Field(description="Search scope: 'document' or 'workspace'")] = "workspace",
    ) -> str:
        """Search for symbols in the codebase"""
        return _run_codegraph(["search", query, "--scope", scope])

    @server.tool(
        title="Get file symbols",
        annotations={"read_only_hint": True, "open_world_hint": False},
    )
    def file_symbols(
        file_path: Annotated[str, Field(description="File path to get symbols from")],
    ) -> str:
        """Get all symbols defined in a file"""
        return _run_codegraph(["symbols", file_path])

    return server
