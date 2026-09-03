"""
LSP MCP Server — 语言服务

提供代码诊断、定义跳转、引用查找、重命名等能力。
根据项目文件自动选择合适语言的 LSP 服务器。
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Annotated

from pydantic import Field

from mcp.server import MCPServer


LANG_SERVER_MAP = {
    ".py": "basedpyright",
    ".js": "typescript-language-server",
    ".ts": "typescript-language-server",
    ".jsx": "typescript-language-server",
    ".tsx": "typescript-language-server",
    ".go": "gopls",
    ".rs": "rust-analyzer",
    ".rb": "ruby-lsp",
    ".java": "jdtls",
    ".c": "clangd",
    ".cpp": "clangd",
    ".h": "clangd",
    ".hpp": "clangd",
}

LANG_CMD_MAP = {
    "basedpyright": lambda args: [os.sys.executable, "-m", "basedpyright"] + args,
    "typescript-language-server": lambda args: ["typescript-language-server"] + args,
    "gopls": lambda args: ["gopls"] + args,
    "rust-analyzer": lambda args: ["rust-analyzer"] + args,
    "ruby-lsp": lambda args: ["ruby-lsp"] + args,
    "clangd": lambda args: ["clangd"] + args,
}


def _detect_language(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    return LANG_SERVER_MAP.get(ext, "basedpyright")


def _build_lsp_cmd(server: str, args: list[str]) -> list[str]:
    builder = LANG_CMD_MAP.get(server)
    if builder:
        return builder(args)
    return [server] + args


def create_lsp_server(project_path: str = "") -> MCPServer:

    server = MCPServer("lsp")

    def _run_lsp_query(query_type, file_path, **kwargs):
        lang = _detect_language(file_path)

        if lang == "basedpyright":
            cmd = [os.sys.executable, "-m", "basedpyright", f"--outputjson", file_path]
        elif lang == "typescript-language-server":
            cmd = ["typescript-language-server", query_type, file_path]
        else:
            cmd = _build_lsp_cmd(lang, [query_type, file_path])

        for k, v in kwargs.items():
            if v is not None:
                cmd.extend([f"--{k.replace('_', '-')}", str(v)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=project_path or os.getcwd(),
            )
            if result.returncode != 0 and not result.stdout.strip():
                return f"Error ({lang}): " + (result.stderr.strip() or "LSP query failed")
            return result.stdout.strip() or "(no results)"
        except FileNotFoundError:
            return f"Error: {lang} not installed"
        except subprocess.TimeoutExpired:
            return f"Error: {lang} query timed out"
        except Exception as e:
            return f"Error: {e}"

    @server.tool(
        title="Get diagnostics",
        annotations={"read_only_hint": True, "open_world_hint": False},
    )
    def diagnostics(
        file_path: Annotated[str, Field(description="文件路径")],
    ) -> str:
        """获取文件的 LSP 诊断信息（错误、警告）"""
        return _run_lsp_query("diagnostics", file_path)

    @server.tool(
        title="Go to definition",
        annotations={"read_only_hint": True, "open_world_hint": False},
    )
    def goto_definition(
        file_path: Annotated[str, Field(description="文件路径")],
        line: Annotated[int, Field(description="行号（从1开始）", ge=1)],
        character: Annotated[int, Field(description="列号（从0开始）", ge=0)],
    ) -> str:
        """跳转到符号定义位置"""
        return _run_lsp_query("definition", file_path, line=line, character=character)

    @server.tool(
        title="Find references",
        annotations={"read_only_hint": True, "open_world_hint": False},
    )
    def find_references(
        file_path: Annotated[str, Field(description="文件路径")],
        line: Annotated[int, Field(description="行号（从1开始）", ge=1)],
        character: Annotated[int, Field(description="列号（从0开始）", ge=0)],
    ) -> str:
        """查找符号的所有引用"""
        return _run_lsp_query("references", file_path, line=line, character=character)

    @server.tool(
        title="Rename symbol",
        annotations={"read_only_hint": False, "destructive_hint": True},
    )
    def rename_symbol(
        file_path: Annotated[str, Field(description="文件路径")],
        line: Annotated[int, Field(description="行号（从1开始）", ge=1)],
        character: Annotated[int, Field(description="列号（从0开始）", ge=0)],
        new_name: Annotated[str, Field(description="新名称")],
    ) -> str:
        """重命名符号（预览变更，不实际执行）"""
        return _run_lsp_query("rename", file_path, line=line, character=character, new_name=new_name)

    @server.tool(
        title="List document symbols",
        annotations={"read_only_hint": True, "open_world_hint": False},
    )
    def document_symbols(
        file_path: Annotated[str, Field(description="文件路径")],
    ) -> str:
        """列出文件中的所有符号"""
        return _run_lsp_query("symbols", file_path)

    return server
