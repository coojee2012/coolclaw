"""
MCP Server — 暴露文件操作和终端命令工具

使用 MCP Python SDK v2，通过 @mcp.tool() 注册工具。
支持 in-memory 连接（Client(mcp)）或 stdio 传输。
"""
import os
import subprocess
from pathlib import Path
from typing import Annotated

from pydantic import Field

from mcp.server import MCPServer


def create_mcp_server(workdir: str = "") -> MCPServer:
    """创建 MCP Server 实例，绑定到指定工作目录"""

    server = MCPServer("coolclaw-tools")
    root = Path(workdir).expanduser().resolve() if workdir else Path.cwd()

    def _safe_path(rel: str) -> Path:
        """将相对路径解析为绝对路径，防止路径逃逸"""
        target = (root / rel).expanduser().resolve()
        if not str(target).startswith(str(root)):
            raise ValueError(f"Path escapes workdir: {rel}")
        return target

    @server.tool(
        title="List files",
        annotations={"read_only_hint": True, "open_world_hint": False},
    )
    def list_files(
        path: Annotated[str, Field(description="相对于工作目录的路径，默认 '.'")] = ".",
    ) -> str:
        """列出目录下的文件和子目录"""
        try:
            target = _safe_path(path)
            if not target.is_dir():
                return f"Error: not a directory: {path}"
            entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            lines = []
            for e in entries[:50]:
                suffix = "/" if e.is_dir() else ""
                size = ""
                if e.is_file():
                    try:
                        size = f"  ({e.stat().st_size} bytes)"
                    except OSError:
                        pass
                lines.append(f"  {e.name}{suffix}{size}")
            return "\n".join(lines) if lines else "(empty directory)"
        except Exception as e:
            return f"Error: {e}"

    @server.tool(
        title="Read file",
        annotations={"read_only_hint": True, "open_world_hint": False},
    )
    def read_file(
        path: Annotated[str, Field(description="相对于工作目录的文件路径")],
        offset: Annotated[int, Field(description="起始行号（从1开始）", ge=1)] = 1,
        limit: Annotated[int, Field(description="最大行数", ge=1, le=2000)] = 500,
    ) -> str:
        """读取文件内容，支持分页"""
        try:
            target = _safe_path(path)
            if not target.is_file():
                return f"Error: not a file: {path}"
            content = target.read_text(errors="replace")
            all_lines = content.splitlines()
            total = len(all_lines)
            start = offset - 1
            selected = all_lines[start : start + limit]
            numbered = [f"{i + offset}: {line}" for i, line in enumerate(selected)]
            header = f"{path} ({total} lines, showing {start + 1}-{min(start + limit, total)})\n"
            return header + "\n".join(numbered)
        except Exception as e:
            return f"Error: {e}"

    @server.tool(
        title="Write file",
        annotations={"read_only_hint": False, "destructive_hint": True, "idempotent_hint": True},
    )
    def write_file(
        path: Annotated[str, Field(description="相对于工作目录的文件路径")],
        content: Annotated[str, Field(description="要写入的文件内容")],
    ) -> str:
        try:
            target = _safe_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            existed = target.exists()
            target.write_text(content, encoding="utf-8")
            lines = content.count("\n") + 1
            action = "覆盖" if existed else "创建"
            return f"已{action}文件 {path} ({lines} 行, {len(content)} 字符)"
        except Exception as e:
            return f"Error: {e}"

    @server.tool(
        title="Edit file",
        annotations={"read_only_hint": False, "destructive_hint": True},
    )
    def edit_file(
        path: Annotated[str, Field(description="相对于工作目录的文件路径")],
        old_str: Annotated[str, Field(description="要替换的原始文本")],
        new_str: Annotated[str, Field(description="替换后的新文本")],
    ) -> str:
        """替换文件中的指定文本"""
        try:
            target = _safe_path(path)
            if not target.is_file():
                return f"Error: not a file: {path}"
            content = target.read_text(errors="replace")
            if old_str not in content:
                return f"Error: old_str not found in {path}"
            count = content.count(old_str)
            new_content = content.replace(old_str, new_str, 1)
            target.write_text(new_content, encoding="utf-8")
            return f"已替换 {path} 中的内容 ({count} 处匹配, 已替换第1处)"
        except Exception as e:
            return f"Error: {e}"

    @server.tool(
        title="Run shell command",
        annotations={"read_only_hint": False, "open_world_hint": True},
    )
    def run_command(
        command: Annotated[str, Field(description="要执行的终端命令")],
        timeout: Annotated[int, Field(description="超时时间（秒）", ge=1, le=120)] = 30,
    ) -> str:
        """执行终端命令并返回输出"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(root),
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {timeout}s"
        except Exception as e:
            return f"Error: {e}"

    return server


# ── Standalone entry point (for testing with `python -m src.mcp_server`) ──
if __name__ == "__main__":
    import sys

    workdir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    server = create_mcp_server(workdir)
    print(f"MCP Server started, workdir={workdir}")
    print(f"Tools: {[t.name for t in server.list_tools()]}")
