"""
Context7 MCP Server — 实时文档查询

提供库文档查询能力，支持 React/Next.js/Prisma/Express 等主流框架。
通过 Context7 API 获取最新文档，避免训练数据过期问题。
"""

import json
import os
from typing import Annotated

import httpx
from pydantic import Field

from mcp.server import MCPServer


def create_context7_server() -> MCPServer:
    """创建 Context7 MCP Server"""

    server = MCPServer("context7")
    BASE_URL = "https://api.context7.com/v1"

    @server.tool(
        title="Resolve library ID",
        annotations={"read_only_hint": True, "open_world_hint": True},
    )
    def resolve_library_id(
        library_name: Annotated[str, Field(description="库名称，如 'react', 'next.js', 'prisma'")],
        query: Annotated[str, Field(description="查询意图，用于匹配最相关的库")] = "",
    ) -> str:
        """解析库名称为 Context7 兼容的 library ID"""
        try:
            with httpx.Client(timeout=15) as client:
                resp = client.get(
                    f"{BASE_URL}/ libraries/search",
                    params={"query": library_name},
                )
                resp.raise_for_status()
                data = resp.json()
                libs = data.get("libraries", [])[:5]
                if not libs:
                    return f"未找到库: {library_name}"
                results = []
                for lib in libs:
                    results.append(
                        f"- ID: {lib.get('id', 'N/A')}\n"
                        f"  Name: {lib.get('name', 'N/A')}\n"
                        f"  Description: {lib.get('description', 'N/A')}\n"
                        f"  Versions: {', '.join(lib.get('versions', [])[:3])}"
                    )
                return "\n\n".join(results)
        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code} - {e.response.text[:200]}"
        except Exception as e:
            return f"Error: {e}"

    @server.tool(
        title="Query documentation",
        annotations={"read_only_hint": True, "open_world_hint": True},
    )
    def query_docs(
        library_id: Annotated[str, Field(description="Context7 库 ID，格式如 '/org/project'")],
        query: Annotated[str, Field(description="文档查询内容，如 'How to set up authentication'")],
    ) -> str:
        """查询指定库的文档内容"""
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(
                    f"{BASE_URL}/ libraries/{library_id}/ docs",
                    params={"query": query},
                )
                resp.raise_for_status()
                data = resp.json()
                docs = data.get("docs", [])
                if not docs:
                    return f"未找到相关文档: {query}"
                results = []
                for doc in docs[:3]:
                    title = doc.get("title", "Untitled")
                    content = doc.get("content", "")[:2000]
                    results.append(f"## {title}\n\n{content}")
                return "\n\n---\n\n".join(results)
        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code} - {e.response.text[:200]}"
        except Exception as e:
            return f"Error: {e}"

    return server
