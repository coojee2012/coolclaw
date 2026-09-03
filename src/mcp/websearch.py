"""
Websearch MCP Server — 网络搜索

提供网络搜索和网页内容获取能力。
"""

import json
import subprocess
from typing import Annotated

from pydantic import Field

from mcp.server import MCPServer


def create_websearch_server() -> MCPServer:
    """创建 Websearch MCP Server"""

    server = MCPServer("websearch")

    def _curl_fetch(url, timeout=15):
        """使用 curl 获取网页内容"""
        try:
            result = subprocess.run(
                ["curl", "-sL", "--max-time", str(timeout), "--noproxy", "*", url],
                capture_output=True,
                text=True,
                timeout=timeout + 5,
            )
            if result.returncode != 0:
                return "Error: curl failed - " + result.stderr[:200]
            return result.stdout[:50000]
        except subprocess.TimeoutExpired:
            return "Error: fetch timed out"
        except Exception as e:
            return "Error: " + str(e)

    @server.tool(
        title="Web search",
        annotations={"read_only_hint": True, "open_world_hint": True},
    )
    def web_search(
        query: Annotated[str, Field(description="搜索查询")],
        num_results: Annotated[int, Field(description="返回结果数量", ge=1, le=10)] = 5,
    ) -> str:
        """使用搜索引擎搜索网络内容"""
        try:
            import httpx
            with httpx.Client(timeout=15, follow_redirects=True) as client:
                resp = client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                text = resp.text
                results = []
                import re
                for m in re.finditer(
                    r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
                    r'class="result__snippet"[^>]*>(.*?)</span>',
                    text,
                    re.DOTALL,
                ):
                    url = m.group(1)
                    title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
                    snippet = re.sub(r"<[^>]+>", "", m.group(3)).strip()
                    results.append(f"**{title}**\n{url}\n{snippet}")
                    if len(results) >= num_results:
                        break
                if not results:
                    return "未找到搜索结果: " + query
                return "\n\n---\n\n".join(results)
        except ImportError:
            return "Error: httpx not installed. Run: pip install httpx"
        except Exception as e:
            return "Error: " + str(e)

    @server.tool(
        title="Fetch URL content",
        annotations={"read_only_hint": True, "open_world_hint": True},
    )
    def fetch_url(
        url: Annotated[str, Field(description="要获取内容的 URL")],
        max_chars: Annotated[int, Field(description="最大返回字符数", ge=100, le=100000)] = 10000,
    ) -> str:
        """获取指定 URL 的内容"""
        try:
            import httpx
            with httpx.Client(timeout=15, follow_redirects=True) as client:
                resp = client.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                content = resp.text[:max_chars]
                return f"Status: {resp.status_code}\nContent-Type: {resp.headers.get('content-type', 'N/A')}\n\n{content}"
        except ImportError:
            return "Error: httpx not installed. Run: pip install httpx"
        except Exception as e:
            return "Error: " + str(e)

    return server
