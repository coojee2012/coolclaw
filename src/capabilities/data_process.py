from .base import capability, BaseCapability, CapabilityOutput, CapabilityCategory


@capability(
    name="web_fetch",
    description="获取网页内容，支持静态页面解析",
    category=CapabilityCategory.NETWORK,
    input_schema={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "网页 URL"},
            "selector": {
                "type": "string",
                "description": "CSS 选择器 (可选，用于提取特定内容)",
            },
            "max_length": {
                "type": "integer",
                "description": "最大内容长度",
                "default": 5000,
            },
        },
        "required": ["url"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "content": {"type": "string"},
            "links": {"type": "array", "items": {"type": "string"}},
            "images": {"type": "array", "items": {"type": "string"}},
        },
    },
    memory_mb=50,
    examples=["获取新闻文章内容", "提取网页中的链接列表"],
)
class WebFetchCapability(BaseCapability):
    async def execute(self, params: dict) -> CapabilityOutput:
        try:
            import requests
            from bs4 import BeautifulSoup

            url = params.get("url", "")
            selector = params.get("selector")
            max_length = params.get("max_length", 5000)

            if not url:
                return CapabilityOutput(success=False, error="URL 不能为空")

            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.encoding = response.apparent_encoding or "utf-8"

            soup = BeautifulSoup(response.text, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            title = soup.title.string if soup.title else ""

            if selector:
                elements = soup.select(selector)
                content = "\n".join(elem.get_text(strip=True) for elem in elements)
            else:
                content = soup.get_text(separator="\n", strip=True)

            content = content[:max_length]

            links = []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("http"):
                    links.append(href)

            images = []
            for img in soup.find_all("img", src=True):
                src = img["src"]
                if src.startswith("http"):
                    images.append(src)

            return CapabilityOutput(
                success=True,
                data={
                    "url": url,
                    "title": title,
                    "content": content,
                    "links": links[:20],
                    "images": images[:10],
                    "fetched_at": __import__("datetime").datetime.now().isoformat(),
                },
            )

        except ImportError:
            return CapabilityOutput(
                success=False,
                error="需要安装 requests 和 beautifulsoup4: pip install requests beautifulsoup4",
            )
        except Exception as e:
            return CapabilityOutput(success=False, error=f"获取失败: {str(e)}")


@capability(
    name="file_write",
    description="写入文本内容到文件",
    category=CapabilityCategory.FILE,
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径"},
            "content": {"type": "string", "description": "文件内容"},
            "append": {
                "type": "boolean",
                "description": "是否追加模式",
                "default": False,
            },
        },
        "required": ["path", "content"],
    },
    output_schema={
        "type": "object",
        "properties": {"written": {"type": "boolean"}, "path": {"type": "string"}},
    },
    memory_mb=5,
    examples=["保存任务结果到文件", "追加日志到文件"],
)
class FileWriteCapability(BaseCapability):
    async def execute(self, params: dict) -> CapabilityOutput:
        try:
            import os
            from pathlib import Path

            file_path = params.get("path", "")
            content = params.get("content", "")
            append = params.get("append", False)

            if not file_path or not content:
                return CapabilityOutput(success=False, error="文件路径和内容不能为空")

            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            mode = "a" if append else "w"
            with open(file_path, mode, encoding="utf-8") as f:
                f.write(content)

            return CapabilityOutput(
                success=True,
                data={"written": True, "path": file_path, "size": len(content)},
            )

        except Exception as e:
            return CapabilityOutput(success=False, error=f"写入失败: {str(e)}")
