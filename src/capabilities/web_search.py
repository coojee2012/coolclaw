from .base import capability, BaseCapability, CapabilityOutput, CapabilityCategory


@capability(
    name="web_search",
    description="使用 DuckDuckGo 搜索网络，返回标题、链接和摘要",
    category=CapabilityCategory.NETWORK,
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词",
                "example": "科技新闻",
            },
            "max_results": {
                "type": "integer",
                "description": "最大结果数",
                "default": 10,
            },
        },
        "required": ["query"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "snippet": {"type": "string"},
                    },
                },
            }
        },
    },
    memory_mb=30,
    examples=["搜索 Python 教程", "查找最新AI新闻", "搜索科技动态"],
)
class WebSearchCapability(BaseCapability):
    async def execute(self, params: dict) -> CapabilityOutput:
        try:
            from ddgs import DDGS

            query = params.get("query", "")
            max_results = params.get("max_results", 10)
            if isinstance(max_results, str):
                max_results = int(max_results)

            if not query:
                return CapabilityOutput(success=False, error="搜索关键词不能为空")

            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(
                        {
                            "title": r.get("title", ""),
                            "url": r.get("href", ""),
                            "snippet": r.get("body", ""),
                        }
                    )

            return CapabilityOutput(
                success=True,
                data={"results": results, "count": len(results)},
                metadata={"query": query},
            )
        except ImportError:
            return CapabilityOutput(
                success=False,
                error="需要安装 ddgs: pip install ddgs",
            )
        except Exception as e:
            return CapabilityOutput(success=False, error=f"搜索失败: {str(e)}")
