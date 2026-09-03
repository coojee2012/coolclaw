from .base import capability, BaseCapability, CapabilityOutput, CapabilityCategory


@capability(
    name="rag_query",
    description="在知识库中检索相关信息并结合上下文回答问题",
    category=CapabilityCategory.AI,
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索问题",
            },
            "top_k": {
                "type": "integer",
                "description": "返回最相关的结果数量",
                "default": 3,
            },
            "include_sources": {
                "type": "boolean",
                "description": "是否包含信息来源",
                "default": True,
            },
        },
        "required": ["query"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "sources": {"type": "array"},
            "results_count": {"type": "integer"},
        },
    },
    memory_mb=50,
    examples=["查询项目文档", "检索技术细节", "查找配置说明"],
)
class RAGQueryCapability(BaseCapability):
    async def execute(self, params: dict) -> CapabilityOutput:
        from ..knowledge_base import knowledge_base

        query = params.get("query", "")
        top_k = params.get("top_k", 3)
        include_sources = params.get("include_sources", True)

        if not query:
            return CapabilityOutput(success=False, error="检索问题不能为空")

        try:
            results = knowledge_base.search(query, top_k=top_k)

            if not results:
                return CapabilityOutput(
                    success=True,
                    data={
                        "answer": "知识库中未找到相关信息。",
                        "sources": [],
                        "results_count": 0,
                    },
                    metadata={"query": query},
                )

            context = knowledge_base.get_context(query, max_length=1500)

            sources = []
            if include_sources:
                for r in results:
                    sources.append(
                        {
                            "content": r.content[:200] + "..."
                            if len(r.content) > 200
                            else r.content,
                            "source": r.source,
                            "relevance": f"{r.score:.2f}",
                        }
                    )

            return CapabilityOutput(
                success=True,
                data={
                    "context": context,
                    "answer": f"在知识库中找到 {len(results)} 条相关信息。",
                    "sources": sources,
                    "results_count": len(results),
                },
                metadata={"query": query, "top_k": top_k},
            )

        except RuntimeError as e:
            return CapabilityOutput(success=False, error=f"知识库未初始化: {str(e)}")
        except Exception as e:
            return CapabilityOutput(success=False, error=f"检索失败: {str(e)}")
