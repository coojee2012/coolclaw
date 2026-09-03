from .base import capability, BaseCapability, CapabilityOutput, CapabilityCategory


@capability(
    name="ai_summarize",
    description="使用本地 AI 模型对文本进行摘要生成",
    category=CapabilityCategory.AI,
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "要摘要的文本内容"},
            "length": {
                "type": "string",
                "description": "摘要长度: short/medium/long",
                "default": "medium",
            },
            "language": {
                "type": "string",
                "description": "输出语言",
                "default": "中文",
            },
        },
        "required": ["text"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "original_length": {"type": "integer"},
            "summary_length": {"type": "integer"},
        },
    },
    memory_mb=0,
    examples=["摘要一篇新闻文章", "总结会议记录要点"],
)
class SummarizeCapability(BaseCapability):
    async def execute(self, params: dict) -> CapabilityOutput:
        try:
            from ..dispatcher import get_dispatcher_model
            from ..local_llm import LocalLLM
            import os

            text = params.get("text", "")
            length = params.get("length", "medium")
            language = params.get("language", "中文")

            if not text:
                return CapabilityOutput(success=False, error="要摘要的文本不能为空")

            length_instruction = {
                "short": "用1-2句话概括要点",
                "medium": "用3-5句话概括主要内容",
                "long": "详细概括所有重要内容",
            }.get(length, "用3-5句话概括主要内容")

            prompt = f"""请用{language}概括以下文本的主要内容。

要求：{length_instruction}

文本内容：
{text[:3000]}

摘要："""

            model_spec = get_dispatcher_model()
            llm = LocalLLM(
                model_path=model_spec.path,
                n_ctx=2048,
                n_gpu_layers=-1,
                flash_attn=True,
            )

            result = llm.complete(
                prompt=prompt,
                max_tokens=500,
                temperature=0.3,
            )

            summary = result.content.strip()

            return CapabilityOutput(
                success=True,
                data={
                    "summary": summary,
                    "original_length": len(text),
                    "summary_length": len(summary),
                    "compression_ratio": f"{len(summary) / len(text) * 100:.1f}%",
                },
            )

        except Exception as e:
            return CapabilityOutput(success=False, error=f"摘要生成失败: {str(e)}")


@capability(
    name="ai_rewrite",
    description="使用本地 AI 模型改写/润色文本",
    category=CapabilityCategory.AI,
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "要改写的文本"},
            "style": {
                "type": "string",
                "description": "改写风格: formal/casual/news/marketing",
                "default": "formal",
            },
        },
        "required": ["text"],
    },
    output_schema={"type": "object", "properties": {"rewritten": {"type": "string"}}},
    memory_mb=0,
    examples=["将口语化文本改写成正式文本", "将文本改写成新闻风格"],
)
class RewriteCapability(BaseCapability):
    async def execute(self, params: dict) -> CapabilityOutput:
        try:
            from ..dispatcher import get_dispatcher_model
            from ..local_llm import LocalLLM

            text = params.get("text", "")
            style = params.get("style", "formal")

            if not text:
                return CapabilityOutput(success=False, error="要改写的文本不能为空")

            style_map = {
                "formal": "正式、专业、书面语",
                "casual": "轻松、随意、口语化",
                "news": "新闻报道风格，客观准确",
                "marketing": "营销文案风格，吸引眼球",
            }

            style_instruction = style_map.get(style, "正式、专业")

            prompt = f"""请将以下文本改写成{style_instruction}的风格。

原文：
{text[:2000]}

改写后的文本："""

            model_spec = get_dispatcher_model()
            llm = LocalLLM(
                model_path=model_spec.path,
                n_ctx=2048,
                n_gpu_layers=-1,
                flash_attn=True,
            )

            result = llm.complete(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.5,
            )

            return CapabilityOutput(
                success=True, data={"rewritten": result.content.strip()}
            )

        except Exception as e:
            return CapabilityOutput(success=False, error=f"改写失败: {str(e)}")
