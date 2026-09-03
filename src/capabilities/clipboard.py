from .base import capability, BaseCapability, CapabilityOutput, CapabilityCategory


@capability(
    name="clipboard_copy",
    description="复制文本内容到系统剪贴板",
    category=CapabilityCategory.SYSTEM,
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "要复制到剪贴板的文本"}
        },
        "required": ["text"],
    },
    output_schema={
        "type": "object",
        "properties": {"copied": {"type": "boolean"}, "length": {"type": "integer"}},
    },
    memory_mb=5,
    examples=["复制生成的文案", "保存摘要到剪贴板"],
)
class ClipboardCopyCapability(BaseCapability):
    async def execute(self, params: dict) -> CapabilityOutput:
        try:
            import pyperclip

            text = params.get("text", "")

            if not text:
                return CapabilityOutput(success=False, error="复制的文本不能为空")

            pyperclip.copy(text)

            return CapabilityOutput(
                success=True,
                data={"copied": True, "length": len(text)},
                metadata={"chars": len(text)},
            )
        except ImportError:
            return CapabilityOutput(
                success=False, error="需要安装 pyperclip: pip install pyperclip"
            )
        except Exception as e:
            return CapabilityOutput(success=False, error=f"复制失败: {str(e)}")


@capability(
    name="clipboard_paste",
    description="从系统剪贴板读取文本",
    category=CapabilityCategory.SYSTEM,
    input_schema={"type": "object", "properties": {}},
    output_schema={"type": "object", "properties": {"text": {"type": "string"}}},
    memory_mb=5,
    examples=["读取剪贴板内容", "获取用户粘贴的文本"],
)
class ClipboardPasteCapability(BaseCapability):
    async def execute(self, params: dict) -> CapabilityOutput:
        try:
            import pyperclip

            text = pyperclip.paste()

            return CapabilityOutput(
                success=True,
                data={"text": text, "length": len(text)},
                metadata={"chars": len(text)},
            )
        except ImportError:
            return CapabilityOutput(
                success=False, error="需要安装 pyperclip: pip install pyperclip"
            )
        except Exception as e:
            return CapabilityOutput(success=False, error=f"读取剪贴板失败: {str(e)}")
