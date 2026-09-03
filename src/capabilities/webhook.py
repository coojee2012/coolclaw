from .base import capability, BaseCapability, CapabilityOutput, CapabilityCategory


@capability(
    name="notify_webhook",
    description="发送 HTTP POST 请求到指定 URL (支持 Slack/Discord webhook)",
    category=CapabilityCategory.NOTIFICATION,
    input_schema={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Webhook URL"},
            "message": {"type": "string", "description": "发送的消息内容"},
            "title": {"type": "string", "description": "可选标题"},
        },
        "required": ["url", "message"],
    },
    output_schema={
        "type": "object",
        "properties": {"sent": {"type": "boolean"}, "status_code": {"type": "integer"}},
    },
    memory_mb=10,
    examples=["发送通知到 Slack", "发送消息到 Discord"],
)
class WebhookCapability(BaseCapability):
    async def execute(self, params: dict) -> CapabilityOutput:
        try:
            import httpx
            import json

            url = params.get("url", "")
            message = params.get("message", "")
            title = params.get("title", "")

            if not url or not message:
                return CapabilityOutput(success=False, error="URL 和消息内容不能为空")

            payload = {"text": message}
            if title:
                payload = {
                    "text": message,
                    "blocks": [
                        {
                            "type": "header",
                            "text": {"type": "plain_text", "text": title},
                        }
                    ],
                }

            response = httpx.post(url, json=payload, timeout=10)

            return CapabilityOutput(
                success=response.status_code in (200, 201, 204),
                data={"sent": True, "status_code": response.status_code},
                metadata={"url": url[:50]},
            )
        except ImportError:
            return CapabilityOutput(
                success=False, error="需要安装 httpx: pip install httpx"
            )
        except Exception as e:
            return CapabilityOutput(success=False, error=f"发送失败: {str(e)}")
