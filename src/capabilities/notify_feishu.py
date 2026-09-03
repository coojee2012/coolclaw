from .base import capability, BaseCapability, CapabilityOutput, CapabilityCategory


@capability(
    name="notify_feishu",
    description="通过飞书（ Lark ）群机器人发送通知",
    category=CapabilityCategory.NOTIFICATION,
    input_schema={
        "type": "object",
        "properties": {
            "webhook": {
                "type": "string",
                "description": "飞书群机器人的 Webhook URL",
            },
            "title": {
                "type": "string",
                "description": "消息标题",
            },
            "content": {
                "type": "string",
                "description": "消息内容（支持 Markdown）",
            },
        },
        "required": ["content"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "message": {"type": "string"},
        },
    },
    memory_mb=20,
    examples=["飞书群通知", "定时任务推送"],
)
class FeishuCapability(BaseCapability):
    async def execute(self, params: dict) -> CapabilityOutput:
        import httpx
        import json

        webhook = params.get("webhook") or params.get("secret_key")
        content = params.get("content", "")
        title = params.get("title", "通知")

        if not webhook:
            return CapabilityOutput(
                success=False, error="缺少 Webhook URL，请配置飞书 Webhook"
            )

        try:
            message = {
                "msg_type": "interactive",
                "card": {
                    "header": {
                        "title": {"tag": "plain_text", "content": title},
                        "template": "blue",
                    },
                    "elements": [{"tag": "markdown", "content": content}],
                },
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(webhook, json=message)
                result = response.json()

            if result.get("code") == 0 or result.get("StatusCode") == 0:
                return CapabilityOutput(
                    success=True,
                    data={"message": "发送成功"},
                    metadata={"title": title},
                )
            else:
                return CapabilityOutput(
                    success=False, error=f"发送失败: {result.get('msg', '未知错误')}"
                )

        except httpx.TimeoutException:
            return CapabilityOutput(success=False, error="请求超时")
        except Exception as e:
            return CapabilityOutput(success=False, error=f"发送失败: {str(e)}")
