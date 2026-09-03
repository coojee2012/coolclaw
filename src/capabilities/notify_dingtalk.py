from .base import capability, BaseCapability, CapabilityOutput, CapabilityCategory


@capability(
    name="notify_dingtalk",
    description="通过钉钉群机器人发送通知消息",
    category=CapabilityCategory.NOTIFICATION,
    input_schema={
        "type": "object",
        "properties": {
            "webhook": {
                "type": "string",
                "description": "钉钉群机器人的 Webhook URL",
            },
            "secret": {
                "type": "string",
                "description": "加签密钥（可选，不填则不使用加签）",
            },
            "title": {
                "type": "string",
                "description": "消息标题",
            },
            "content": {
                "type": "string",
                "description": "消息内容（支持 Markdown）",
            },
            "at_mobiles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "被 @ 的手机号列表",
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
    examples=["发送任务完成通知", "定时推送新闻摘要"],
)
class DingTalkCapability(BaseCapability):
    async def execute(self, params: dict) -> CapabilityOutput:
        import hashlib
        import hmac
        import base64
        import time
        import urllib.parse
        import json
        import httpx

        webhook = params.get("webhook") or params.get(
            "secret_key"
        )  # 支持从 secret 获取
        content = params.get("content", "")
        title = params.get("title", "通知")
        at_mobiles = params.get("at_mobiles", [])
        secret = params.get("secret", "")

        if not webhook:
            return CapabilityOutput(
                success=False, error="缺少 Webhook URL，请配置钉钉 Webhook"
            )

        try:
            timestamp = str(round(time.time() * 1000))
            sign = ""

            if secret:
                secret_enc = secret.encode("utf-8")
                string_to_sign = f"{timestamp}\n{secret}"
                string_to_sign_enc = string_to_sign.encode("utf-8")
                hmac_code = hmac.new(
                    secret_enc, string_to_sign_enc, digestmod=hashlib.sha256
                ).digest()
                sign = urllib.parse.quote_plus(
                    base64.b64encode(hmac_code).decode("utf-8")
                )
                webhook_url = f"{webhook}&timestamp={timestamp}&sign={sign}"
            else:
                webhook_url = webhook

            message = {
                "msgtype": "markdown",
                "markdown": {"title": title, "text": content},
            }

            if at_mobiles:
                message["at"] = {"atMobiles": at_mobiles, "isAtAll": False}

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(webhook_url, json=message)
                result = response.json()

            if result.get("errcode") == 0:
                return CapabilityOutput(
                    success=True,
                    data={"message": "发送成功"},
                    metadata={"title": title},
                )
            else:
                return CapabilityOutput(
                    success=False, error=f"发送失败: {result.get('errmsg', '未知错误')}"
                )

        except httpx.TimeoutException:
            return CapabilityOutput(success=False, error="请求超时")
        except Exception as e:
            return CapabilityOutput(success=False, error=f"发送失败: {str(e)}")
