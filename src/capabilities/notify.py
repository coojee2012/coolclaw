from .base import capability, BaseCapability, CapabilityOutput, CapabilityCategory


@capability(
    name="notify_telegram",
    description="通过 Telegram Bot 发送消息通知",
    category=CapabilityCategory.NOTIFICATION,
    input_schema={
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "要发送的消息内容"},
            "chat_id": {
                "type": "string",
                "description": "Telegram Chat ID (用户的chat_id)",
            },
        },
        "required": ["message"],
    },
    output_schema={
        "type": "object",
        "properties": {"sent": {"type": "boolean"}, "message_id": {"type": "integer"}},
    },
    memory_mb=10,
    examples=["发送任务完成通知", "发送告警消息"],
)
class TelegramCapability(BaseCapability):
    async def execute(self, params: dict) -> CapabilityOutput:
        try:
            import httpx
            import os

            from ..storage import secrets

            bot_token = secrets.get("telegram_bot_token")
            if not bot_token:
                bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")

            if not bot_token:
                return CapabilityOutput(
                    success=False,
                    error="未配置 Telegram Bot Token。请先在 设置 → 敏感数据 中配置 telegram_bot_token",
                )

            message = params.get("message", "")
            chat_id = params.get("chat_id") or secrets.get("telegram_chat_id")

            if not chat_id:
                return CapabilityOutput(
                    success=False,
                    error="未配置 Telegram Chat ID。请先在 设置 → 敏感数据 中配置 telegram_chat_id",
                )

            if not message:
                return CapabilityOutput(success=False, error="消息内容不能为空")

            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}

            response = httpx.post(url, json=payload, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return CapabilityOutput(
                    success=True,
                    data={
                        "sent": True,
                        "message_id": data.get("result", {}).get("message_id"),
                        "chat_id": chat_id,
                    },
                )
            else:
                return CapabilityOutput(
                    success=False, error=f"发送失败: {response.text}"
                )

        except ImportError:
            return CapabilityOutput(
                success=False, error="需要安装 httpx: pip install httpx"
            )
        except Exception as e:
            return CapabilityOutput(success=False, error=f"发送失败: {str(e)}")


@capability(
    name="notify_email",
    description="发送电子邮件通知",
    category=CapabilityCategory.NOTIFICATION,
    input_schema={
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "收件人邮箱"},
            "subject": {"type": "string", "description": "邮件主题"},
            "body": {"type": "string", "description": "邮件正文"},
        },
        "required": ["to", "subject", "body"],
    },
    output_schema={"type": "object", "properties": {"sent": {"type": "boolean"}}},
    memory_mb=20,
    examples=["发送邮件报告", "发送告警邮件"],
)
class EmailCapability(BaseCapability):
    async def execute(self, params: dict) -> CapabilityOutput:
        try:
            import smtplib
            import os
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
            smtp_port = int(os.environ.get("SMTP_PORT", "587"))
            smtp_user = os.environ.get("SMTP_USER")
            smtp_password = os.environ.get("SMTP_PASSWORD")

            from ..storage import secrets

            if not smtp_user:
                smtp_user = secrets.get("smtp_user")
            if not smtp_password:
                smtp_password = secrets.get("smtp_password")

            if not smtp_user or not smtp_password:
                return CapabilityOutput(
                    success=False,
                    error="未配置邮件服务器。请设置环境变量 SMTP_USER, SMTP_PASSWORD 或在敏感数据中配置",
                )

            to_email = params.get("to", "")
            subject = params.get("subject", "")
            body = params.get("body", "")

            msg = MIMEMultipart()
            msg["From"] = smtp_user
            msg["To"] = to_email
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "html"))

            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)

            return CapabilityOutput(success=True, data={"sent": True, "to": to_email})

        except Exception as e:
            return CapabilityOutput(success=False, error=f"发送失败: {str(e)}")
