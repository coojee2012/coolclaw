from .base import capability, BaseCapability, CapabilityOutput, CapabilityCategory


@capability(
    name="notify_email",
    description="发送邮件通知（支持 SMTP）",
    category=CapabilityCategory.NOTIFICATION,
    input_schema={
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "description": "收件人邮箱",
            },
            "subject": {
                "type": "string",
                "description": "邮件主题",
            },
            "body": {
                "type": "string",
                "description": "邮件正文（支持 HTML）",
            },
            "cc": {
                "type": "string",
                "description": "抄送邮箱（可选）",
            },
        },
        "required": ["to", "subject", "body"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "message": {"type": "string"},
        },
    },
    memory_mb=20,
    examples=["发送邮件报告", "定时邮件通知"],
)
class EmailCapability(BaseCapability):
    async def execute(self, params: dict) -> CapabilityOutput:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        import asyncio

        to_email = params.get("to", "")
        subject = params.get("subject", "通知")
        body = params.get("body", "")
        cc_email = params.get("cc", "")

        if not to_email:
            return CapabilityOutput(success=False, error="缺少收件人邮箱")

        try:
            from ..config import get_config

            config = get_config()
            smtp_host = getattr(config.smtp, "host", "smtp.gmail.com")
            smtp_port = getattr(config.smtp, "port", 587)
            smtp_user = getattr(config.smtp, "user", "")
            smtp_password = getattr(config.smtp, "password", "")
            from_email = getattr(config.smtp, "from_email", smtp_user)

            if not smtp_user or not smtp_password:
                return CapabilityOutput(
                    success=False, error="SMTP 未配置，请在设置中配置邮箱服务"
                )

            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = from_email
            msg["To"] = to_email

            if cc_email:
                msg["Cc"] = cc_email

            html_part = MIMEText(body, "html", "utf-8")
            text_part = MIMEText(
                body.replace("<br>", "\n").replace("<p>", "\n").strip(),
                "plain",
                "utf-8",
            )
            msg.attach(text_part)
            msg.attach(html_part)

            def send_sync():
                with smtplib.SMTP(smtp_host, smtp_port) as server:
                    server.starttls()
                    server.login(smtp_user, smtp_password)
                    to_list = [to_email]
                    if cc_email:
                        to_list.extend(cc_email.split(","))
                    server.sendmail(from_email, to_list, msg.as_string())

            await asyncio.get_event_loop().run_in_executor(None, send_sync)

            return CapabilityOutput(
                success=True,
                data={"message": "发送成功", "to": to_email},
                metadata={"subject": subject},
            )

        except smtplib.SMTPAuthenticationError:
            return CapabilityOutput(
                success=False, error="邮箱认证失败，请检查用户名和密码"
            )
        except smtplib.SMTPException as e:
            return CapabilityOutput(success=False, error=f"SMTP 错误: {str(e)}")
        except Exception as e:
            return CapabilityOutput(success=False, error=f"发送失败: {str(e)}")
