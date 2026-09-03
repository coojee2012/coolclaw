from .base import (
    CapabilityRegistry,
    Capability,
    CapabilityCategory,
    BaseCapability,
    CapabilityOutput,
    capability,
)
from .web_search import WebSearchCapability
from .clipboard import ClipboardCopyCapability, ClipboardPasteCapability
from .webhook import WebhookCapability
from .file_watch import FileWatchCapability
from .notify import TelegramCapability, EmailCapability
from .notify_dingtalk import DingTalkCapability
from .notify_feishu import FeishuCapability
from .notify_email import EmailCapability
from .ai_process import SummarizeCapability, RewriteCapability
from .data_process import WebFetchCapability, FileWriteCapability
from .rag_query import RAGQueryCapability
from .document_upload import DocumentUploadCapability


def register_all():
    """Import all capability modules to trigger auto-registration via decorators."""
    from .web_search import WebSearchCapability  # noqa: F401
    from .clipboard import ClipboardCopyCapability, ClipboardPasteCapability  # noqa: F401
    from .webhook import WebhookCapability  # noqa: F401
    from .file_watch import FileWatchCapability  # noqa: F401
    from .notify import TelegramCapability  # noqa: F401
    from .notify_dingtalk import DingTalkCapability  # noqa: F401
    from .notify_feishu import FeishuCapability  # noqa: F401
    from .ai_process import SummarizeCapability, RewriteCapability  # noqa: F401
    from .data_process import WebFetchCapability, FileWriteCapability  # noqa: F401
    from .rag_query import RAGQueryCapability  # noqa: F401
    from .document_upload import DocumentUploadCapability  # noqa: F401


__all__ = [
    "CapabilityRegistry",
    "Capability",
    "CapabilityCategory",
    "BaseCapability",
    "CapabilityOutput",
    "capability",
    "register_all",
]
