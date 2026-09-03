from .base import capability, BaseCapability, CapabilityOutput, CapabilityCategory


@capability(
    name="document_upload",
    description="上传文档到知识库，支持 PDF、TXT、MD、DOCX",
    category=CapabilityCategory.FILE,
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "文件路径",
            },
            "file_content": {
                "type": "string",
                "description": "文件内容（Base64 或纯文本）",
            },
            "file_name": {
                "type": "string",
                "description": "文件名",
            },
            "metadata": {
                "type": "object",
                "description": "附加元数据",
            },
        },
        "required": ["file_content", "file_name"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "document_id": {"type": "string"},
            "chunks": {"type": "integer"},
            "name": {"type": "string"},
        },
    },
    memory_mb=100,
    examples=["上传项目文档", "添加技术文档到知识库"],
)
class DocumentUploadCapability(BaseCapability):
    async def execute(self, params: dict) -> CapabilityOutput:
        from ..knowledge_base import knowledge_base
        import base64

        file_content = params.get("file_content", "")
        file_name = params.get("file_name", "document.txt")
        metadata = params.get("metadata", {})

        try:
            if file_name.endswith((".pdf", ".docx")):
                return CapabilityOutput(
                    success=False, error="PDF/DOCX 支持开发中，请使用纯文本文件"
                )

            if len(file_content) > 4 * 1024 * 1024:
                return CapabilityOutput(success=False, error="文件大小超过 4MB 限制")

            try:
                content = base64.b64decode(file_content).decode("utf-8")
            except:
                content = file_content

            if not content.strip():
                return CapabilityOutput(success=False, error="文档内容为空")

            doc_id = knowledge_base.add_document(
                name=file_name, content=content, metadata=metadata
            )

            doc_list = knowledge_base.list_documents()
            doc_info = next((d for d in doc_list if d["id"] == doc_id), None)

            return CapabilityOutput(
                success=True,
                data={
                    "document_id": doc_id,
                    "chunks": doc_info["chunks"] if doc_info else 0,
                    "name": file_name,
                    "size": len(content),
                },
                metadata={"file_name": file_name},
            )

        except RuntimeError as e:
            return CapabilityOutput(success=False, error=f"知识库未初始化: {str(e)}")
        except Exception as e:
            return CapabilityOutput(success=False, error=f"上传失败: {str(e)}")
