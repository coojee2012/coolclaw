from typing import Optional
from enum import Enum
from pydantic import BaseModel


class ModelRole(str, Enum):
    DISPATCHER = "dispatcher"
    CODING = "coding"
    GENERAL = "general"
    REASONING = "reasoning"
    FAST = "fast"
    OCR = "ocr"
    VISION = "vision"
    DOCUMENTS = "documents"


class ModelCapability(BaseModel):
    input_format: str
    output_format: str
    capabilities: list[str]
    limitations: list[str]
    best_for: list[str]


class ModelSpec(BaseModel):
    key: str
    path: str
    role: ModelRole
    name: str
    description: str
    strengths: list[str]
    capabilities: ModelCapability
    context_window: int = 131072
    speed: str = "medium"


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "qwen2.5-3b": ModelSpec(
        key="qwen2.5-3b",
        path="/Volumes/LynnData/myclaw_models/qwen2.5-3b-instruct-q4_k_m.gguf",
        role=ModelRole.DISPATCHER,
        name="Qwen2.5-3B",
        description="智能调度中心，理解意图并选择最佳处理方式",
        strengths=["任务分类", "意图识别", "快速响应", "简单问答"],
        capabilities=ModelCapability(
            input_format="自然语言请求",
            output_format="JSON决策 + 可选直接回答",
            capabilities=["任务分析", "意图识别", "模型选择", "简单问答", "多轮对话"],
            limitations=["不擅长复杂代码生成", "不擅长长文本处理"],
            best_for=["日常对话", "任务分类", "系统协调"],
        ),
        speed="fast",
    ),
    "qwen2.5-coder-7b": ModelSpec(
        key="qwen2.5-coder-7b",
        path="/Volumes/LynnData/myclaw_models/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        role=ModelRole.CODING,
        name="Qwen2.5-Coder-7B",
        description="代码专家，擅长代码生成、debug、优化",
        strengths=[
            "Python",
            "JavaScript",
            "SQL",
            "代码审查",
            "Bug修复",
            "重构",
            "算法",
        ],
        capabilities=ModelCapability(
            input_format="问题描述 / 代码片段 / 错误信息",
            output_format="代码 + 解释（中文）",
            capabilities=[
                "代码生成",
                "代码审查",
                "Bug定位",
                "性能优化",
                "代码重构",
                "算法解释",
                "SQL查询",
                "API设计",
            ],
            limitations=["不处理图片", "不处理PDF文档"],
            best_for=["写代码", "debug", "代码优化", "技术问题"],
        ),
        speed="medium",
    ),
    # qwen2.5-7b model not downloaded yet - removed to avoid errors
    # "qwen2.5-7b": ModelSpec(
    #     key="qwen2.5-7b",
    #     path="/Volumes/LynnData/myclaw_models/qwen2.5-7b-instruct-q4_k_m.gguf",
    #     role=ModelRole.GENERAL,
    #     name="Qwen2.5-7B",
    #     description="通用对话，处理日常对话和通用任务",
    #     strengths=["闲聊", "写作", "总结", "解释概念"],
    #     capabilities=ModelCapability(
    #         input_format="自然语言",
    #         output_format="自然语言回答",
    #         capabilities=["闲聊", "写作", "总结", "解释", "问答", "创意"],
    #         limitations=["不擅长专业代码", "不擅长图片/文档处理"],
    #         best_for=["日常对话", "写作", "总结", "解释概念"],
    #     ),
    #     speed="medium",
    # ),
    "lightonocr-2-1b": ModelSpec(
        key="lightonocr-2-1b",
        path="/Volumes/LynnData/myclaw_models/LightOnOCR-2-1B-Q4_K_M.gguf",
        role=ModelRole.OCR,
        name="LightOnOCR-2-1B",
        description="OCR专家，从扫描件、图片、PDF提取文字",
        strengths=["PDF扫描件", "图片文字识别", "手写体识别", "多语言OCR"],
        capabilities=ModelCapability(
            input_format="图片文件路径或base64",
            output_format="提取的文本",
            capabilities=["图片文字识别", "PDF文字提取", "手写体识别", "多语言支持"],
            limitations=["只返回纯文本", "不保留格式"],
            best_for=["扫描件转文字", "图片文字提取", "PDF内容获取"],
        ),
        context_window=8192,
        speed="fast",
    ),
    "smolvlm-256m": ModelSpec(
        key="smolvlm-256m",
        path="/Volumes/LynnData/myclaw_models/SmolVLM-256M-Instruct-Q4_K_M.gguf",
        role=ModelRole.VISION,
        name="SmolVLM-256M",
        description="视觉理解专家，轻量级图片分析",
        strengths=["图片描述", "图表理解", "简单图像问答"],
        capabilities=ModelCapability(
            input_format="图片 + 问题",
            output_format="文字描述/回答",
            capabilities=["图片描述", "图表理解", "物体识别", "简单问答"],
            limitations=["不擅长复杂场景", "不处理文档格式"],
            best_for=["看图说话", "图表分析", "简单图片问答"],
        ),
        speed="fast",
    ),
    "minicpm-v-4": ModelSpec(
        key="minicpm-v-4",
        path="/Volumes/LynnData/myclaw_models/MiniCPM-V-4-q4_k_m.gguf",
        role=ModelRole.VISION,
        name="MiniCPM-V-4",
        description="视觉理解专家，高质量图片分析(需mmproj文件)",
        strengths=["图片描述", "PDF解析", "复杂图像理解", "OCR"],
        capabilities=ModelCapability(
            input_format="图片/PDF + 问题",
            output_format="详细描述/分析",
            capabilities=["图片描述", "PDF解析", "复杂图像理解", "OCR", "表格识别"],
            limitations=["需要mmproj文件", "占用内存较大"],
            best_for=["复杂图片分析", "PDF内容理解", "图表解读"],
        ),
        context_window=4096,
        speed="medium",
    ),
    "granite-docling-258m": ModelSpec(
        key="granite-docling-258m",
        path="/Volumes/LynnData/myclaw_models/granite-docling-258M-q4_k_m.gguf",
        role=ModelRole.DOCUMENTS,
        name="Granite-Docling-258M",
        description="文档处理专家，将文档转换为结构化格式",
        strengths=["PDF转换", "表格提取", "文档结构化", "Docling格式"],
        capabilities=ModelCapability(
            input_format="文档图片/PDF",
            output_format="结构化文档（Docling格式/Markdown）",
            capabilities=["文档结构化", "表格提取", "布局分析", "格式转换"],
            limitations=["需要配合文档处理流程"],
            best_for=["PDF转结构化", "表格提取", "文档分析"],
        ),
        speed="fast",
    ),
    "smoldocling-256m": ModelSpec(
        key="smoldocling-256m",
        path="/Volumes/LynnData/myclaw_models/SmolDocling-256M-q4_k_m.gguf",
        role=ModelRole.DOCUMENTS,
        name="SmolDocling-256M",
        description="文档转换专家，极轻量级文档处理",
        strengths=["文档转换", "页面布局分析", "格式转换"],
        capabilities=ModelCapability(
            input_format="文档图片",
            output_format="DocTags/结构化内容",
            capabilities=["文档转换", "布局分析", "格式转换"],
            limitations=["极轻量，功能有限"],
            best_for=["轻量文档转换", "页面布局分析"],
        ),
        speed="fast",
    ),
}


def get_model_by_role(role: ModelRole) -> Optional[ModelSpec]:
    import os

    for model in MODEL_REGISTRY.values():
        if model.role == role:
            if os.path.exists(model.path):
                return model
    return None


def get_available_model(key: str) -> Optional[ModelSpec]:
    import os

    model = MODEL_REGISTRY.get(key)
    if model and os.path.exists(model.path):
        return model
    return None


def get_fallback_model() -> ModelSpec:
    available = [
        ("qwen2.5-coder-7b", "代码专家"),
        ("qwen2.5-3b", "调度中心"),
    ]
    import os

    for key, name in available:
        model = MODEL_REGISTRY.get(key)
        if model and os.path.exists(model.path):
            return model
    return get_dispatcher_model()


def get_dispatcher_model() -> ModelSpec:
    return MODEL_REGISTRY.get("qwen2.5-3b") or ModelSpec(
        key="qwen2.5-3b",
        path="/Volumes/LynnData/myclaw_models/qwen2.5-3b-instruct-q4_k_m.gguf",
        role=ModelRole.DISPATCHER,
        name="Qwen2.5-3B",
        description="智能调度中心",
        strengths=["任务分类", "意图识别"],
        capabilities=ModelCapability(
            input_format="自然语言请求",
            output_format="JSON决策",
            capabilities=["任务分析", "意图识别", "模型选择"],
            limitations=["不擅长复杂代码"],
            best_for=["日常对话", "任务分类"],
        ),
    )


def get_specialist_for_task(task_description: str) -> ModelSpec:
    task_lower = task_description.lower()

    coding_keywords = [
        "code",
        "function",
        "bug",
        "debug",
        "python",
        "javascript",
        "sql",
        "refactor",
        "optimize",
        "implement",
        "class",
        "api",
        "script",
        "programming",
        "algorithm",
        "syntax",
        "compile",
        "代码",
        "编程",
    ]

    reasoning_keywords = [
        "why",
        "think",
        "reason",
        "explain",
        "analyze",
        "solve",
        "calculate",
        "math",
        "logic",
        "prove",
        "为什么",
        "推理",
        "分析",
    ]

    ocr_keywords = [
        "ocr",
        "scan",
        "pdf",
        "图片文字",
        "扫描件",
        "手写",
        "提取文字",
        "recognize text",
        "from image",
        "from pdf",
        "截图",
        "图片转文字",
    ]

    vision_keywords = [
        "image",
        "picture",
        "photo",
        "图",
        "图片",
        "照片",
        "描述图片",
        "what is in",
        "图片里有什么",
        "看懂图片",
        "图表",
    ]

    doc_keywords = [
        "document",
        "文档",
        "pdf转换",
        "表格提取",
        "结构化",
        "extract table",
        "convert doc",
        "表格",
        "excel",
        "word",
    ]

    fast_keywords = [
        "what is",
        "quick",
        "brief",
        "simple",
        "short",
        "summarize",
        "简单",
        "是什么",
    ]

    coding_score = sum(1 for kw in coding_keywords if kw in task_lower)
    reasoning_score = sum(1 for kw in reasoning_keywords if kw in task_lower)
    ocr_score = sum(1 for kw in ocr_keywords if kw in task_lower)
    vision_score = sum(1 for kw in vision_keywords if kw in task_lower)
    doc_score = sum(1 for kw in doc_keywords if kw in task_lower)
    fast_score = sum(1 for kw in fast_keywords if kw in task_lower)

    if ocr_score >= 1:
        return get_available_model("lightonocr-2-1b") or get_fallback_model()
    elif vision_score >= 1:
        return get_available_model("smolvlm-256m") or get_fallback_model()
    elif doc_score >= 1:
        return get_available_model("granite-docling-258m") or get_fallback_model()
    elif coding_score >= 2:
        return get_available_model("qwen2.5-coder-7b") or get_fallback_model()
    elif reasoning_score >= 2:
        return get_available_model("qwen2.5-coder-7b") or get_fallback_model()
    elif fast_score >= 1 and len(task_lower) < 50:
        return get_available_model("qwen2.5-3b") or get_fallback_model()

    return get_fallback_model()


def list_available_models() -> list[ModelSpec]:
    return list(MODEL_REGISTRY.values())


SYSTEM_PROMPT = """你是一个任务调度专家。你的职责是：
1. 理解用户请求
2. 选择最合适的专家模型来处理

可用专家：
{models_info}

请直接输出专家模型名称，不要解释。

用户请求：{task}"""
