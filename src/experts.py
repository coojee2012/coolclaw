"""Expert roster — task-type routing inspired by OpenCode agents & OMO capabilities.

Each expert maps a *kind of work* to a model-pool *role*. The dispatcher picks an
expert first; model_pool.resolve_chain() then picks the best available model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ExpertSpec:
    id: str
    name: str
    title: str
    description: str
    role: str  # model-pool role: coding | fast | general | thinking | ocr | vision | documents
    icon: str
    opencode_analog: str  # build | explore | oracle | quick | librarian
    omo_capability: str  # ai_code | ai_ocr | ai_vision | ai_document | ai_summarize
    signals: tuple[str, ...]
    preferred_models: tuple[str, ...]  # MODEL_REGISTRY keys, in priority order
    capabilities: tuple[str, ...]
    when_to_use: tuple[str, ...]
    avoid_when: tuple[str, ...] = field(default_factory=tuple)


EXPERT_REGISTRY: dict[str, ExpertSpec] = {
    "build": ExpertSpec(
        id="build",
        name="代码工匠",
        title="Build Agent",
        description="编写/修改代码、调试、文件操作、运行命令与 Skill 自动化",
        role="coding",
        icon="🔨",
        opencode_analog="build",
        omo_capability="ai_code",
        signals=(
            "code", "function", "bug", "debug", "python", "javascript", "typescript",
            "sql", "refactor", "implement", "api", "script", "compile", "error",
            "文件", "代码", "编程", "修改", "创建", "删除", "运行", "终端", "skill",
            "修复", "重构", "部署", "git", "npm", "pip", "目录", "读取", "写入",
        ),
        preferred_models=("qwen2.5-coder-7b",),
        capabilities=("代码生成", "Bug 修复", "文件读写", "终端命令", "Skill 执行"),
        when_to_use=(
            "任何涉及文件或代码的任务",
            "需要执行工具/MCP 的多步骤任务",
            "技术实现与调试",
        ),
    ),
    "explore": ExpertSpec(
        id="explore",
        name="代码侦探",
        title="Explore Agent",
        description="理解代码库结构、查找实现、解释现有逻辑（只读分析为主）",
        role="coding",
        icon="🔍",
        opencode_analog="explore",
        omo_capability="ai_code",
        signals=(
            "where is", "how does", "find", "search", "explain this code",
            "architecture", "structure", "在哪", "怎么实现", "查找", "搜索",
            "代码库", "项目结构", "这个函数", "调用链",
        ),
        preferred_models=("qwen2.5-coder-7b", "qwen2.5-3b"),
        capabilities=("代码搜索", "结构分析", "调用关系", "模式发现"),
        when_to_use=("理解现有代码", "定位定义与引用", "梳理模块关系"),
        avoid_when=("需要直接修改大量文件时 — 转 build"),
    ),
    "oracle": ExpertSpec(
        id="oracle",
        name="架构军师",
        title="Oracle Agent",
        description="复杂推理、架构权衡、方案设计与深度分析",
        role="thinking",
        icon="🧠",
        opencode_analog="oracle",
        omo_capability="ai_summarize",
        signals=(
            "why", "tradeoff", "design", "architecture", "compare", "pros and cons",
            "strategy", "plan", "analyze", "reason", "math", "prove",
            "为什么", "架构", "方案", "对比", "权衡", "设计", "推理", "分析", "规划",
        ),
        preferred_models=("qwen2.5-coder-7b",),
        capabilities=("深度推理", "方案对比", "架构建议", "复杂问题拆解"),
        when_to_use=("多系统权衡", "技术选型", "复杂逻辑与数学"),
    ),
    "chat": ExpertSpec(
        id="chat",
        name="日常助手",
        title="General Agent",
        description="闲聊、写作、翻译、摘要与一般知识问答",
        role="general",
        icon="💬",
        opencode_analog="general",
        omo_capability="ai_summarize",
        signals=(
            "write", "translate", "summarize", "explain", "what is", "tell me",
            "hello", "hi", "thanks", "写", "翻译", "总结", "解释", "是什么", "你好",
            "最新", "最近", "改进", "更新", "有什么新", "recent", "news",
        ),
        preferred_models=("qwen2.5-3b",),
        capabilities=("写作", "翻译", "摘要", "常识问答", "多轮对话"),
        when_to_use=("非技术日常对话", "文案与总结", "概念解释"),
    ),
    "quick": ExpertSpec(
        id="quick",
        name="闪电应答",
        title="Quick Agent",
        description="极短问题、格式转换、简单计算 — 优先低延迟",
        role="fast",
        icon="⚡",
        opencode_analog="quick",
        omo_capability="ai_summarize",
        signals=("quick", "brief", "short", "简单", "快速", "多少", "转换"),
        preferred_models=("qwen2.5-3b",),
        capabilities=("快速问答", "简单计算", "格式转换"),
        when_to_use=("一句话能答完", "明显低复杂度"),
        avoid_when=("需要工具执行或长文生成"),
    ),
    "ocr": ExpertSpec(
        id="ocr",
        name="文字提取师",
        title="OCR Expert",
        description="从扫描件、截图、PDF 提取纯文字",
        role="ocr",
        icon="📄",
        opencode_analog="librarian",
        omo_capability="ai_ocr",
        signals=(
            "ocr", "scan", "extract text", "手写", "扫描件", "图片转文字",
            "截图文字", "pdf文字", "识别文字",
        ),
        preferred_models=("lightonocr-2-1b",),
        capabilities=("图片 OCR", "PDF 文字提取", "手写识别"),
        when_to_use=("只要纯文本、不需理解版面"),
    ),
    "vision": ExpertSpec(
        id="vision",
        name="视觉分析师",
        title="Vision Expert",
        description="理解图片内容、图表、场景描述与视觉问答",
        role="vision",
        icon="👁",
        opencode_analog="librarian",
        omo_capability="ai_vision",
        signals=(
            "image", "picture", "photo", "chart", "diagram", "screenshot",
            "图片", "照片", "图表", "截图", "看图", "描述这张",
        ),
        preferred_models=("smolvlm-256m", "minicpm-v-4"),
        capabilities=("图片描述", "图表理解", "视觉问答"),
        when_to_use=("需要理解图像语义而非仅提取文字"),
    ),
    "documents": ExpertSpec(
        id="documents",
        name="文档结构化",
        title="Document Expert",
        description="PDF/文档转 Markdown、表格提取与版面结构化",
        role="documents",
        icon="📑",
        opencode_analog="librarian",
        omo_capability="ai_document",
        signals=(
            "document", "docling", "markdown", "table extract", "layout",
            "文档", "表格提取", "结构化", "pdf转换", "版面",
        ),
        preferred_models=("granite-docling-258m", "smoldocling-256m"),
        capabilities=("文档结构化", "表格提取", "版面分析"),
        when_to_use=("需要保留文档结构与表格"),
    ),
}

# Dispatcher itself — not dispatched to, but listed for clarity
DISPATCHER_EXPERT = ExpertSpec(
    id="dispatcher",
    name="调度中心",
    title="Dispatcher",
    description="分析意图、选择专家，简单问题直接回答",
    role="fast",
    icon="🎯",
    opencode_analog="router",
    omo_capability="planner",
    signals=(),
    preferred_models=("qwen2.5-3b",),
    capabilities=("意图识别", "专家选择", "简单直答"),
    when_to_use=("所有请求的入口"),
)

EXPERT_ORDER = ("build", "explore", "oracle", "chat", "quick", "ocr", "vision", "documents")


def get_expert(expert_id: str) -> Optional[ExpertSpec]:
    return EXPERT_REGISTRY.get(expert_id)


def classify_task(text: str) -> tuple[str, float, list[str]]:
    """Score-based expert classification. Returns (expert_id, confidence, alternatives)."""
    task_lower = text.lower()
    scores: dict[str, float] = {}

    for expert in EXPERT_REGISTRY.values():
        score = 0.0
        for sig in expert.signals:
            if sig in task_lower:
                score += 1.0 if len(sig) > 4 else 0.6
        if score > 0:
            scores[expert.id] = score

    # File/code ops strongly bias toward build (OpenCode pattern: tools → build agent)
    file_ops = ("文件", "目录", "创建", "删除", "读取", "写入", "list", "mkdir", "workdir")
    if any(k in task_lower for k in file_ops):
        scores["build"] = scores.get("build", 0) + 2.0

    if is_workdir_query(text):
        scores["build"] = scores.get("build", 0) + 4.0

    if not scores:
        return "chat", 0.4, ["build", "quick"]

    # 通用知识 / 行业动态 → 日常助手，而非代码探索
    research_signals = (
        "最新", "最近", "改进", "更新", "新功能", "发布", "changelog", "release",
        "what's new", "recent", "news", "趋势", "有什么新",
    )
    code_context = (
        "代码", "文件", "函数", "类", "模块", "仓库", "repo", "codebase",
        ".py", ".js", ".ts", ".go", "workdir", "目录",
    )
    if any(s in task_lower for s in research_signals):
        if not any(c in task_lower for c in code_context):
            scores["chat"] = scores.get("chat", 0) + 3.0

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    best_id, best_score = ranked[0]
    confidence = min(0.95, 0.45 + best_score * 0.12)
    alts = [eid for eid, _ in ranked[1:4]]
    return best_id, confidence, alts


def expert_to_role(expert_id: str) -> str:
    expert = EXPERT_REGISTRY.get(expert_id)
    return expert.role if expert else "general"


def preferred_models_for_expert(expert_id: str) -> list[str]:
    expert = EXPERT_REGISTRY.get(expert_id)
    return list(expert.preferred_models) if expert else []


def needs_mcp_agent(expert_id: str, task: str) -> bool:
    """Whether the task needs full MCP filesystem/tool loop (not just cloud chat)."""
    if not expert_id:
        return False
    task_lower = (task or "").lower()
    file_ops = (
        "文件", "目录", "创建", "删除", "读取", "写入", "修改", "运行",
        "list", "mkdir", "workdir", "skill", "git ", "npm ", "pip ",
        ".py", ".js", ".ts", ".go", ".rs", ".yaml", ".json",
    )
    has_file_ops = any(k in task_lower for k in file_ops)

    if expert_id == "build":
        return True
    if expert_id == "explore":
        return has_file_ops or any(
            k in task_lower for k in ("代码库", "项目结构", "在哪", "怎么实现", "调用链", "where is", "how does")
        )
    # chat/oracle/quick/ocr/vision/documents: cloud chat suffices
    return False


WEB_SEARCH_SIGNALS = (
    "价格", "股价", "汇率", "行情", "涨跌", "市值", "quote", "price",
    "天气", "气温", "weather", "forecast",
    "新闻", "头条", "今日", "今天", "此刻", "此时", "实时", "现在",
    "latest", "live", "right now", "today",
    "btc", "bitcoin", "以太坊", "eth", "股票", "基金", "黄金",
    "多少度", "多少钱", "几点了", "什么时间",
)
WEB_SEARCH_AMBIGUOUS = ("当前", "current")


WORKDIR_QUERY_SIGNALS = (
    "工作目录", "workdir", "working directory", "work directory",
    "当前目录", "当前工作目录", "当前会话目录", "会话目录",
    "项目目录", "工作路径", "当前路径", "cwd",
)


def is_workdir_query(task: str) -> bool:
    """User asks about the session's bound project directory."""
    t = (task or "").lower()
    if any(sig in t for sig in WORKDIR_QUERY_SIGNALS):
        return True
    if "目录" in t and any(w in t for w in ("哪", "是什么", "在哪", "哪里", "什么")):
        if any(w in t for w in ("工作", "当前", "项目", "我们", "你")):
            return True
    return False


def _scrub_workdir_phrases(task: str) -> str:
    t = (task or "").lower()
    for sig in sorted(WORKDIR_QUERY_SIGNALS, key=len, reverse=True):
        t = t.replace(sig, " ")
    for phrase in (
        "你当前", "当前我们", "我们的工作目录", "你的工作目录",
        "目录是", "目录在", "目录在哪", "在哪个目录",
    ):
        t = t.replace(phrase, " ")
    return t


def needs_web_search(task: str) -> bool:
    """Tasks that need live/network data — must not be answered from model memory."""
    task_lower = (task or "").lower()
    if is_workdir_query(task):
        scrubbed = _scrub_workdir_phrases(task)
        if any(sig in scrubbed for sig in WEB_SEARCH_SIGNALS):
            return True
        return any(sig in scrubbed for sig in WEB_SEARCH_AMBIGUOUS)
    return any(sig in task_lower for sig in WEB_SEARCH_SIGNALS + WEB_SEARCH_AMBIGUOUS)


def try_workdir_answer(task: str, workdir: str) -> str | None:
    """Direct answer for session workdir — no LLM routing needed."""
    if not is_workdir_query(task) or needs_web_search(task):
        return None
    if workdir:
        from pathlib import Path
        p = Path(workdir).expanduser()
        if p.is_dir():
            return f"当前会话的工作目录是：**{workdir}**"
        return f"当前会话配置的工作目录是 `{workdir}`（路径暂时不可访问，请检查是否已挂载）。"
    return (
        "当前会话尚未设置工作目录。"
        "你可以在左侧文件面板顶部设置项目目录，之后我就能在该目录下读写文件、运行命令。"
    )


def resolve_agent_mode(expert_id: str, task: str, workdir: str = "") -> str | None:
    """Return agent mode: 'web' (search only), 'full' (filesystem+mcp), or None."""
    if is_workdir_query(task):
        return None
    if needs_mcp_agent(expert_id, task):
        return "full" if workdir else None
    if needs_web_search(task):
        return "web"
    return None


def format_experts_for_dispatcher() -> str:
    """Build the expert catalog section for the dispatcher system prompt."""
    lines = []
    for eid in EXPERT_ORDER:
        e = EXPERT_REGISTRY[eid]
        caps = "、".join(e.capabilities[:4])
        when = "；".join(e.when_to_use[:2])
        models = "、".join(e.preferred_models) or f"role={e.role}"
        lines.append(
            f"### {e.icon} {e.name} (`{e.id}`)\n"
            f"- **职责**: {e.description}\n"
            f"- **擅长**: {caps}\n"
            f"- **何时选用**: {when}\n"
            f"- **模型角色**: `{e.role}` | 首选: {models}\n"
            f"- **OpenCode 类比**: {e.opencode_analog} | **OMO 能力**: {e.omo_capability}"
        )
    return "\n\n".join(lines)


def is_multimodal_cloud_model(provider_name: str, model_name: str) -> bool:
    """Whether a cloud endpoint supports image input (Gemini multimodal, etc.)."""
    mn = (model_name or "").lower()
    pn = (provider_name or "").lower()
    if any(x in mn for x in ("gemini", "gpt-4o", "gpt-4-vision", "claude-3", "vl", "vision")):
        return True
    if pn in ("google_ai",) and "gemma" not in mn:
        return True
    return False


def infer_cloud_roles(provider_name: str, model_name: str, multimodal: bool = False) -> list[str]:
    """Infer which pool roles a cloud endpoint can serve."""
    mn = (model_name or "").lower()
    pn = (provider_name or "").lower()
    roles: list[str] = []

    if multimodal or is_multimodal_cloud_model(provider_name, model_name):
        roles.extend(["vision", "ocr"])

    if any(x in mn for x in ("coder", "code", "deepseek-coder")):
        roles.append("coding")
    if any(x in mn for x in ("flash", "lite", "2b", "3b", "1.5b", "instant", "haiku")):
        roles.append("fast")
    if any(x in mn for x in ("pro", "opus", "think", "reason", "r1")):
        roles.append("thinking")
    if any(x in mn for x in ("vision", "vl", "multimodal")):
        for r in ("vision", "ocr"):
            if r not in roles:
                roles.append(r)

    is_chat_llm = any(x in mn for x in ("gemini", "gemma", "gpt", "claude", "qwen", "deepseek", "llama"))
    if is_chat_llm or "gemma" in pn or "google" in pn:
        for r in ("coding", "general"):
            if r not in roles:
                roles.append(r)
        if any(x in mn for x in ("flash", "lite", "gemini")) and "fast" not in roles:
            roles.insert(0, "fast")

    if not roles:
        roles.append("general")
    return list(dict.fromkeys(roles))


def dispatch_payload(decision, trace: dict | None = None) -> dict:
    """SSE/API payload for expert dispatch metadata."""
    from .run_trace import trace_payload

    expert = get_expert(decision.target_expert) if decision.target_expert else None
    payload = {
        "expert": decision.target_expert or "",
        "expert_name": expert.name if expert else "",
        "expert_icon": expert.icon if expert else "🎯",
        "expert_title": expert.title if expert else "",
        "role": decision.target_role or "",
        "model": decision.target_model or "",
        "reasoning": decision.reasoning or "",
        "confidence": decision.confidence,
        "action": decision.action,
        "route_source": getattr(decision, "route_source", "") or "",
        "capabilities": getattr(decision, "capabilities", None) or [],
    }
    tp = trace or trace_payload()
    if tp:
        payload["trace"] = tp if isinstance(tp, dict) else tp
    return payload


def experts_status(pool) -> list[dict]:
    """Expert roster with currently available models from pool."""
    try:
        mode = pool._routing_mode()
    except Exception:
        mode = "auto"

    ROLE_FALLBACK = {
        "thinking": ("coding", "general"),
        "ocr": ("vision", "general"),
        "vision": ("general",),
        "documents": ("general",),
    }

    result = []
    bindings: dict = {}
    try:
        from .config import get_config
        bindings = get_config().routing.expert_bindings or {}
    except Exception:
        pass

    for eid in EXPERT_ORDER:
        expert = EXPERT_REGISTRY[eid]
        models = pool.by_role(expert.role, mode)
        fallback_used = False
        if not models:
            for fb in ROLE_FALLBACK.get(expert.role, ()):
                models = pool.by_role(fb, mode)
                if models:
                    fallback_used = True
                    break
        result.append({
            "id": expert.id,
            "name": expert.name,
            "title": expert.title,
            "icon": expert.icon,
            "role": expert.role,
            "description": expert.description,
            "opencode_analog": expert.opencode_analog,
            "omo_capability": expert.omo_capability,
            "preferred_models": list(expert.preferred_models),
            "available_models": [m.to_dict() for m in models[:6]],
            "ready": len(models) > 0,
            "fallback_role": models[0].role if models and fallback_used else None,
            "bound_provider": bindings.get(eid, ""),
        })
    return result
