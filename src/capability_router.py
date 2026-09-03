"""OMO-style capability routing: rules → capability detection → expert → optional LLM tie-break.

Pipeline (0-token first, LLM last):
  1. User override (forced expert)
  2. Session metadata (workdir, greetings)
  3. Capability chain detection
  4. Heuristic expert scoring
  5. LLM tie-break (only when confidence < threshold)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from .protocol import DispatchAction, DispatchDecision
from .experts import (
    EXPERT_REGISTRY,
    classify_task,
    expert_to_role,
    get_expert,
    is_workdir_query,
    needs_web_search,
    preferred_models_for_expert,
    try_workdir_answer,
)

logger = logging.getLogger(__name__)

LLM_CONFIDENCE_THRESHOLD = 0.65

SLIM_ROUTER_PROMPT = """你是 CoolClaw 路由助手。根据用户请求选择最合适的专家，输出严格 JSON（不要 markdown）：
专家: build=代码/文件/命令, explore=读代码库, oracle=架构设计, chat=日常/联网查询, quick=极简转换, ocr=扫描文字, vision=看图, documents=PDF结构化

规则:
- 实时价格/天气/新闻 → chat
- 写改文件/跑命令 → build
- 工作目录问题 → build（系统已知路径，不要 say 无权限）
- 简单问候可 answer

{{"action":"dispatch|answer","target_expert":"build|explore|oracle|chat|quick|ocr|vision|documents","can_answer":false,"direct_answer":null,"confidence":0.0-1.0,"reasoning":"20字内"}}

用户请求: {user_input}
{context_block}JSON:"""

CONTINUATION_SIGNALS = (
    "它", "这个", "那个", "刚才", "上面", "继续", "之前", "同样",
    "结果呢", "结果", "然后呢", "没看到", "没看", "丢了", "空白", "怎么没",
    "this", "that", "continue", "above", "result",
)


class Capability(str, Enum):
    SESSION_META = "session_meta"
    WEB_SEARCH = "web_search"
    FILE_OPS = "file_ops"
    CODE_EXPLORE = "code_explore"
    REASONING = "reasoning"
    OCR = "ocr"
    VISION = "vision"
    DOCUMENTS = "documents"
    CHAT = "chat"
    QUICK = "quick"


# Priority: higher index = lower priority when resolving conflicts
_CAPABILITY_PRIORITY = (
    Capability.OCR,
    Capability.VISION,
    Capability.DOCUMENTS,
    Capability.FILE_OPS,
    Capability.CODE_EXPLORE,
    Capability.REASONING,
    Capability.WEB_SEARCH,
    Capability.QUICK,
    Capability.CHAT,
    Capability.SESSION_META,
)

_CAPABILITY_EXPERT: dict[Capability, str] = {
    Capability.SESSION_META: "build",
    Capability.FILE_OPS: "build",
    Capability.CODE_EXPLORE: "explore",
    Capability.REASONING: "oracle",
    Capability.WEB_SEARCH: "chat",
    Capability.CHAT: "chat",
    Capability.QUICK: "quick",
    Capability.OCR: "ocr",
    Capability.VISION: "vision",
    Capability.DOCUMENTS: "documents",
}

_GREETINGS = frozenset({
    "你好", "您好", "hi", "hello", "hey", "早上好", "下午好", "晚上好",
    "谢谢", "感谢", "thanks", "thank you", "再见", "bye",
})

_OCR_SIGNALS = ("ocr", "扫描", "扫描件", "识别文字", "提取文字", "图片转文字")
_VISION_SIGNALS = ("图片", "截图", "图表", "看图", "图像", "photo", "image", "screenshot", "chart")
_DOC_SIGNALS = ("pdf", "docling", "表格提取", "版面", "结构化文档", "document parse")
_REASONING_SIGNALS = (
    "架构", "方案", "权衡", "设计", "tradeoff", "architecture",
    "怎么选", "优缺点", "技术选型", "长期",
)
_QUICK_SIGNALS = ("翻译成", "translate", "格式转换", "缩写", "扩写", "一句话")


@dataclass
class RouteContext:
    user_input: str
    workdir: str = ""
    forced_expert: str = ""
    has_attachment: bool = False
    dispatch_context: str = ""


@dataclass
class RoutePlan:
    expert_id: str
    confidence: float
    source: str
    reasoning: str
    capabilities: list[Capability] = field(default_factory=list)
    action: str = DispatchAction.DISPATCH
    direct_answer: Optional[str] = None
    alternative_experts: list[str] = field(default_factory=list)

    def to_dispatch_decision(self) -> DispatchDecision:
        expert = get_expert(self.expert_id)
        target_model = None
        alt_models: list[str] = []
        if expert and expert.preferred_models:
            target_model = expert.preferred_models[0]
            alt_models = list(expert.preferred_models[1:])
        for alt_eid in self.alternative_experts:
            for m in preferred_models_for_expert(alt_eid):
                if m not in alt_models:
                    alt_models.append(m)
        return DispatchDecision(
            action=self.action,
            target_expert=self.expert_id,
            target_role=expert_to_role(self.expert_id),
            target_model=target_model,
            confidence=self.confidence,
            reasoning=self.reasoning,
            can_answer=self.action == DispatchAction.ANSWER,
            direct_answer=self.direct_answer,
            alternative_models=alt_models,
            route_source=self.source,
            capabilities=[c.value for c in self.capabilities],
        )


class CapabilityRouter:
    """OMO-inspired planner: detect capabilities, map to expert, LLM only as tie-breaker."""

    def __init__(
        self,
        llm_dispatch_fn: Optional[Callable[["RouteContext"], DispatchDecision]] = None,
        llm_threshold: float = LLM_CONFIDENCE_THRESHOLD,
    ):
        self._llm_dispatch_fn = llm_dispatch_fn
        self._llm_threshold = llm_threshold

    def route(self, ctx: RouteContext) -> DispatchDecision:
        plan = self._plan(ctx)
        decision = plan.to_dispatch_decision()
        decision = _enrich(decision, ctx.user_input)

        if (
            plan.source != "llm"
            and plan.confidence < self._llm_threshold
            and self._llm_dispatch_fn
            and not ctx.forced_expert
        ):
            try:
                llm_decision = self._llm_dispatch_fn(ctx)
                if llm_decision.confidence > decision.confidence:
                    llm_decision = _enrich(llm_decision, ctx.user_input)
                    llm_decision.reasoning = f"LLM兜底: {llm_decision.reasoning}"
                    logger.info(
                        "[ROUTE] LLM override: %s→%s conf %.2f→%.2f",
                        plan.expert_id,
                        llm_decision.target_expert,
                        plan.confidence,
                        llm_decision.confidence,
                    )
                    return llm_decision
            except Exception as e:
                logger.warning("[ROUTE] LLM fallback failed: %s", e)

        logger.info(
            "[ROUTE] %s → expert=%s caps=%s conf=%.2f (%s)",
            ctx.user_input[:60],
            plan.expert_id,
            [c.value for c in plan.capabilities],
            plan.confidence,
            plan.source,
        )
        return decision

    def _plan(self, ctx: RouteContext) -> RoutePlan:
        text = (ctx.user_input or "").strip()
        if not text:
            return RoutePlan("chat", 0.5, "default", "空输入", [Capability.CHAT])

        # ① User override (OpenCode-style explicit agent)
        forced = (ctx.forced_expert or "").strip().lower()
        if forced and forced in EXPERT_REGISTRY:
            expert = get_expert(forced)
            return RoutePlan(
                expert_id=forced,
                confidence=1.0,
                source="user",
                reasoning=f"用户指定 {expert.name if expert else forced}",
                capabilities=_detect_capabilities(ctx),
            )

        # ② Session metadata (0-token, deterministic)
        workdir_ans = try_workdir_answer(text, ctx.workdir)
        if workdir_ans:
            return RoutePlan(
                expert_id="build",
                confidence=1.0,
                source="rule",
                reasoning="会话工作目录",
                capabilities=[Capability.SESSION_META],
                action=DispatchAction.ANSWER,
                direct_answer=workdir_ans,
            )

        greeting = _try_greeting(text)
        if greeting:
            return RoutePlan(
                expert_id="chat",
                confidence=0.95,
                source="rule",
                reasoning="问候语",
                capabilities=[Capability.CHAT],
                action=DispatchAction.ANSWER,
                direct_answer=greeting,
            )

        # ③ Capability detection
        caps = _detect_capabilities(ctx)
        cap_expert = _expert_from_capabilities(caps)

        # ④ Heuristic scoring (+ multi-turn continuation hint)
        hint_id, hint_conf, hint_alts = classify_task(text)
        hint_id, hint_conf = _apply_continuation_hint(text, ctx.dispatch_context, hint_id, hint_conf)

        # Merge capability + heuristic
        expert_id, confidence, reasoning = _merge_routing(cap_expert, caps, hint_id, hint_conf)

        return RoutePlan(
            expert_id=expert_id,
            confidence=confidence,
            source="capability",
            reasoning=reasoning,
            capabilities=caps,
            alternative_experts=hint_alts,
        )


def _detect_capabilities(ctx: RouteContext) -> list[Capability]:
    t = (ctx.user_input or "").lower()
    caps: list[Capability] = []

    if is_workdir_query(ctx.user_input):
        caps.append(Capability.SESSION_META)
    if needs_web_search(ctx.user_input):
        caps.append(Capability.WEB_SEARCH)
    if ctx.has_attachment:
        caps.append(Capability.VISION)
    if any(s in t for s in _OCR_SIGNALS):
        caps.append(Capability.OCR)
    elif any(s in t for s in _VISION_SIGNALS):
        caps.append(Capability.VISION)
    if any(s in t for s in _DOC_SIGNALS):
        caps.append(Capability.DOCUMENTS)
    if any(s in t for s in _REASONING_SIGNALS):
        caps.append(Capability.REASONING)
    if any(s in t for s in _QUICK_SIGNALS) and len(t) < 120:
        caps.append(Capability.QUICK)

    file_ops = (
        "文件", "目录", "创建", "删除", "读取", "写入", "修改", "运行",
        "list", "mkdir", "skill", "git ", "npm ", "pip ", "终端", "命令",
        ".py", ".js", ".ts", ".go", ".rs", ".yaml", ".json",
    )
    if any(k in t for k in file_ops):
        caps.append(Capability.FILE_OPS)

    explore_ops = (
        "在哪", "怎么实现", "调用链", "where is", "how does", "代码库",
        "项目结构", "这个函数", "explain this",
    )
    if any(k in t for k in explore_ops) and Capability.FILE_OPS not in caps:
        caps.append(Capability.CODE_EXPLORE)

    if not caps:
        caps.append(Capability.CHAT)
    return _dedupe_caps(caps)


def _dedupe_caps(caps: list[Capability]) -> list[Capability]:
    seen: set[Capability] = set()
    out: list[Capability] = []
    for c in caps:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _expert_from_capabilities(caps: list[Capability]) -> tuple[str, float]:
    """Pick highest-priority capability's expert."""
    best_cap = min(caps, key=lambda c: _CAPABILITY_PRIORITY.index(c))
    expert_id = _CAPABILITY_EXPERT.get(best_cap, "chat")
    # Higher confidence when multiple signals agree
    agreeing = sum(1 for c in caps if _CAPABILITY_EXPERT.get(c) == expert_id)
    conf = min(0.92, 0.55 + agreeing * 0.12)
    return expert_id, conf


def _merge_routing(
    cap_expert: tuple[str, float],
    caps: list[Capability],
    hint_id: str,
    hint_conf: float,
) -> tuple[str, float, str]:
    cap_id, cap_conf = cap_expert
    if cap_id == hint_id:
        return cap_id, max(cap_conf, hint_conf), f"能力+启发式一致→{cap_id}"
    if cap_conf >= hint_conf:
        return cap_id, cap_conf, f"能力链→{cap_id} ({', '.join(c.value for c in caps[:3])})"
    return hint_id, hint_conf, f"启发式→{hint_id}"


def _apply_continuation_hint(
    text: str, dispatch_context: str, hint_id: str, hint_conf: float,
) -> tuple[str, float]:
    """Boost prior expert when user continues a thread ('它', '继续', etc.)."""
    if not dispatch_context:
        return hint_id, hint_conf
    stripped = (text or "").strip()
    is_short_followup = len(stripped) <= 24
    has_signal = any(s in stripped for s in CONTINUATION_SIGNALS)
    if not has_signal and not is_short_followup:
        return hint_id, hint_conf
    ctx = dispatch_context.lower()
    if any(k in ctx for k in ("列出", "目录", "文件", "list_files", ".py", "build", "代码")):
        return "build", max(hint_conf, 0.78)
    if any(k in ctx for k in ("[assistant]", "build", "文件", "代码", ".py")):
        return "build", max(hint_conf, 0.72)
    if any(k in ctx for k in ("天气", "价格", "btc", "新闻", "web_search")):
        return "chat", max(hint_conf, 0.72)
    if is_short_followup and has_signal:
        return "build", max(hint_conf, 0.7)
    return hint_id, hint_conf


def _try_greeting(text: str) -> Optional[str]:
    stripped = re.sub(r"[!！?？。.\s]+$", "", text.strip().lower())
    if stripped in _GREETINGS:
        return "你好！我是 CoolClaw，可以帮你写代码、查资料、操作项目文件。有什么需要？"
    if len(stripped) <= 12 and stripped.startswith(("你好", "hi", "hello")):
        return "你好！我是 CoolClaw，有什么可以帮你的？"
    return None


def _enrich(decision: DispatchDecision, task: str) -> DispatchDecision:
    """Apply hard rules on top of any routing source (OMO executor guards)."""
    if needs_web_search(task):
        if decision.action == DispatchAction.ANSWER and decision.can_answer:
            if not is_workdir_query(task):
                decision.action = DispatchAction.DISPATCH
                decision.can_answer = False
                decision.direct_answer = None
        decision.target_expert = decision.target_expert or "chat"
        decision.target_role = decision.target_role or "general"
        if decision.confidence < 0.75:
            decision.confidence = 0.75

    if decision.target_expert:
        expert = get_expert(decision.target_expert)
        if expert:
            if not decision.target_role:
                decision.target_role = expert.role
            if not decision.target_model and expert.preferred_models:
                decision.target_model = expert.preferred_models[0]
            for m in expert.preferred_models[1:]:
                if m not in decision.alternative_models:
                    decision.alternative_models.append(m)

    if not decision.target_role and decision.target_expert:
        decision.target_role = expert_to_role(decision.target_expert)

    return decision


def plan_summary(plan: RoutePlan) -> dict:
    expert = get_expert(plan.expert_id)
    return {
        "source": plan.source,
        "capabilities": [c.value for c in plan.capabilities],
        "expert": plan.expert_id,
        "expert_name": expert.name if expert else "",
        "expert_icon": expert.icon if expert else "🎯",
        "confidence": plan.confidence,
        "reasoning": plan.reasoning,
    }
