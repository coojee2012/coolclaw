import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from mcp.server import MCPServer

logger = logging.getLogger(__name__)


@dataclass
class AgentStep:
    role: str
    content: str
    tool_calls: list = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


@dataclass
class PipelineResult:
    steps: list[AgentStep] = field(default_factory=list)
    final_content: str = ""
    total_tool_calls: int = 0
    rounds: int = 0


def is_complex_task(task: str) -> bool:
    action_keywords = {
        "分析", "搜索", "读取", "查看", "检查", "诊断",
        "创建", "编写", "修改", "编辑", "实现", "添加",
        "测试", "验证", "审查", "优化", "重构",
        "analyze", "search", "read", "check", "diagnose",
        "create", "write", "modify", "edit", "implement", "add",
        "test", "verify", "review", "optimize", "refactor",
    }
    task_lower = task.lower()
    matches = sum(1 for kw in action_keywords if kw in task_lower)
    if matches >= 3 and len(task) > 100:
        return True
    file_extensions = sum(task.count(ext) for ext in (".py", ".js", ".ts", ".go", ".rs"))
    if file_extensions >= 3:
        return True
    return False


class Orchestrator:

    def __init__(
        self,
        llm_fn=None,
        workdir: str = "",
        max_rounds: int = 15,
        model_name: str = "",
        extra_server: MCPServer = None,
        trace_id: str = "",
    ):
        self.llm_fn = llm_fn
        self.workdir = workdir
        self.max_rounds = max_rounds
        self.model_name = model_name
        self.extra_server = extra_server
        self.trace_id = trace_id

    def _create_agent(self, role: str, max_rounds: int = None):
        from .agent import Agent
        return Agent(
            llm_fn=self.llm_fn,
            workdir=self.workdir,
            max_rounds=max_rounds or self.max_rounds,
            model_name=self.model_name,
            extra_server=self.extra_server,
            role=role,
            trace_id=self.trace_id,
        )

    async def run(self, task: str, system_prompt: str = "", force_pipeline: bool = False) -> PipelineResult:
        use_pipeline = force_pipeline or is_complex_task(task)

        if not use_pipeline:
            agent = self._create_agent(role="full")
            result = await agent.run(task, system_prompt)
            return PipelineResult(
                steps=[AgentStep(role="full", content=result.content, tool_calls=[tc.__dict__ for tc in result.tool_calls])],
                final_content=result.content,
                total_tool_calls=len(result.tool_calls),
                rounds=result.rounds,
            )

        logger.info(f"[ORCH] pipeline: {task[:80]}")

        architect_result = await self._run_architect(task, system_prompt)
        if not architect_result or not architect_result.tool_calls:
            if architect_result:
                return PipelineResult(
                    steps=[AgentStep(role="architect", content=architect_result.content)],
                    final_content=architect_result.content,
                    rounds=1,
                )

        coder_result = await self._run_coder(task, architect_result, system_prompt)
        reviewer_result = await self._run_reviewer(task, coder_result, system_prompt)

        all_tool_calls = []
        for step in [architect_result, coder_result, reviewer_result]:
            if step:
                all_tool_calls.extend(step.tool_calls)

        final_content = reviewer_result.content if reviewer_result else (coder_result.content if coder_result else "")

        def _step_data(result, role):
            if not result:
                return AgentStep(role=role, content="", success=False)
            return AgentStep(
                role=role,
                content=result.content,
                tool_calls=[tc.__dict__ for tc in result.tool_calls],
            )

        return PipelineResult(
            steps=[_step_data(architect_result, "architect"), _step_data(coder_result, "coder"), _step_data(reviewer_result, "reviewer")],
            final_content=final_content,
            total_tool_calls=len(all_tool_calls),
            rounds=3,
        )

    async def _run_architect(self, task: str, system_prompt: str):
        prompt = (
            "你是架构师。分析代码库并制定实现方案。\n"
            "你可以读取文件、搜索符号、查看类型诊断。\n"
            "输出：分析结论 + 编号实现步骤。\n"
            "不要直接修改代码。"
        )
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        agent = self._create_agent(role="architect", max_rounds=5)
        try:
            return await agent.run(task, full_prompt)
        except Exception as e:
            logger.error(f"[ORCH] architect failed: {e}")
            return None

    async def _run_coder(self, task: str, architect_result, system_prompt: str):
        plan = architect_result.content if architect_result else ""
        prompt = (
            "你是开发者。根据架构师方案实现代码修改。\n"
            "你可以读写文件、执行命令、操作 Git。\n"
            "每次修改后检查 LSP 诊断。"
        )
        full_task = f"原始任务: {task}\n\n架构师方案:\n{plan}"
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        agent = self._create_agent(role="coder", max_rounds=10)
        try:
            return await agent.run(full_task, full_prompt)
        except Exception as e:
            logger.error(f"[ORCH] coder failed: {e}")
            return None

    async def _run_reviewer(self, task: str, coder_result, system_prompt: str):
        changes = coder_result.content if coder_result else ""
        prompt = (
            "你是审查者。检查代码变更的质量。\n"
            "你可以运行测试、查看 Git diff、读取文件、运行类型诊断。\n"
            "输出审查结论。"
        )
        full_task = f"原始任务: {task}\n\n实现结果:\n{changes}"
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        agent = self._create_agent(role="reviewer", max_rounds=5)
        try:
            return await agent.run(full_task, full_prompt)
        except Exception as e:
            logger.error(f"[ORCH] reviewer failed: {e}")
            return None

    async def run_stream(self, task: str, system_prompt: str = "", force_pipeline: bool = False):
        use_pipeline = force_pipeline or is_complex_task(task)

        if not use_pipeline:
            agent = self._create_agent(role="full")
            async for event in agent.run_stream(task, system_prompt):
                yield event
            return

        yield json.dumps({"type": "token", "content": "[Pipeline: Architect → Coder → Reviewer]\n"})

        yield json.dumps({"type": "token", "content": "\n## Phase 1: Architecture Analysis\n"})
        architect_result = await self._run_architect(task, system_prompt)
        if architect_result:
            yield json.dumps({"type": "token", "content": architect_result.content + "\n"})

        yield json.dumps({"type": "token", "content": "\n## Phase 2: Implementation\n"})
        coder_result = await self._run_coder(task, architect_result, system_prompt)
        if coder_result:
            yield json.dumps({"type": "token", "content": coder_result.content + "\n"})

        yield json.dumps({"type": "token", "content": "\n## Phase 3: Review\n"})
        reviewer_result = await self._run_reviewer(task, coder_result, system_prompt)
        if reviewer_result:
            yield json.dumps({"type": "token", "content": reviewer_result.content + "\n"})

        final = reviewer_result.content if reviewer_result else (coder_result.content if coder_result else "")
        yield json.dumps({"type": "done", "content": final, "tool_calls": [], "rounds": 3})
