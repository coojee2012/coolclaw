"""
AI Agent — LLM + MCP 工具调用循环

核心流程：
1. 创建 MCP Server（绑定 workdir）
2. 用 in-memory Client 连接 Server
3. 获取工具列表 → 转为 LLM function calling 格式
4. 循环：LLM 思考 → 调用工具 → 结果反馈 → 直到完成

支持两种模式：
- native: LLM 原生 function calling（Gemini 等）
- prompt: 基于提示词的工具调用（Gemma 等不支持 function calling 的模型）
"""
import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, AsyncIterator

from mcp import Client
from mcp.server import MCPServer

from .mcp_server import create_mcp_server

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """一次工具调用记录"""
    name: str
    args: dict
    result: str = ""
    success: bool = True


@dataclass
class AgentResult:
    """Agent 运行结果"""
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    rounds: int = 0
    model: str = ""
    error: Optional[str] = None


class Agent:
    """
    AI Agent：LLM 思考 + MCP 工具调用循环

    用法：
        agent = Agent(llm_fn=your_llm_function)
        result = await agent.run("帮我创建一个 hello.py 文件")
    """

    def __init__(
        self,
        llm_fn=None,
        workdir: str = "",
        max_rounds: int = 10,
        model_name: str = "",
        extra_server: MCPServer = None,
        role: str = "full",
        trace_id: str = "",
    ):
        """
        Args:
            llm_fn: async 函数 (messages, tools) -> {content, tool_calls}
                     tool_calls 格式: [{"id": "call_xxx", "function": {"name": "xxx", "arguments": "{...}"}}]
            workdir: 工作目录（沙箱根目录）
            max_rounds: 最大工具调用轮数
            model_name: 模型名称（用于日志）
            extra_server: 额外的 MCP Server（包含 context7, codegraph, websearch, lsp 工具）
            role: Agent 角色 (full/architect/coder/reviewer) — 决定可用工具范围
        """
        self.llm_fn = llm_fn
        self.workdir = workdir
        self.max_rounds = max_rounds
        self.model_name = model_name
        self.extra_server = extra_server
        self.role = role
        self.trace_id = trace_id
        self._project_memory = None
        self._files_touched: list[str] = []

    def _load_memory(self):
        """加载跨会话项目记忆"""
        if self._project_memory is not None:
            return self._project_memory
        try:
            from .memory import ProjectMemory
            if self.workdir:
                self._project_memory = ProjectMemory(self.workdir)
            else:
                self._project_memory = ProjectMemory(".")
        except Exception as e:
            logger.debug(f"[AGENT] Failed to load project memory: {e}")
        return self._project_memory

    def _get_memory_context(self) -> str:
        """获取记忆上下文块"""
        mem = self._load_memory()
        if mem:
            try:
                return mem.get_context_block(max_tokens=600)
            except Exception:
                pass
        return ""

    def _record_error(self, tool: str, error: str):
        """记录工具错误到项目记忆"""
        mem = self._load_memory()
        if mem:
            try:
                mem.record_error(tool=tool, error=error)
            except Exception:
                pass

    def _save_work_summary(self, content: str):
        """保存工作摘要到项目记忆"""
        mem = self._load_memory()
        if mem:
            try:
                project_name = Path(self.workdir).name if self.workdir else "unknown"
                mem.save_summary(
                    project_name=project_name,
                    summary=content[:500],
                    files_touched=list(set(self._files_touched)),
                )
            except Exception:
                pass

    async def run(self, task: str, system_prompt: str = "", plan_mode: bool = False) -> AgentResult:
        """
        运行 Agent：接收任务，LLM 思考，调用工具，循环直到完成

        Args:
            task: 用户任务描述
            system_prompt: 系统提示（可选）
            plan_mode: 启用任务规划（复杂任务先拆解再执行）
        """
        if not self.llm_fn:
            return AgentResult(content="Error: no LLM function provided", error="no llm_fn")

        # 创建 MCP Server 并用 in-memory Client 连接
        file_server = create_mcp_server(self.workdir)
        all_tool_calls: list[ToolCall] = []

        # 使用合并服务器（如果提供）或仅文件服务器
        server_to_use = self.extra_server if self.extra_server else file_server
        
        async with Client(server_to_use) as client:
            # 获取工具列表
            tools_result = await client.list_tools()
            mcp_tools = tools_result.tools

            if not mcp_tools:
                return AgentResult(content="Error: no tools available", error="no tools")

            # 转为 LLM function calling 格式
            tools_schema = _mcp_tools_to_openai(mcp_tools)

            # 按角色过滤工具 (B2 多 Agent)
            if self.role != "full":
                from .mcp.tool_schema import get_tools_for_role
                allowed = get_tools_for_role(self.role)
                tools_schema = [t for t in tools_schema if t["function"]["name"] in allowed]
                logger.info(f"[AGENT] role={self.role}, filtered to {len(tools_schema)} tools")

            log_prefix = f"[AGENT:{self.trace_id}]" if self.trace_id else "[AGENT]"
            logger.info(
                f"{log_prefix} start: model={self.model_name}, tools={len(tools_schema)}, "
                f"workdir={self.workdir}, task={task[:80]}"
            )

            plan_context = ""
            if plan_mode:
                plan_context = await self._plan_task(task, client, tools_schema)
                logger.info(f"[AGENT] plan generated: {plan_context[:200]}")

            # 构建消息
            system_content = system_prompt or "You are an AI coding assistant with access to file and shell tools."
            memory_ctx = self._get_memory_context()
            if memory_ctx:
                system_content += f"\n\n## 项目记忆\n{memory_ctx}"
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": task},
            ]

            if plan_context:
                messages.append({
                    "role": "user",
                    "content": f"[任务规划]\n{plan_context}\n\n请按上述规划逐步执行。每完成一步后，检查是否需要调整计划。"
                })

            for round_num in range(1, self.max_rounds + 1):
                _compress_old_messages(messages, max_chars=12000)
                try:
                    llm_response = await self.llm_fn(messages, tools_schema)
                except Exception as e:
                    logger.error(f"[AGENT] LLM call failed round {round_num}: {e}")
                    return AgentResult(
                        content=f"LLM error: {e}",
                        tool_calls=all_tool_calls,
                        rounds=round_num,
                        model=self.model_name,
                        error=str(e),
                    )

                content = llm_response.get("content", "")
                tool_calls_raw = llm_response.get("tool_calls", [])
                finish_reason = llm_response.get("finish_reason", "")

                if not tool_calls_raw and (
                    finish_reason == "MALFORMED_FUNCTION_CALL"
                    or _looks_like_internal_reasoning(content)
                ):
                    if round_num < self.max_rounds:
                        logger.warning(
                            "[AGENT] reasoning/malformed FC at round %s, nudging tool use",
                            round_num,
                        )
                        messages.append({
                            "role": "user",
                            "content": (
                                "不要输出英文思考过程。请直接调用合适的工具完成任务，"
                                "拿到工具结果后用中文回答用户。"
                            ),
                        })
                        continue

                # 没有工具调用 → 返回最终答案
                if not tool_calls_raw:
                    if _looks_like_internal_reasoning(content):
                        content = (
                            "抱歉，模型未能正确调用工具完成任务。"
                            "请重试或明确说明需要的操作（例如：列出工作目录中的 .py 文件）。"
                        )
                    logger.info(f"[AGENT] done at round {round_num}, no more tool calls")
                    self._save_work_summary(content[:300] if content else task[:200])
                    return AgentResult(
                        content=content,
                        tool_calls=all_tool_calls,
                        rounds=round_num,
                        model=self.model_name,
                    )

                # 有工具调用 → 执行
                tool_call_text = ""
                for tc in tool_calls_raw:
                    func = tc.get("function", {})
                    tool_call_text += f"TOOL:{func.get('name', '')}({func.get('arguments', '{}')})\n"

                messages.append({
                    "role": "assistant",
                    "content": (content or "") + "\n" + tool_call_text if content else tool_call_text,
                })

                READ_ONLY_TOOLS = {
                    "read_file", "list_files", "git_status", "git_log", "git_diff",
                    "git_diff_staged", "git_blame", "git_branch",
                    "codegraph_search", "codegraph_explore", "codegraph_list_symbols",
                    "lsp_diagnostics", "lsp_goto_definition", "lsp_find_references",
                    "lsp_document_symbols", "web_search", "fetch_url",
                }

                async def _exec_tool(tc):
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    tool_args_str = func.get("arguments", "{}")
                    try:
                        tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                    except json.JSONDecodeError:
                        tool_args = {}

                    required_params = {
                        "write_file": ["path", "content"],
                        "edit_file": ["path", "old_str", "new_str"],
                        "run_command": ["command"],
                        "run_test": [],
                        "read_file": ["path"],
                        "list_files": [],
                    }
                    missing = [p for p in required_params.get(tool_name, []) if p not in tool_args]
                    if missing:
                        tool_output = (
                            f"Error: 缺少必要参数 {missing}。"
                            f"正确格式: TOOL:{tool_name}("
                            + ", ".join(f'{p}="值"' for p in required_params.get(tool_name, []))
                            + ")"
                        )
                        success = False
                    else:
                        try:
                            result = await client.call_tool(tool_name, tool_args)
                            tool_output = result.content[0].text if result.content else ""
                            success = not tool_output.startswith("Error")
                        except Exception as e:
                            tool_output = f"Error: {e}"
                            success = False
                    return tool_name, tool_args, tool_output, success

                for tc in tool_calls_raw:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    logger.info(f"[AGENT] round {round_num}: {tool_name}({json.loads(tc.get('function', {}).get('arguments', '{}')) if isinstance(tc.get('function', {}).get('arguments', '{}'), str) else tc.get('function', {}).get('arguments', '{}')})")

                readonly_batch = [tc for tc in tool_calls_raw if tc.get("function", {}).get("name", "") in READ_ONLY_TOOLS]
                write_batch = [tc for tc in tool_calls_raw if tc.get("function", {}).get("name", "") not in READ_ONLY_TOOLS]

                if readonly_batch and len(readonly_batch) > 1:
                    results = await asyncio.gather(*[_exec_tool(tc) for tc in readonly_batch])
                else:
                    results = []
                    for tc in readonly_batch:
                        results.append(await _exec_tool(tc))

                for tc in write_batch:
                    results.append(await _exec_tool(tc))

                WRITE_TOOLS = {"write_file", "edit_file"}

                for tool_name, tool_args, tool_output, success in results:
                    logger.info(f"[AGENT] result: {tool_output[:200]}")
                    tc_record = ToolCall(name=tool_name, args=tool_args, result=tool_output, success=success)
                    all_tool_calls.append(tc_record)

                    if not success:
                        self._record_error(tool_name, tool_output)
                        recent_failures = [tc for tc in all_tool_calls[-3:] if not tc.success]
                        if len(recent_failures) >= 3 and all(tc.name == tool_name for tc in recent_failures):
                            error_detail = f"工具 {tool_name} 连续失败 3 次。错误: {tool_output}\n请检查参数是否正确，或换一种方式完成任务。"
                            messages.append({"role": "user", "content": error_detail})
                            continue

                    tool_result_text = f"[Tool Result: {tool_name}]\n{tool_output}"
                    messages.append({"role": "user", "content": tool_result_text})

                    if success and tool_name in WRITE_TOOLS:
                        changed_path = tool_args.get("path", "")
                        if changed_path:
                            self._files_touched.append(changed_path)
                        if changed_path and changed_path.endswith(".py"):
                            try:
                                diag_result = await client.call_tool("lsp_diagnostics", {"file_path": changed_path})
                                diag_text = diag_result.content[0].text if diag_result.content else ""
                                if diag_text and "error" in diag_text.lower():
                                    diag_summary = diag_text[:600]
                                    messages.append({
                                        "role": "user",
                                        "content": f"[Self-Review] {changed_path} 存在类型/语法错误:\n{diag_summary}\n请修复这些错误后重试。",
                                    })
                                    logger.info(f"[AGENT] self-review: found issues in {changed_path}")
                            except Exception:
                                pass

                            if diag_text and "error" in diag_text.lower():
                                if not hasattr(self, '_consecutive_edit_failures'):
                                    self._consecutive_edit_failures = 0
                                self._consecutive_edit_failures += 1

                                if self._consecutive_edit_failures >= 2:
                                    logger.info("[AGENT] auto git stash due to consecutive edit failures")
                                    try:
                                        stash_result = await client.call_tool(
                                            "run_command",
                                            {"command": "git stash push -m 'auto-stash: too many edit failures'"}
                                        )
                                        stash_text = stash_result.content[0].text if stash_result.content else ""
                                        if "No local changes" not in stash_text:
                                            messages.append({
                                                "role": "user",
                                                "content": f"[Self-Review] 自动回滚: 已执行 `git stash` 保存当前修改。请换一种方式重新实现。\n\n{diag_summary}"
                                            })
                                            self._consecutive_edit_failures = 0
                                    except Exception:
                                        pass

                        elif success and tool_name in WRITE_TOOLS:
                            if hasattr(self, '_consecutive_edit_failures'):
                                self._consecutive_edit_failures = 0

            # 达到最大轮数 — 再调一次 LLM 做总结（业界 Agent 最佳实践）
            logger.warning(f"[AGENT] max rounds ({self.max_rounds}) reached")
            return await self._finalize_after_max_rounds(
                messages, all_tool_calls, fallback_content=content,
            )


    async def run_stream(self, task: str, system_prompt: str = "", plan_mode: bool = False) -> AsyncIterator[str]:
        """
        Streaming version of run() -- yields SSE events as JSON strings.

        Events:
          {"type": "token", "content": "..."}
          {"type": "tool_start", "tool": "git_status", "args": {...}}
          {"type": "tool_end", "tool": "git_status", "success": true, "preview": "..."}
          {"type": "done", "content": "...", "tool_calls": [...], "rounds": N}
          {"type": "error", "message": "..."}
        """
        if not self.llm_fn:
            yield json.dumps({"type": "error", "message": "no llm_fn"})
            return

        file_server = create_mcp_server(self.workdir)
        server_to_use = self.extra_server if self.extra_server else file_server
        all_tool_calls: list[ToolCall] = []

        async with Client(server_to_use) as client:
            tools_result = await client.list_tools()
            mcp_tools = tools_result.tools
            if not mcp_tools:
                yield json.dumps({"type": "error", "message": "no tools available"})
                return

            tools_schema = _mcp_tools_to_openai(mcp_tools)

            if self.role != "full":
                from .mcp.tool_schema import get_tools_for_role
                allowed = get_tools_for_role(self.role)
                tools_schema = [t for t in tools_schema if t["function"]["name"] in allowed]

            plan_context = ""
            if plan_mode:
                yield json.dumps({"type": "token", "content": "[Planning...]\n"})
                plan_context = await self._plan_task(task, client, tools_schema)
                if plan_context:
                    yield json.dumps({"type": "token", "content": plan_context + "\n\n"})

            system_content = system_prompt or "You are an AI coding assistant with access to file and shell tools."
            memory_ctx = self._get_memory_context()
            if memory_ctx:
                system_content += f"\n\n## 项目记忆\n{memory_ctx}"
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": task},
            ]
            if plan_context:
                messages.append({"role": "user", "content": f"[任务规划]\n{plan_context}\n\n请按上述规划逐步执行。"})

            READ_ONLY_TOOLS = {
                "read_file", "list_files", "git_status", "git_log", "git_diff",
                "git_diff_staged", "git_blame", "git_branch",
                "codegraph_search", "codegraph_explore", "codegraph_list_symbols",
                "lsp_diagnostics", "lsp_goto_definition", "lsp_find_references",
                "lsp_document_symbols", "web_search", "fetch_url",
            }
            WRITE_TOOLS = {"write_file", "edit_file"}

            for round_num in range(1, self.max_rounds + 1):
                _compress_old_messages(messages, max_chars=12000)
                try:
                    llm_response = await self.llm_fn(messages, tools_schema)
                except Exception as e:
                    yield json.dumps({"type": "error", "message": f"LLM error: {e}"})
                    return

                content = llm_response.get("content", "")
                tool_calls_raw = llm_response.get("tool_calls", [])
                finish_reason = llm_response.get("finish_reason", "")

                if not tool_calls_raw and (
                    finish_reason == "MALFORMED_FUNCTION_CALL"
                    or _looks_like_internal_reasoning(content)
                ):
                    if round_num < self.max_rounds:
                        messages.append({
                            "role": "user",
                            "content": (
                                "不要输出思考过程。请直接调用工具，完成后用中文回答。"
                            ),
                        })
                        continue

                # 仅最终轮（无工具调用）才流式输出，避免思考过程泄漏到 UI
                if content and not tool_calls_raw:
                    yield json.dumps({"type": "token", "content": content})

                if not tool_calls_raw:
                    if _looks_like_internal_reasoning(content):
                        content = (
                            "抱歉，模型未能正确调用工具完成任务。"
                            "请重试或明确说明需要的操作。"
                        )
                    self._save_work_summary(content[:300] if content else task[:200])
                    yield json.dumps({
                        "type": "done",
                        "content": content,
                        "tool_calls": [
                            {"name": tc.name, "args": tc.args, "success": tc.success}
                            for tc in all_tool_calls
                        ],
                        "rounds": round_num,
                    })
                    return

                tool_call_text = ""
                for tc in tool_calls_raw:
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    tool_call_text += f"TOOL:{name}({func.get('arguments', '{}')})\n"

                messages.append({
                    "role": "assistant",
                    "content": (content or "") + "\n" + tool_call_text if content else tool_call_text,
                })

                async def _exec_one(tc):
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    tool_args_str = func.get("arguments", "{}")
                    try:
                        tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                    except json.JSONDecodeError:
                        tool_args = {}

                    required_params = {
                        "write_file": ["path", "content"],
                        "edit_file": ["path", "old_str", "new_str"],
                        "run_command": ["command"],
                        "read_file": ["path"],
                    }
                    missing = [p for p in required_params.get(tool_name, []) if p not in tool_args]
                    if missing:
                        tool_output = f"Error: missing params {missing}"
                        success = False
                    else:
                        try:
                            result = await client.call_tool(tool_name, tool_args)
                            tool_output = result.content[0].text if result.content else ""
                            success = not tool_output.startswith("Error")
                        except Exception as e:
                            tool_output = f"Error: {e}"
                            success = False
                    return tool_name, tool_args, tool_output, success

                for tc in tool_calls_raw:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    yield json.dumps({"type": "tool_start", "tool": tool_name})

                readonly_batch = [tc for tc in tool_calls_raw if tc.get("function", {}).get("name", "") in READ_ONLY_TOOLS]
                write_batch = [tc for tc in tool_calls_raw if tc.get("function", {}).get("name", "") not in READ_ONLY_TOOLS]

                results = []
                if readonly_batch and len(readonly_batch) > 1:
                    results = list(await asyncio.gather(*[_exec_one(tc) for tc in readonly_batch]))
                else:
                    for tc in readonly_batch:
                        results.append(await _exec_one(tc))
                for tc in write_batch:
                    results.append(await _exec_one(tc))

                for tool_name, tool_args, tool_output, success in results:
                    preview = tool_output[:200] if len(tool_output) > 200 else tool_output
                    yield json.dumps({"type": "tool_end", "tool": tool_name, "success": success, "preview": preview})

                    tc_record = ToolCall(name=tool_name, args=tool_args, result=tool_output, success=success)
                    all_tool_calls.append(tc_record)

                    if not success:
                        self._record_error(tool_name, tool_output)

                    messages.append({"role": "user", "content": f"[Tool Result: {tool_name}]\n{tool_output}"})

                    if success and tool_name in WRITE_TOOLS:
                        changed_path = tool_args.get("path", "")
                        if changed_path:
                            self._files_touched.append(changed_path)

            final = await self._finalize_after_max_rounds(
                messages, all_tool_calls,
                fallback_content="达到最大工具调用轮数，请根据已有信息回答。",
            )
            if final.content:
                yield json.dumps({"type": "token", "content": final.content})
            yield json.dumps({
                "type": "done",
                "content": final.content,
                "tool_calls": [
                    {"name": tc.name, "args": tc.args, "success": tc.success}
                    for tc in all_tool_calls
                ],
                "rounds": self.max_rounds,
            })


    def run_sync(self, task: str, system_prompt: str = "") -> AgentResult:
        import concurrent.futures
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.run(task, system_prompt))
                return future.result(timeout=300)
        except Exception as e:
            logger.error(f"[AGENT] run_sync failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return AgentResult(content=f"Agent error: {e}", error=str(e))

    async def _finalize_after_max_rounds(
        self,
        messages: list,
        all_tool_calls: list[ToolCall],
        fallback_content: str = "",
    ) -> AgentResult:
        """Summarize when tool rounds are exhausted instead of hard-stopping."""
        _compress_old_messages(messages, max_chars=12000)
        messages.append({
            "role": "user",
            "content": (
                "工具调用轮数已达上限。请根据以上对话和工具结果，用中文给出最终回答，"
                "总结已完成的工作与剩余事项，不要再调用工具。"
            ),
        })
        try:
            llm_response = await self.llm_fn(messages, [])
            content = (llm_response.get("content") or "").strip()
            if content:
                self._save_work_summary(content[:300])
                return AgentResult(
                    content=content,
                    tool_calls=all_tool_calls,
                    rounds=self.max_rounds,
                    model=self.model_name,
                )
        except Exception as e:
            logger.warning("[AGENT] max-rounds finalize failed: %s", e)
        return AgentResult(
            content=fallback_content or "达到最大工具调用轮数，请根据已有信息回答。",
            tool_calls=all_tool_calls,
            rounds=self.max_rounds,
            model=self.model_name,
        )

    async def _plan_task(self, task: str, client, tools_schema: list) -> str:
        """对复杂任务先让 LLM 拆解为编号步骤，返回规划文本"""
        complexity_hint = (
            "分析以下任务并输出一个编号步骤计划。"
            "每步一行，格式：\n"
            "1. [动作] 目标描述\n"
            "2. [动作] 目标描述\n"
            "...\n\n"
            "可用动作：读取(read_file)、搜索(codegraph_search)、"
            "诊断(lsp_diagnostics)、编辑(edit_file)、写入(write_file)、"
            "运行(run_command/run_test)、Git操作、网络查询\n\n"
            "要求：\n"
            "- 步骤之间如有依赖请标注\n"
            "- 预估每步可能遇到的问题\n"
            "- 输出完计划后直接开始执行，不要重复说明"
        )
        try:
            plan_response = await self.llm_fn(
                [
                    {"role": "system", "content": complexity_hint},
                    {"role": "user", "content": task},
                ],
                [],
            )
            return plan_response.get("content", "")
        except Exception as e:
            logger.warning(f"[AGENT] plan generation failed: {e}")
            return ""


# ── Helper functions ──


def _mcp_tools_to_openai(mcp_tools) -> list[dict]:
    result = []
    for tool in mcp_tools:
        schema = tool.input_schema or {"type": "object", "properties": {}}
        result.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": schema,
            },
        })
    return result


def _looks_like_internal_reasoning(text: str) -> bool:
    """Detect chain-of-thought leaked as assistant content instead of tool calls."""
    if not text or len(text) < 80:
        return False
    t = text.lower()
    markers = (
        "wait, the prompt", "actually, looking at", "i should use",
        "i'll assume", "let me think", "the user wants", "i need to",
        "思考过程", "我应该", "让我先", "用户想要",
    )
    hits = sum(1 for m in markers if m in t)
    return hits >= 2 or (hits >= 1 and len(text) > 400)


def _infer_simple_tool_call(messages: list, tools: list) -> list[dict]:
    """Fallback when cloud model returns MALFORMED_FUNCTION_CALL."""
    names = {t.get("function", {}).get("name", "") for t in tools}
    user_text = " ".join(
        m.get("content", "") for m in messages if m.get("role") == "user"
    ).lower()
    if "list_files" in names and any(k in user_text for k in ("列出", "目录", "list", "文件")):
        args = {"path": "."}
        if ".py" in user_text or "py 文件" in user_text or "python" in user_text:
            if "run_command" in names:
                return [{
                    "id": "call_infer",
                    "function": {
                        "name": "run_command",
                        "arguments": json.dumps({
                            "command": "find . -maxdepth 4 -name '*.py' -type f 2>/dev/null | head -80",
                        }, ensure_ascii=False),
                    },
                }]
        return [{
            "id": "call_infer",
            "function": {
                "name": "list_files",
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        }]
    return []


def _parse_prompt_tool_calls(text: str) -> list[dict]:
    calls = []
    for m in re.finditer(r'TOOL:(\w+)\(([^)]*)\)', text):
        name = m.group(1)
        args_str = m.group(2).strip()
        args = {}

        if args_str.startswith('{') and args_str.endswith('}'):
            try:
                args = json.loads(args_str.replace("'", '"'))
            except json.JSONDecodeError:
                try:
                    fixed = args_str.replace("'", '"')
                    fixed = re.sub(r'(\w+)\s*:', r'"\1":', fixed)
                    args = json.loads(fixed)
                except json.JSONDecodeError:
                    pass
        else:
            for kv in re.finditer(r'(\w+)=["\']([^"\']*)["\']', args_str):
                args[kv.group(1)] = kv.group(2)

        if name == "write_file" and "content" not in args:
            code_block = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
            if code_block:
                args["content"] = code_block.group(1).strip()

        if name == "write_file" and "content" not in args:
            after_tool = text[m.end():]
            code_block = re.search(r'```(?:python)?\s*\n(.*?)```', after_tool, re.DOTALL)
            if code_block:
                args["content"] = code_block.group(1).strip()

        calls.append({
            "id": f"call_{len(calls)}",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })
    return calls


def _compress_old_messages(messages: list, max_chars: int = 12000) -> None:
    """当消息历史过长时，裁剪早期工具结果以控制 context window 大小"""
    total = sum(len(m.get("content", "")) for m in messages)
    if total <= max_chars:
        return
    trimmed = 0
    for m in messages:
        if m.get("role") == "user" and m.get("content", "").startswith("[Tool Result:"):
            if total - trimmed <= max_chars:
                break
            original = m["content"]
            lines = original.split("\n")
            if len(lines) > 6:
                m["content"] = "\n".join(lines[:6]) + f"\n... (truncated, {len(original)} chars total)"
                trimmed += len(original) - len(m["content"])


def _run_async(coro):
    """在同步上下文中运行异步协程"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 已在事件循环中，创建新线程
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)
