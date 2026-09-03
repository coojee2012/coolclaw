"""
Combined MCP Server — 一个 MCPServer 包含所有工具

文件操作 + context7 + codegraph(SQLite直查) + websearch + lsp(basedpyright) = 一个 server。
"""

from __future__ import annotations

import os
import subprocess
import json
import hashlib
import sqlite3
import sys
import time
from pathlib import Path
from typing import Annotated

import httpx
from pydantic import Field
from mcp.server import MCPServer
from src.sandbox import is_command_allowed, confined_subprocess_run, get_capabilities


def _find_codegraph_db(project_path: str) -> str | None:
    """向上查找 .codegraph/codegraph.db"""
    p = Path(project_path).expanduser().resolve()
    for _ in range(10):
        db = p / ".codegraph" / "codegraph.db"
        if db.is_file():
            return str(db)
        parent = p.parent
        if parent == p:
            break
        p = parent
    return None


def _cg_query(db_path: str, sql: str, params: tuple = ()) -> list[dict]:
    """查询 codegraph SQLite，返回字典列表"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()





def create_combined_server(workdir: str = "", project_path: str = "", proxy: str = "") -> MCPServer:
    server = MCPServer("coolclaw-combined")
    root = Path(workdir).expanduser().resolve() if workdir else Path.cwd()
    project = project_path or workdir
    http_proxy = proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY", "")

    def _safe_path(rel: str) -> Path:
        target = (root / rel).expanduser().resolve()
        if not str(target).startswith(str(root)):
            raise ValueError(f"Path escapes workdir: {rel}")
        return target

    # --- 文件操作 ---

    @server.tool(
        title="list_files",
        annotations={"read_only_hint": True, "open_world_hint": False},
    )
    def list_files(
        path: Annotated[str, Field(description="相对于工作目录的路径，默认 '.'")] = ".",
    ) -> str:
        try:
            target = _safe_path(path)
            if not target.is_dir():
                return f"Error: not a directory: {path}"
            entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            lines = []
            for e in entries[:50]:
                suffix = "/" if e.is_dir() else ""
                size = ""
                if e.is_file():
                    try:
                        size = f"  ({e.stat().st_size} bytes)"
                    except OSError:
                        pass
                lines.append(f"  {e.name}{suffix}{size}")
            return "\n".join(lines) if lines else "(empty directory)"
        except Exception as e:
            return f"Error: {e}"

    @server.tool(
        title="read_file",
        annotations={"read_only_hint": True, "open_world_hint": False},
    )
    def read_file(
        path: Annotated[str, Field(description="相对于工作目录的文件路径")],
        offset: Annotated[int, Field(description="起始行号（从1开始）", ge=1)] = 1,
        limit: Annotated[int, Field(description="最大行数", ge=1, le=2000)] = 500,
    ) -> str:
        try:
            target = _safe_path(path)
            if not target.is_file():
                return f"Error: not a file: {path}"
            content = target.read_text(errors="replace")
            all_lines = content.splitlines()
            total = len(all_lines)
            start = offset - 1
            selected = all_lines[start : start + limit]
            numbered = [f"{i + offset}: {line}" for i, line in enumerate(selected)]
            header = f"{path} ({total} lines, showing {start + 1}-{min(start + limit, total)})\n"
            return header + "\n".join(numbered)
        except Exception as e:
            return f"Error: {e}"

    @server.tool(
        title="write_file",
        annotations={"read_only_hint": False, "destructive_hint": True, "idempotent_hint": True},
    )
    def write_file(
        path: Annotated[str, Field(description="相对于工作目录的文件路径")],
        content: Annotated[str, Field(description="要写入的文件内容")],
    ) -> str:
        try:
            target = _safe_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            existed = target.exists()
            target.write_text(content, encoding="utf-8")
            lines = content.count("\n") + 1
            action = "覆盖" if existed else "创建"
            return f"已{action}文件 {path} ({lines} 行, {len(content)} 字符)"
        except Exception as e:
            return f"Error: {e}"

    @server.tool(
        title="edit_file",
        annotations={"read_only_hint": False, "destructive_hint": True},
    )
    def edit_file(
        path: Annotated[str, Field(description="相对于工作目录的文件路径")],
        old_str: Annotated[str, Field(description="要替换的原始文本")],
        new_str: Annotated[str, Field(description="替换后的新文本")],
    ) -> str:
        try:
            target = _safe_path(path)
            if not target.is_file():
                return f"Error: not a file: {path}"
            content = target.read_text(errors="replace")
            if old_str not in content:
                return f"Error: old_str not found in {path}"
            count = content.count(old_str)
            new_content = content.replace(old_str, new_str, 1)
            target.write_text(new_content, encoding="utf-8")
            return f"已替换 {path} 中的内容 ({count} 处匹配, 已替换第1处)"
        except Exception as e:
            return f"Error: {e}"

    @server.tool(
        title="run_command",
        annotations={"read_only_hint": False, "open_world_hint": True},
    )
    def run_command(
        command: Annotated[str, Field(description="要执行的终端命令")],
        timeout: Annotated[int, Field(description="超时时间（秒）", ge=1, le=120)] = 30,
    ) -> str:
        # Allowlist check (replaces old blacklist)
        blocked = is_command_allowed(command)
        if blocked:
            return f"Error: {blocked}"
        try:
            # Use sandbox-confined subprocess when OS sandbox is available
            caps = get_capabilities()
            if caps.seatbelt or caps.landlock or caps.bubblewrap:
                result = confined_subprocess_run(
                    command,
                    str(root),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            else:
                # Fallback: no OS sandbox, but allowlist still applies
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(root),
                )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            return output.strip() or "(no output)"
        except PermissionError as e:
            return f"Error: {e}"
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {timeout}s"
        except Exception as e:
            return f"Error: {e}"

    _cache: dict[str, tuple[float, object]] = {}
    _CACHE_TTL = 600

    def _is_cache_fresh(cache_key: str) -> bool:
        if cache_key not in _cache:
            return False
        ts, _ = _cache[cache_key]
        return (time.time() - ts) < _CACHE_TTL

    # --- Context7 ---

    @server.tool(title="resolve_library_id", annotations={"readOnlyHint": True})
    def resolve_library_id(library_name: str) -> str:
        if _is_cache_fresh(f"lib:{library_name}"):
            return json.dumps(_cache[f"lib:{library_name}"][1], ensure_ascii=False)
        resp = httpx.post(
            "https://api.context7.com/api/v1/resolve-library-id",
            json={"libraryName": library_name},
            timeout=10,
            proxy=http_proxy or None,
        )
        if resp.status_code != 200:
            return f"Context7 API error: HTTP {resp.status_code}"
        data = resp.json()
        _cache[f"lib:{library_name}"] = (time.time(), data)
        return json.dumps(data, ensure_ascii=False)

    @server.tool(title="query_docs", annotations={"readOnlyHint": True})
    def query_docs(library_id: str, query: str) -> str:
        cache_key = f"doc:{library_id}:{hashlib.md5(query.encode()).hexdigest()}"
        if _is_cache_fresh(cache_key):
            return json.dumps(_cache[cache_key][1], ensure_ascii=False)
        resp = httpx.post(
            "https://api.context7.com/api/v1/query-docs",
            json={"libraryId": library_id, "query": query, "tokens": 10000},
            timeout=15,
            proxy=http_proxy or None,
        )
        if resp.status_code != 200:
            return f"Context7 API error: HTTP {resp.status_code}"
        data = resp.json()
        _cache[cache_key] = (time.time(), data)
        return json.dumps(data, ensure_ascii=False)

    # --- Codegraph (SQLite 直查) ---

    # 先从 project 查找，找不到则从 coolclaw 项目根目录查找
    _cg_db = _find_codegraph_db(project) if project else None
    if not _cg_db:
        # 回退到 coolclaw 项目自身的 .codegraph
        _this_dir = str(Path(__file__).resolve().parent.parent.parent)
        _cg_db = _find_codegraph_db(_this_dir)

    def _cg_available() -> str | None:
        """返回错误信息或 None（可用）"""
        if not _cg_db:
            return "Error: .codegraph/codegraph.db not found. Run 'opencode' in project to index."
        return None

    @server.tool(title="codegraph_search", annotations={"readOnlyHint": True})
    def codegraph_search(query: str, limit: int = 20) -> str:
        """FTS 全文搜索符号（名称、签名、docstring）"""
        err = _cg_available()
        if err:
            return err
        try:
            # FTS5 查询：搜索 name + qualified_name + docstring + signature
            sql = """
                SELECT n.id, n.kind, n.name, n.qualified_name, n.file_path,
                       n.start_line, n.end_line, n.signature, n.docstring
                FROM nodes_fts f
                JOIN nodes n ON f.rowid = n.rowid
                WHERE nodes_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """
            rows = _cg_query(_cg_db, sql, (query, limit))
            if not rows:
                return f"No symbols found for: {query}"
            lines = []
            for r in rows:
                sig = (r.get("signature") or "").strip().split("\n")[0][:80]
                doc = (r.get("docstring") or "").strip()[:120]
                lines.append(
                    f"[{r['kind']}] {r['name']}  ({r['file_path']}:{r['start_line']}-{r['end_line']})"
                    + (f"\n  sig: {sig}" if sig else "")
                    + (f"\n  doc: {doc}" if doc else "")
                )
            return f"Found {len(rows)} symbols:\n\n" + "\n\n".join(lines)
        except Exception as e:
            return f"codegraph_search error: {e}"

    @server.tool(title="codegraph_explore", annotations={"readOnlyHint": True})
    def codegraph_explore(query: str) -> str:
        """搜索符号并返回源码 + 调用路径（call graph blast radius）"""
        err = _cg_available()
        if err:
            return err
        try:
            # Step 1: FTS 搜索匹配的符号
            sql = """
                SELECT n.id, n.kind, n.name, n.file_path, n.start_line, n.end_line
                FROM nodes_fts f
                JOIN nodes n ON f.rowid = n.rowid
                WHERE nodes_fts MATCH ?
                ORDER BY rank
                LIMIT 10
            """
            symbols = _cg_query(_cg_db, sql, (query,))
            if not symbols:
                return f"No symbols found for: {query}"

            results = []
            for sym in symbols:
                sym_id = sym["id"]
                name = sym["name"]
                fpath = sym["file_path"]
                lines_range = f"L{sym['start_line']}-{sym['end_line']}"

                # Step 2: 读取源码片段
                source_preview = ""
                full_path = root / fpath
                if full_path.is_file():
                    try:
                        all_lines = full_path.read_text(errors="replace").splitlines()
                        start = max(0, sym["start_line"] - 1)
                        end = min(len(all_lines), sym["end_line"])
                        snippet = all_lines[start:end]
                        source_preview = "\n".join(f"  {i + start + 1}: {l}" for i, l in enumerate(snippet))
                        if len(source_preview) > 2000:
                            source_preview = source_preview[:2000] + "\n  ... (truncated)"
                    except Exception:
                        source_preview = "  (could not read file)"

                # Step 3: 调用路径 — 谁调用了这个符号
                callers_sql = """
                    SELECT DISTINCT n2.name, n2.file_path, n2.start_line, e2.kind
                    FROM edges e2
                    JOIN nodes n2 ON e2.source = n2.id
                    WHERE e2.target = ? AND e2.kind IN ('calls', 'references', 'instantiates')
                    LIMIT 10
                """
                callers = _cg_query(_cg_db, callers_sql, (sym_id,))

                # Step 4: 这个符号调用了谁
                callees_sql = """
                    SELECT DISTINCT n2.name, n2.file_path, n2.start_line, e2.kind
                    FROM edges e2
                    JOIN nodes n2 ON e2.target = n2.id
                    WHERE e2.source = ? AND e2.kind IN ('calls', 'references', 'instantiates')
                    LIMIT 10
                """
                callees = _cg_query(_cg_db, callees_sql, (sym_id,))

                # Step 5: 包含的子节点
                children_sql = """
                    SELECT n2.name, n2.kind, n2.start_line, n2.end_line
                    FROM edges e2
                    JOIN nodes n2 ON e2.target = n2.id
                    WHERE e2.source = ? AND e2.kind = 'contains'
                    LIMIT 20
                """
                children = _cg_query(_cg_db, children_sql, (sym_id,))

                # 组装结果
                entry = f"## [{sym['kind']}] {name}  ({fpath}:{lines_range})"

                if source_preview:
                    entry += f"\n\n**Source:**\n{source_preview}"

                if callers:
                    caller_lines = [f"  - {c['name']} ({c['file_path']}:{c['start_line']}) [{c['kind']}]" for c in callers]
                    entry += f"\n\n**Called by ({len(callers)}):**\n" + "\n".join(caller_lines)

                if callees:
                    callee_lines = [f"  - {c['name']} ({c['file_path']}:{c['start_line']}) [{c['kind']}]" for c in callees]
                    entry += f"\n\n**Calls ({len(callees)}):**\n" + "\n".join(callee_lines)

                if children:
                    child_lines = [f"  - [{c['kind']}] {c['name']} L{c['start_line']}-{c['end_line']}" for c in children]
                    entry += f"\n\n**Contains ({len(children)}):**\n" + "\n".join(child_lines)

                results.append(entry)

            return "\n\n---\n\n".join(results)

        except Exception as e:
            return f"codegraph_explore error: {e}"

    @server.tool(title="codegraph_list_symbols", annotations={"readOnlyHint": True})
    def codegraph_list_symbols(file_path: str) -> str:
        """列出指定文件中的所有符号"""
        err = _cg_available()
        if err:
            return err
        try:
            sql = """
                SELECT kind, name, qualified_name, start_line, end_line, visibility, return_type
                FROM nodes
                WHERE file_path = ? AND kind != 'file'
                ORDER BY start_line
            """
            rows = _cg_query(_cg_db, sql, (file_path,))
            if not rows:
                return f"No symbols found in: {file_path}"
            lines = []
            for r in rows:
                vis = r.get("visibility") or ""
                ret = r.get("return_type") or ""
                vis_str = f" [{vis}]" if vis else ""
                ret_str = f" → {ret}" if ret else ""
                lines.append(
                    f"L{r['start_line']}-{r['end_line']}: [{r['kind']}] {r['name']}{vis_str}{ret_str}"
                )
            return f"Symbols in {file_path} ({len(rows)}):\n\n" + "\n".join(lines)
        except Exception as e:
            return f"codegraph_list_symbols error: {e}"

    # --- Websearch ---

    @server.tool(title="web_search", annotations={"readOnlyHint": True})
    def web_search(query: str, num_results: int = 5) -> str:
        ddgs_cls = None
        for mod, name in (("ddgs", "DDGS"), ("duckduckgo_search", "DDGS")):
            try:
                ddgs_cls = getattr(__import__(mod, fromlist=[name]), name)
                break
            except ImportError:
                continue
        if ddgs_cls is None:
            return "Error: install web search: pip install ddgs"
        try:
            with ddgs_cls(proxy=http_proxy or None) as ddgs:
                results = list(ddgs.text(query, max_results=num_results))
                if not results:
                    return f"No results found for: {query}"
                lines = []
                for i, r in enumerate(results, 1):
                    lines.append(
                        f"{i}. **{r.get('title', '')}**\n"
                        f"   {r.get('body', '')}\n"
                        f"   URL: {r.get('href', '')}"
                    )
                return "\n\n".join(lines)
        except Exception as e:
            return f"Search error: {e}"

    @server.tool(title="fetch_url", annotations={"readOnlyHint": True})
    def fetch_url(url: str) -> str:
        try:
            resp = httpx.get(url, timeout=15, follow_redirects=True, proxy=http_proxy or None)
            if resp.status_code != 200:
                return f"HTTP {resp.status_code}"
            ct = resp.headers.get("content-type", "")
            if "text" in ct or "html" in ct or "json" in ct:
                text = resp.text
                if len(text) > 5000:
                    return text[:5000] + f"\n\n... (truncated, {len(text)} chars total)"
                return text
            return f"Fetched {len(resp.content)} bytes (content-type: {ct})"
        except Exception as e:
            return f"Fetch error: {e}"

    # --- Git ---

    def _git(args: list[str], timeout: int = 30) -> str:
        """运行 git 命令"""
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            if result.returncode != 0 and not output.strip():
                return f"git error (exit {result.returncode}): {result.stderr.strip()}"
            return output.strip() or "(no output)"
        except FileNotFoundError:
            return "Error: git not installed"
        except subprocess.TimeoutExpired:
            return f"Error: git timed out ({timeout}s)"
        except Exception as e:
            return f"Error: {e}"

    @server.tool(title="git_status", annotations={"readOnlyHint": True})
    def git_status() -> str:
        """查看工作区状态（修改/新增/删除的文件）"""
        return _git(["status", "--short"])

    @server.tool(title="git_log", annotations={"readOnlyHint": True})
    def git_log(
        count: Annotated[int, Field(description="显示最近 N 条提交", ge=1, le=50)] = 10,
    ) -> str:
        """查看提交历史"""
        return _git(["log", f"--oneline", f"-{count}", "--decorate"])

    @server.tool(title="git_diff", annotations={"readOnlyHint": True})
    def git_diff(
        file_path: Annotated[str, Field(description="文件路径（可选，不传则显示所有变更）")] = "",
    ) -> str:
        """查看文件差异（工作区 vs HEAD）"""
        args = ["diff", "--stat"]
        if file_path:
            args.append(file_path)
        stat = _git(args)
        # 也显示实际 diff（限制长度）
        diff_args = ["diff"]
        if file_path:
            diff_args.append(file_path)
        full_diff = _git(diff_args)
        if len(full_diff) > 4000:
            full_diff = full_diff[:4000] + f"\n\n... (truncated, total {len(full_diff)} chars)"
        return f"=== Status ===\n{stat}\n\n=== Diff ===\n{full_diff}" if stat else full_diff

    @server.tool(title="git_diff_staged", annotations={"readOnlyHint": True})
    def git_diff_staged() -> str:
        """查看已暂存（staged）的差异"""
        return _git(["diff", "--cached", "--stat"])

    @server.tool(title="git_blame", annotations={"readOnlyHint": True})
    def git_blame(
        file_path: Annotated[str, Field(description="文件路径")],
        start_line: Annotated[int, Field(description="起始行号", ge=1)] = 0,
        end_line: Annotated[int, Field(description="结束行号", ge=0)] = 0,
    ) -> str:
        """查看文件的 git blame（逐行注释）"""
        target = _safe_path(file_path)
        if not target.is_file():
            return f"Error: file not found: {file_path}"
        args = ["blame", "--porcelain"]
        if start_line > 0 and end_line > 0:
            args += [f"-L{start_line},{end_line}"]
        args.append(str(target))
        output = _git(args, timeout=15)
        if "fatal: not part of the repository" in output or "no such path" in output:
            return f"Error: {file_path} is not tracked by git (untracked file cannot be blamed)"
        if len(output) > 4000:
            output = output[:4000] + f"\n... (truncated)"
        return output

    @server.tool(title="git_commit", annotations={"readOnlyHint": False, "destructive_hint": True})
    def git_commit(
        message: Annotated[str, Field(description="提交信息")],
        files: Annotated[str, Field(description="要暂存的文件（空格分隔），默认全部")] = "",
    ) -> str:
        """暂存文件并提交"""
        if files:
            file_list = files.split()
            for f in file_list:
                result = _git(["add", f])
                if result.startswith("Error") or result.startswith("git error"):
                    return f"Failed to stage {f}: {result}"
        else:
            result = _git(["add", "-A"])
            if result.startswith("Error") or result.startswith("git error"):
                return f"Failed to stage: {result}"
        return _git(["commit", "-m", message])

    @server.tool(title="git_branch", annotations={"readOnlyHint": True})
    def git_branch() -> str:
        """列出所有分支"""
        return _git(["branch", "-a"])

    @server.tool(title="git_checkout", annotations={"readOnlyHint": False, "destructive_hint": True})
    def git_checkout(
        branch: Annotated[str, Field(description="分支名称或 commit hash")],
    ) -> str:
        """切换分支"""
        return _git(["checkout", branch])

    # --- Test Runner ---

    def _detect_test_framework() -> str:
        """检测项目使用的测试框架"""
        indicators = {
            "pytest": ["pytest.ini", "pyproject.toml", "setup.cfg", "conftest.py"],
            "unittest": ["test_*.py"],
        }
        # 检查配置文件
        for fw, files in indicators.items():
            for f in files:
                if f.endswith("*"):
                    # 通配符模式
                    pattern = f
                    import glob as globmod
                    if globmod.glob(str(root / "**" / pattern), recursive=True):
                        return fw
                elif (root / f).exists():
                    return fw
        # 检查 pyproject.toml [tool.pytest]
        pyproject = root / "pyproject.toml"
        if pyproject.exists():
            content = pyproject.read_text(errors="ignore")
            if "[tool.pytest" in content:
                return "pytest"
        return "pytest"  # 默认用 pytest

    @server.tool(title="run_test", annotations={"readOnlyHint": False})
    def run_test(
        path: Annotated[str, Field(description="测试文件或目录路径（空=全部）")] = "",
        marker: Annotated[str, Field(description="pytest marker 表达式，如 'not slow'")] = "",
        extra_args: Annotated[str, Field(description="额外参数，如 '-v --tb=short'")] = "",
    ) -> str:
        """运行测试并返回结构化结果（通过/失败/错误详情）"""
        framework = _detect_test_framework()
        target = _safe_path(path) if path else root

        if framework == "pytest":
            cmd = [sys.executable, "-m", "pytest"]
            if path:
                cmd.append(str(target))
            if marker:
                cmd += ["-m", marker]
            cmd += ["--tb=short", "-q", "--no-header"]
            if extra_args:
                cmd += extra_args.split()
        else:
            cmd = [sys.executable, "-m", "unittest"]
            if path:
                cmd.append(str(target))
            else:
                cmd += ["discover"]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = result.stdout
            if result.stderr:
                # stderr 中的 pytest 输出有时很重要
                stderr = result.stderr.strip()
                if stderr and "error" in stderr.lower():
                    output += f"\n[stderr]\n{stderr}"

            # 结构化解析
            lines = []
            if "passed" in output or "failed" in output or "error" in output:
                # 提取摘要行
                for line in output.split("\n"):
                    if any(kw in line for kw in ["passed", "failed", "error", "warnings", "FAILED", "ERROR"]):
                        lines.append(line.strip())
                summary = "\n".join(lines) if lines else output[-500:]
            else:
                summary = output[-500:] if len(output) > 500 else output

            status = "✅ PASS" if result.returncode == 0 else "❌ FAIL"
            return f"Exit code: {result.returncode} ({status})\n\n{summary}"
        except FileNotFoundError:
            return f"Error: {framework} not installed. Run: pip install {framework}"
        except subprocess.TimeoutExpired:
            return f"Error: tests timed out (120s limit)"
        except Exception as e:
            return f"Error: {e}"

    def _lsp_run(args: list[str], timeout: int = 30) -> str:
        """运行 basedpyright CLI 命令"""
        try:
            cmd = [sys.executable, "-m", "basedpyright"] + args
            result = subprocess.run(
                cmd, cwd=str(root), capture_output=True, text=True, timeout=timeout,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            return output.strip() or "(no output)"
        except FileNotFoundError:
            return "Error: basedpyright not installed. Run: pip install basedpyright"
        except subprocess.TimeoutExpired:
            return f"Error: basedpyright timed out ({timeout}s)"
        except Exception as e:
            return f"Error: {e}"

    @server.tool(title="lsp_diagnostics", annotations={"readOnlyHint": True})
    def lsp_diagnostics(file_path: str) -> str:
        """获取文件的类型诊断（错误、警告）"""
        target = _safe_path(file_path)
        if not target.is_file():
            return f"Error: file not found: {file_path}"
        return _lsp_run(["--outputjson", str(target)])

    @server.tool(title="lsp_goto_definition", annotations={"readOnlyHint": True})
    def lsp_goto_definition(file_path: str, line: int, character: int) -> str:
        """跳转到符号定义（返回位置信息）"""
        target = _safe_path(file_path)
        if not target.is_file():
            return f"Error: file not found: {file_path}"
        # basedpyright 没有 goto-definition CLI，用 codegraph 替代
        if _cg_db:
            try:
                sql = """
                    SELECT n.name, n.kind, n.file_path, n.start_line, n.end_line
                    FROM nodes n
                    WHERE n.file_path = ? AND n.start_line <= ? AND n.end_line >= ?
                    LIMIT 5
                """
                rows = _cg_query(_cg_db, sql, (file_path, line, line))
                if rows:
                    lines = [f"[{r['kind']}] {r['name']} at {r['file_path']}:{r['start_line']}-{r['end_line']}" for r in rows]
                    return f"Definitions at line {line}:\n" + "\n".join(lines)
                return f"No symbols found at line {line}"
            except Exception as e:
                return f"codegraph fallback error: {e}"
        return "Error: neither basedpyright nor codegraph available for goto-definition"

    @server.tool(title="lsp_find_references", annotations={"readOnlyHint": True})
    def lsp_find_references(file_path: str, line: int, character: int) -> str:
        """查找符号的所有引用"""
        target = _safe_path(file_path)
        if not target.is_file():
            return f"Error: file not found: {file_path}"
        if _cg_db:
            try:
                # 先找到该位置的符号
                sym_sql = """
                    SELECT id, name, kind FROM nodes
                    WHERE file_path = ? AND start_line <= ? AND end_line >= ?
                    AND kind IN ('class', 'function', 'method', 'variable')
                    LIMIT 1
                """
                sym_rows = _cg_query(_cg_db, sym_sql, (file_path, line, line))
                if not sym_rows:
                    return f"No symbol found at line {line}"
                sym = sym_rows[0]
                # 查找引用
                ref_sql = """
                    SELECT n.name, n.file_path, n.start_line, e.kind
                    FROM edges e JOIN nodes n ON e.source = n.id
                    WHERE e.target = ? AND e.kind = 'references'
                    LIMIT 20
                """
                refs = _cg_query(_cg_db, ref_sql, (sym["id"],))
                header = f"References to [{sym['kind']}] {sym['name']}:\n"
                if not refs:
                    return header + "(no references found)"
                lines = [f"  - {r['name']} at {r['file_path']}:{r['start_line']} [{r['kind']}]" for r in refs]
                return header + "\n".join(lines)
            except Exception as e:
                return f"codegraph fallback error: {e}"
        return "Error: codegraph not available for find-references"

    @server.tool(title="lsp_rename", annotations={"readOnlyHint": True})
    def lsp_rename(file_path: str, line: int, character: int, new_name: str) -> str:
        """重命名符号（返回需要修改的位置列表，不自动修改）"""
        target = _safe_path(file_path)
        if not target.is_file():
            return f"Error: file not found: {file_path}"
        if _cg_db:
            try:
                sym_sql = """
                    SELECT id, name, kind FROM nodes
                    WHERE file_path = ? AND start_line <= ? AND end_line >= ?
                    AND kind IN ('class', 'function', 'method', 'variable')
                    LIMIT 1
                """
                sym_rows = _cg_query(_cg_db, sym_sql, (file_path, line, line))
                if not sym_rows:
                    return f"No symbol found at line {line}"
                sym = sym_rows[0]
                ref_sql = """
                    SELECT n.name, n.file_path, n.start_line
                    FROM edges e JOIN nodes n ON e.source = n.id
                    WHERE e.target = ?
                    LIMIT 50
                """
                all_refs = _cg_query(_cg_db, ref_sql, (sym["id"],))
                header = f"Rename [{sym['kind']}] {sym['name']} → {new_name}:\n"
                header += f"(rename is advisory — apply changes manually)\n\n"
                header += f"Original: {file_path}:{line}\n"
                locations = [f"  - {r['file_path']}:{r['start_line']}" for r in all_refs]
                return header + "\n".join(locations) if locations else header + "(no references found)"
            except Exception as e:
                return f"codegraph fallback error: {e}"
        return "Error: codegraph not available for rename"

    @server.tool(title="lsp_document_symbols", annotations={"readOnlyHint": True})
    def lsp_document_symbols(file_path: str) -> str:
        """列出文档中的所有符号（使用 codegraph 数据）"""
        target = _safe_path(file_path)
        if not target.is_file():
            return f"Error: file not found: {file_path}"
        if _cg_db:
            return codegraph_list_symbols(file_path)
        # fallback: 用 basedpyright --outputjson 解析
        output = _lsp_run(["--outputjson", str(target)])
        if output.startswith("Error:"):
            return output
        try:
            data = json.loads(output)
            diagnostics = data.get("generalDiagnostics", [])
            if not diagnostics:
                return f"No issues found in {file_path}"
            lines = []
            for d in diagnostics[:30]:
                range_ = d.get("range", {})
                start = range_.get("start", {})
                lines.append(
                    f"L{start.get('line', 0)+1}:{start.get('character', 0)} "
                    f"[{d.get('severity', '?')}] {d.get('message', '')}"
                )
            return f"Diagnostics for {file_path} ({len(diagnostics)} total):\n" + "\n".join(lines)
        except json.JSONDecodeError:
            return output[:2000]

    @server.tool(
        title="clone_and_index",
        annotations={"readOnlyHint": False, "destructiveHint": False, "openWorldHint": True},
    )
    def clone_and_index(
        github_url: Annotated[str, Field(description="GitHub repo URL to clone")],
        target_dir: Annotated[str, Field(description="Target directory name (optional)", default="")],
    ) -> str:
        """Clone a GitHub repo and index it with codegraph"""
        import re as _re

        url_match = _re.match(r'https?://github\.com/([^/]+/[^/.]+)', github_url)
        if not url_match:
            return "Error: invalid GitHub URL. Expected format: https://github.com/user/repo"

        repo_name = url_match.group(1).replace("/", "_")
        if target_dir:
            repo_name = target_dir

        clone_dir = Path(project_path or ".") / "repos" / repo_name

        if clone_dir.exists():
            return f"Directory already exists: {clone_dir}. Use list_files to explore."

        try:
            clone_dir.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                ["git", "clone", "--depth=1", github_url, str(clone_dir)],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=project_path or os.getcwd(),
            )
            if result.returncode != 0:
                return f"Error cloning: {result.stderr.strip()}"

            try:
                index_result = subprocess.run(
                    ["codegraph", "init"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=str(clone_dir),
                )
                index_status = "indexed" if index_result.returncode == 0 else "not indexed"
            except FileNotFoundError:
                index_status = "codegraph not installed, skipped indexing"

            file_count = sum(1 for _ in clone_dir.rglob("*") if _.is_file())
            py_files = list(clone_dir.rglob("*.py"))[:10]
            ts_files = list(clone_dir.rglob("*.ts"))[:10]
            go_files = list(clone_dir.rglob("*.go"))[:10]
            rs_files = list(clone_dir.rglob("*.rs"))[:10]

            summary = [
                f"✅ Cloned to: {clone_dir}",
                f"Files: {file_count}",
                f"Index: {index_status}",
                "",
                "Language breakdown:",
                f"  Python: {len(list(clone_dir.rglob('*.py')))} files" if py_files else "",
                f"  TypeScript: {len(list(clone_dir.rglob('*.ts')))} files" if ts_files else "",
                f"  Go: {len(list(clone_dir.rglob('*.go')))} files" if go_files else "",
                f"  Rust: {len(list(clone_dir.rglob('*.rs')))} files" if rs_files else "",
            ]

            readme = clone_dir / "README.md"
            if readme.exists():
                summary.append(f"\nREADME: {readme.read_text(errors='ignore')[:500]}")

            return "\n".join(filter(None, summary))

        except subprocess.TimeoutExpired:
            return "Error: clone timed out (repo too large?)"
        except Exception as e:
            return f"Error: {e}"

    # ── Skill 工具 ──────────────────────────────────────────────────────

    @server.tool(title="list_skills")
    def list_skills() -> str:
        """列出所有可用的 Skill（内置 + 用户自定义）。

        Returns:
            JSON 字符串，包含所有已注册 skill 的名称、描述、参数、来源
        """
        try:
            from src.skills.registry import SkillRegistry

            registry = SkillRegistry()
            registry.scan()

            result = []
            for skill in registry.skills.values():
                result.append({
                    "name": skill.name,
                    "description": skill.description,
                    "version": skill.version,
                    "source": skill.source,
                    "parameters": {k: {"type": v.type, "description": v.description, "required": v.required} for k, v in skill.parameters.items()},
                })

            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    @server.tool(title="run_skill")
    def run_skill(
        skill_name: Annotated[str, Field(description="Skill 名称，如 'weather', 'docx_template'")],
        args: Annotated[dict, Field(description="Skill 参数，如 {\"city\": \"北京\"}")] = {},
    ) -> str:
        """执行指定的 Skill。

        Args:
            skill_name: Skill 名称（从 list_skills 获取）
            args: Skill 参数字典

        Returns:
            JSON 字符串，包含执行结果或错误信息
        """
        try:
            from src.skills.runner import run_skill as _run_skill

            result = _run_skill(skill_name, args, timeout=60)
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    @server.tool(title="create_skill")
    def create_skill(
        name: Annotated[str, Field(description="Skill 名称（仅字母和下划线）")],
        description: Annotated[str, Field(description="Skill 描述")],
        parameters: Annotated[dict, Field(description="参数定义，如 {\"city\": {\"type\": \"string\", \"description\": \"城市名称\", \"required\": true}}")],
        entry_code: Annotated[str, Field(description="Python 入口代码，定义 run() 函数")],
    ) -> str:
        """创建新的 Skill（保存到 skills/ 目录）。

        Args:
            name: Skill 名称（仅字母和下划线）
            description: Skill 描述
            parameters: 参数定义字典
            entry_code: Python 代码（必须定义 run() 函数）

        Returns:
            JSON 字符串，包含创建结果
        """
        import re
        from pathlib import Path

        # 验证名称
        if not re.match(r"^[a-z][a-z0-9_]*$", name):
            return json.dumps({
                "error": "Skill 名称只能包含小写字母、数字和下划线，且以字母开头",
            }, ensure_ascii=False)

        if not entry_code.strip():
            return json.dumps({"error": "entry_code 不能为空"}, ensure_ascii=False)

        skill_dir = Path("skills") / name

        try:
            skill_dir.mkdir(parents=True, exist_ok=True)

            # 写入 manifest.json
            manifest = {
                "name": name,
                "description": description,
                "version": "1.0.0",
                "parameters": parameters,
                "dependencies": [],
                "entry": "main.py",
            }
            (skill_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # 写入 main.py
            (skill_dir / "main.py").write_text(entry_code, encoding="utf-8")

            # 验证可以加载
            from src.skills.registry import SkillRegistry

            registry = SkillRegistry()
            registry.scan()
            skill = registry.get_skill(name)

            return json.dumps({
                "status": "ok",
                "name": name,
                "path": str(skill_dir / "main.py"),
                "manifest": str(skill_dir / "manifest.json"),
                "message": "Skill 创建成功",
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    return server
