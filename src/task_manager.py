import os
import json
import uuid
import asyncio
import logging
from typing import Optional, Any, List
from datetime import datetime
from pathlib import Path
from enum import Enum
from pydantic import BaseModel, Field
from dataclasses import dataclass, field, asdict
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from .capabilities.base import CapabilityRegistry, CapabilityOutput


logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class TriggerType(str, Enum):
    MANUAL = "manual"
    CRON = "cron"
    INTERVAL = "interval"
    ONCE = "once"


@dataclass
class TaskStep:
    capability: str
    params: dict = field(default_factory=dict)


@dataclass
class TaskTrigger:
    type: TriggerType = TriggerType.MANUAL
    cron_expr: Optional[str] = None
    interval_seconds: Optional[int] = None
    run_at: Optional[str] = None


@dataclass
class StepLog:
    step_index: int
    capability: str
    params: dict
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: int = 0
    started_at: str = ""
    finished_at: str = ""


@dataclass
class ExecutionLog:
    id: str
    task_id: str
    task_name: str
    started_at: str
    finished_at: Optional[str] = None
    success: bool = False
    status: str = "running"
    step_logs: list[StepLog] = field(default_factory=list)
    final_output: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        def step_to_dict(s):
            if isinstance(s, dict):
                return s
            return {
                "step_index": s.step_index,
                "capability": s.capability,
                "params": s.params,
                "success": s.success,
                "output": s.output,
                "error": s.error,
                "duration_ms": s.duration_ms,
                "started_at": s.started_at,
                "finished_at": s.finished_at,
            }

        return {
            "id": self.id,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "success": self.success,
            "status": self.status,
            "step_logs": [step_to_dict(s) for s in self.step_logs],
            "final_output": self.final_output,
            "error": self.error,
        }


@dataclass
class Task:
    id: str
    name: str
    description: str = ""
    steps: list[TaskStep] = field(default_factory=list)
    trigger: TaskTrigger = field(default_factory=TaskTrigger)
    status: TaskStatus = TaskStatus.ACTIVE
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    output_template: str = ""
    notification_enabled: bool = False
    notification_channels: list[str] = field(default_factory=list)
    last_execution_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": [
                {"capability": s.capability, "params": s.params} for s in self.steps
            ],
            "trigger": asdict(self.trigger),
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_run": self.last_run,
            "next_run": self.next_run,
            "output_template": self.output_template,
            "notification_enabled": self.notification_enabled,
            "notification_channels": self.notification_channels,
            "last_execution_id": self.last_execution_id,
        }


class TaskManager:
    _instance = None
    _scheduler: Optional[BackgroundScheduler] = None
    _tasks: dict[str, Task] = {}
    _execution_logs: dict[str, ExecutionLog] = {}
    _data_path: Path = Path.home() / ".opencode_helper" / "tasks.json"
    _logs_path: Path = Path.home() / ".opencode_helper" / "logs"
    _outputs_path: Path = Path.home() / ".opencode_helper" / "outputs"
    _max_logs_per_task: int = 10
    _max_total_logs: int = 100

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
            cls._instance._load_logs()
            cls._instance._start_scheduler()
        return cls._instance

    def _load(self):
        self._data_path.parent.mkdir(parents=True, exist_ok=True)
        if self._data_path.exists():
            try:
                with open(self._data_path, "r") as f:
                    data = json.load(f)
                    for t in data.get("tasks", []):
                        steps = [TaskStep(**s) for s in t.get("steps", [])]
                        trigger = TaskTrigger(**t.get("trigger", {}))
                        task = Task(
                            id=t["id"],
                            name=t["name"],
                            description=t.get("description", ""),
                            steps=steps,
                            trigger=trigger,
                            status=TaskStatus(t.get("status", "active")),
                            created_at=t.get("created_at", datetime.now().isoformat()),
                            updated_at=t.get("updated_at", datetime.now().isoformat()),
                            last_run=t.get("last_run"),
                            next_run=t.get("next_run"),
                            output_template=t.get("output_template", ""),
                            notification_enabled=t.get("notification_enabled", False),
                            notification_channels=t.get("notification_channels", []),
                            last_execution_id=t.get("last_execution_id"),
                        )
                        self._tasks[task.id] = task
            except Exception as e:
                logger.error(f"Failed to load tasks: {e}")

    def _load_logs(self):
        self._logs_path.mkdir(parents=True, exist_ok=True)
        self._outputs_path.mkdir(parents=True, exist_ok=True)
        log_files = sorted(
            self._logs_path.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:50]
        for log_file in log_files:
            try:
                with open(log_file, "r") as f:
                    log_data = json.load(f)
                    self._execution_logs[log_data["id"]] = ExecutionLog(**log_data)
            except Exception as e:
                logger.warning(f"Failed to load log {log_file}: {e}")

    def _save(self):
        data = {"tasks": [t.to_dict() for t in self._tasks.values()]}
        with open(self._data_path, "w") as f:
            json.dump(data, f, indent=2)

    def _save_log(self, log: ExecutionLog):
        log_file = self._logs_path / f"{log.id}.json"
        with open(log_file, "w") as f:
            json.dump(log.to_dict(), f, indent=2)

        if log.final_output:
            output_file = self._outputs_path / f"{log.id}.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(log.final_output)

        self._execution_logs[log.id] = log

        self._cleanup_old_logs(log.task_id)

    def _cleanup_old_logs(self, task_id: Optional[str] = None):
        if task_id:
            task_logs = [
                (log_id, log)
                for log_id, log in self._execution_logs.items()
                if log.task_id == task_id
            ]
            task_logs.sort(key=lambda x: x[1].started_at, reverse=True)

            if len(task_logs) > self._max_logs_per_task:
                logs_to_delete = task_logs[self._max_logs_per_task :]
                for log_id, _ in logs_to_delete:
                    self._delete_log(log_id)

        all_logs = sorted(
            self._execution_logs.items(), key=lambda x: x[1].started_at, reverse=True
        )
        if len(all_logs) > self._max_total_logs:
            logs_to_delete = all_logs[self._max_total_logs :]
            for log_id, _ in logs_to_delete:
                self._delete_log(log_id)

    def _delete_log(self, log_id: str):
        if log_id in self._execution_logs:
            del self._execution_logs[log_id]

        log_file = self._logs_path / f"{log_id}.json"
        if log_file.exists():
            log_file.unlink()

        output_file = self._outputs_path / f"{log_id}.txt"
        if output_file.exists():
            output_file.unlink()

        logger.info(f"Deleted old log: {log_id}")

    def _start_scheduler(self):
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR

        self._scheduler = BackgroundScheduler()

        def job_executed_listener(event):
            if event.exception:
                logger.error(f"Scheduled job failed: {event.exception}")
            else:
                logger.info(f"Scheduled job completed: {event.job_id}")

        self._scheduler.add_listener(
            job_executed_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
        )
        self._scheduler.start()
        logger.info("Task scheduler started")

        for task in self._tasks.values():
            if (
                task.status == TaskStatus.ACTIVE
                and task.trigger.type != TriggerType.MANUAL
            ):
                self._schedule_task(task)

    def _schedule_task(self, task: Task):
        if not self._scheduler:
            return

        job_id = f"task_{task.id}"
        logger.info(
            f"Scheduling task: {task.name} ({job_id}) with trigger: {task.trigger.type}"
        )

        def sync_run_task(task_id):
            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.run_task(task_id))
            finally:
                loop.close()

        if task.trigger.type == TriggerType.CRON and task.trigger.cron_expr:
            try:
                self._scheduler.add_job(
                    sync_run_task,
                    CronTrigger.from_crontab(task.trigger.cron_expr),
                    args=[task.id],
                    id=job_id,
                    replace_existing=True,
                )
                logger.info(f"Cron task scheduled: {task.trigger.cron_expr}")
            except Exception as e:
                logger.error(f"Failed to schedule cron task: {e}")

        elif (
            task.trigger.type == TriggerType.INTERVAL and task.trigger.interval_seconds
        ):
            try:
                self._scheduler.add_job(
                    sync_run_task,
                    IntervalTrigger(seconds=task.trigger.interval_seconds),
                    args=[task.id],
                    id=job_id,
                    replace_existing=True,
                )
                logger.info(
                    f"Interval task scheduled: every {task.trigger.interval_seconds}s"
                )
            except Exception as e:
                logger.error(f"Failed to schedule interval task: {e}")

    def create_task(
        self,
        name: str,
        description: str = "",
        steps: list[dict] = None,
        trigger: dict = None,
        output_template: str = "",
        notification_enabled: bool = False,
        notification_channels: list[str] = None,
    ) -> Task:
        task_id = str(uuid.uuid4())[:8]
        task = Task(
            id=task_id,
            name=name,
            description=description,
            steps=[TaskStep(**s) for s in (steps or [])],
            trigger=TaskTrigger(**(trigger or {"type": "manual"})),
            output_template=output_template,
            notification_enabled=notification_enabled,
            notification_channels=notification_channels or [],
        )
        self._tasks[task_id] = task
        self._save()

        if task.trigger.type != TriggerType.MANUAL:
            self._schedule_task(task)

        return task

    def update_task(self, task_id: str, **kwargs) -> Optional[Task]:
        task = self._tasks.get(task_id)
        if not task:
            return None

        if "name" in kwargs:
            task.name = kwargs["name"]
        if "description" in kwargs:
            task.description = kwargs["description"]
        if "steps" in kwargs:
            task.steps = [TaskStep(**s) for s in kwargs["steps"]]
        if "trigger" in kwargs:
            task.trigger = TaskTrigger(**kwargs["trigger"])
        if "output_template" in kwargs:
            task.output_template = kwargs["output_template"]
        if "status" in kwargs:
            task.status = TaskStatus(kwargs["status"])
        if "notification_enabled" in kwargs:
            task.notification_enabled = kwargs["notification_enabled"]
        if "notification_channels" in kwargs:
            task.notification_channels = kwargs["notification_channels"]

        task.updated_at = datetime.now().isoformat()
        self._save()

        if task.trigger.type != TriggerType.MANUAL:
            job_id = f"task_{task.id}"
            if self._scheduler:
                self._scheduler.remove_job(job_id)
            self._schedule_task(task)

        return task

    def delete_task(self, task_id: str) -> bool:
        if task_id in self._tasks:
            task = self._tasks[task_id]
            job_id = f"task_{task.id}"
            if self._scheduler:
                try:
                    self._scheduler.remove_job(job_id)
                except:
                    pass
            del self._tasks[task_id]
            self._save()
            return True
        return False

    def get_task(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def list_tasks(self, status: Optional[TaskStatus] = None) -> list[Task]:
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    def get_execution_logs(
        self, task_id: Optional[str] = None, limit: int = 50
    ) -> list[dict]:
        logs = list(self._execution_logs.values())
        if task_id:
            logs = [l for l in logs if l.task_id == task_id]
        logs.sort(key=lambda l: l.started_at, reverse=True)
        return [l.to_dict() for l in logs[:limit]]

    def get_log(self, log_id: str) -> Optional[dict]:
        log = self._execution_logs.get(log_id)
        return log.to_dict() if log else None

    def get_output_file(self, log_id: str) -> Optional[str]:
        output_file = self._outputs_path / f"{log_id}.txt"
        if output_file.exists():
            return str(output_file)
        return None

    async def run_task(self, task_id: str, use_output_template: bool = True) -> dict:
        task = self._tasks.get(task_id)
        if not task:
            return {"success": False, "error": "Task not found"}

        exec_id = str(uuid.uuid4())[:8]
        started_at = datetime.now().isoformat()

        exec_log = ExecutionLog(
            id=exec_id,
            task_id=task_id,
            task_name=task.name,
            started_at=started_at,
            status="running",
        )
        self._save_log(exec_log)

        task.last_run = started_at
        task.last_execution_id = exec_id
        self._save()

        step_outputs = {}

        for i, step in enumerate(task.steps):
            step_started = datetime.now().isoformat()
            start_time = asyncio.get_event_loop().time()

            logger.info(f"[Task {task_id}] Executing step {i + 1}: {step.capability}")
            logger.info(f"[Task {task_id}] Params: {json.dumps(step.params)}")

            result = await CapabilityRegistry.execute(step.capability, step.params)
            duration = int((asyncio.get_event_loop().time() - start_time) * 1000)

            step_log = StepLog(
                step_index=i,
                capability=step.capability,
                params=step.params,
                success=result.success,
                output=result.data,
                error=result.error,
                duration_ms=duration,
                started_at=step_started,
                finished_at=datetime.now().isoformat(),
            )
            exec_log.step_logs.append(step_log)
            step_outputs[step.capability] = result.data

            logger.info(
                f"[Task {task_id}] Step {i + 1} result: success={result.success}, duration={duration}ms"
            )

            if not result.success:
                exec_log.status = "failed"
                exec_log.error = result.error
                task.status = TaskStatus.FAILED
                break

        finished_at = datetime.now().isoformat()
        exec_log.finished_at = finished_at

        if (
            use_output_template
            and task.output_template
            and all(s.success for s in exec_log.step_logs)
        ):
            exec_log.final_output = self._format_output(
                task.output_template, step_outputs
            )
        elif exec_log.step_logs:
            exec_log.final_output = self._format_results(exec_log.step_logs)

        exec_log.success = all(s.success for s in exec_log.step_logs)
        exec_log.status = "completed" if exec_log.success else "failed"

        task.updated_at = finished_at
        task.status = TaskStatus.COMPLETED if exec_log.success else TaskStatus.FAILED

        self._save_log(exec_log)
        self._save()

        logger.info(
            f"[Task {task_id}] Execution {exec_id} completed: success={exec_log.success}"
        )

        result_dict = {
            "success": exec_log.success,
            "task_id": task_id,
            "task_name": task.name,
            "execution_id": exec_id,
            "results": [
                s.to_dict() if hasattr(s, "to_dict") else s for s in exec_log.step_logs
            ],
            "output": exec_log.final_output,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": sum(s.duration_ms for s in exec_log.step_logs),
        }

        if task.notification_enabled and exec_log.success:
            await self._send_notification(task, exec_log)

        return result_dict

    async def _send_notification(self, task: Task, log: ExecutionLog):
        if "telegram" in task.notification_channels:
            try:
                from .capabilities.notify import TelegramCapability

                cap = TelegramCapability()
                await cap.execute(
                    {"message": f"✅ 任务完成: {task.name}\n\n{log.final_output[:500]}"}
                )
            except Exception as e:
                logger.error(f"Telegram notification failed: {e}")

        if "webhook" in task.notification_channels:
            try:
                from .storage import secrets

                webhook_url = secrets.get("default_webhook_url")
                if webhook_url:
                    from .capabilities.webhook import WebhookCapability

                    cap = WebhookCapability()
                    await cap.execute(
                        {
                            "url": webhook_url,
                            "message": f"任务完成: {task.name}\n\n{log.final_output[:500]}",
                        }
                    )
            except Exception as e:
                logger.error(f"Webhook notification failed: {e}")

    def _format_output(self, template: str, outputs: dict) -> str:
        result = template
        for key, value in outputs.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    result = result.replace(f"{{{key}.{k}}}", str(v))
            elif isinstance(value, list):
                result = result.replace(
                    f"{{{key}}}", "\n".join(str(v) for v in value[:5])
                )
            else:
                result = result.replace(f"{{{key}}}", str(value))
        return result

    def _format_results(self, step_logs: list[StepLog]) -> str:
        parts = []
        for r in step_logs:
            if r.success:
                if isinstance(r.output, dict):
                    if "results" in r.output and isinstance(r.output["results"], list):
                        results = r.output["results"]
                        if results:
                            result_lines = []
                            for i, item in enumerate(results[:10], 1):
                                title = item.get("title", "无标题")[:60]
                                url = item.get("url", "")
                                snippet = item.get("snippet", "")[:100]
                                result_lines.append(f"{i}. {title}")
                                if snippet:
                                    result_lines.append(f"   {snippet}...")
                                if url:
                                    result_lines.append(f"   {url}")
                            parts.append(
                                f"[{r.capability}] 找到 {len(results)} 条结果:\n"
                                + "\n".join(result_lines)
                            )
                        else:
                            parts.append(f"[{r.capability}] 无结果")
                    elif "message" in r.output:
                        parts.append(f"[{r.capability}] {r.output['message']}")
                    elif "content" in r.output:
                        parts.append(f"[{r.capability}]\n{r.output['content']}")
                    else:
                        import json

                        parts.append(
                            f"[{r.capability}]\n{json.dumps(r.output, ensure_ascii=False, indent=2)}"
                        )
                elif isinstance(r.output, list):
                    parts.append(f"[{r.capability}] {len(r.output)} 项结果")
                else:
                    parts.append(f"[{r.capability}] {r.output}")
            else:
                parts.append(f"[{r.capability}] 失败: {r.error}")
        return "\n".join(parts)

    def pause_task(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task:
            task.status = TaskStatus.PAUSED
            self._save()
            job_id = f"task_{task.id}"
            if self._scheduler:
                try:
                    self._scheduler.remove_job(job_id)
                except:
                    pass
            return True
        return False

    def resume_task(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task:
            task.status = TaskStatus.ACTIVE
            self._save()
            self._schedule_task(task)
            return True
        return False


task_manager = TaskManager()
