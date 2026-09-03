import os
import asyncio
from pathlib import Path
from typing import Optional
from .base import capability, BaseCapability, CapabilityOutput, CapabilityCategory


@capability(
    name="file_watch",
    description="监控指定目录的新文件或文件变化",
    category=CapabilityCategory.FILE,
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "要监控的目录路径"},
            "pattern": {
                "type": "string",
                "description": "文件过滤模式 (如 *.pdf, *.txt)",
                "default": "*",
            },
            "recursive": {
                "type": "boolean",
                "description": "是否监控子目录",
                "default": False,
            },
        },
        "required": ["path"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "files": {"type": "array", "items": {"type": "string"}},
            "count": {"type": "integer"},
        },
    },
    memory_mb=20,
    examples=["监控下载文件夹的新PDF", "监听指定目录的所有图片"],
)
class FileWatchCapability(BaseCapability):
    _watchers = {}

    async def execute(self, params: dict) -> CapabilityOutput:
        try:
            import time
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            watch_path = params.get("path", "")
            pattern = params.get("pattern", "*")
            recursive = params.get("recursive", False)

            if not os.path.exists(watch_path):
                return CapabilityOutput(
                    success=False, error=f"路径不存在: {watch_path}"
                )

            class SimpleHandler(FileSystemEventHandler):
                def __init__(self):
                    self.new_files = []
                    self.changed_files = []

                def on_created(self, event):
                    if not event.is_directory:
                        self.new_files.append(event.src_path)

                def on_modified(self, event):
                    if not event.is_directory:
                        self.changed_files.append(event.src_path)

            handler = SimpleHandler()
            observer = Observer()
            observer.schedule(handler, watch_path, recursive=recursive)
            observer.start()

            try:
                time.sleep(2)
            finally:
                observer.stop()
                observer.join()

            all_files = list(
                Path(watch_path).rglob(pattern)
                if recursive
                else Path(watch_path).glob(pattern)
            )

            return CapabilityOutput(
                success=True,
                data={
                    "watched_path": watch_path,
                    "new_files": handler.new_files,
                    "changed_files": handler.changed_files,
                    "all_matching_files": [str(f) for f in all_files],
                    "count": len(all_files),
                },
            )
        except ImportError:
            return CapabilityOutput(
                success=False, error="需要安装 watchdog: pip install watchdog"
            )
        except Exception as e:
            return CapabilityOutput(success=False, error=f"监控失败: {str(e)}")
