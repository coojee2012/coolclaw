"""Background GGUF model downloader from HuggingFace."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

from .model_catalog import find_local_model

logger = logging.getLogger(__name__)


@dataclass
class DownloadTask:
  id: str
  model_id: str
  quant: str
  filename: str
  repo: str
  dest_path: str
  status: str = "pending"  # pending | downloading | completed | failed | cancelled
  progress: float = 0.0
  downloaded_bytes: int = 0
  total_bytes: int = 0
  speed_bps: float = 0.0
  error: str = ""
  created_at: float = field(default_factory=time.time)
  completed_at: Optional[float] = None

  def to_dict(self) -> dict:
    return {
      "id": self.id,
      "model_id": self.model_id,
      "quant": self.quant,
      "filename": self.filename,
      "repo": self.repo,
      "dest_path": self.dest_path,
      "status": self.status,
      "progress": round(self.progress, 1),
      "downloaded_bytes": self.downloaded_bytes,
      "total_bytes": self.total_bytes,
      "size_gb": round(self.total_bytes / (1024 ** 3), 2) if self.total_bytes else 0,
      "speed_mbps": round(self.speed_bps * 8 / (1024 ** 2), 2),
      "error": self.error,
      "created_at": self.created_at,
      "completed_at": self.completed_at,
    }


class ModelDownloadManager:
  _instance: Optional["ModelDownloadManager"] = None

  def __new__(cls):
    if cls._instance is None:
      cls._instance = super().__new__(cls)
      cls._instance._tasks: dict[str, DownloadTask] = {}
      cls._instance._lock = threading.Lock()
    return cls._instance

  def list_tasks(self) -> list[dict]:
    with self._lock:
      return [t.to_dict() for t in sorted(self._tasks.values(), key=lambda x: x.created_at, reverse=True)]

  def get_task(self, task_id: str) -> DownloadTask | None:
    with self._lock:
      return self._tasks.get(task_id)

  def start_download(self, model_id: str, quant: str, models_dir: str) -> DownloadTask:
    entry = find_local_model(model_id)
    if not entry:
      raise ValueError(f"Unknown model: {model_id}")

    variant = next((v for v in entry.variants if v.quant == quant), None)
    if not variant:
      variant = next((v for v in entry.variants if v.recommended), entry.variants[0])

    dest_dir = Path(models_dir).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / variant.filename

    if dest_path.exists() and dest_path.stat().st_size > 1024 * 1024:
      task = DownloadTask(
        id=str(uuid.uuid4())[:8],
        model_id=model_id,
        quant=variant.quant,
        filename=variant.filename,
        repo=entry.huggingface_repo,
        dest_path=str(dest_path),
        status="completed",
        progress=100.0,
        downloaded_bytes=dest_path.stat().st_size,
        total_bytes=dest_path.stat().st_size,
        completed_at=time.time(),
      )
      with self._lock:
        self._tasks[task.id] = task
      return task

    task = DownloadTask(
      id=str(uuid.uuid4())[:8],
      model_id=model_id,
      quant=variant.quant,
      filename=variant.filename,
      repo=entry.huggingface_repo,
      dest_path=str(dest_path),
    )
    with self._lock:
      self._tasks[task.id] = task

    thread = threading.Thread(target=self._run_download, args=(task,), daemon=True)
    thread.start()
    return task

  def start_hf_download(self, repo_id: str, filename: str, models_dir: str) -> DownloadTask:
    """Download a specific GGUF file from any HuggingFace repo."""
    dest_dir = Path(models_dir).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / Path(filename).name

    if dest_path.exists() and dest_path.stat().st_size > 1024 * 1024:
      task = DownloadTask(
        id=str(uuid.uuid4())[:8],
        model_id=repo_id,
        quant="",
        filename=Path(filename).name,
        repo=repo_id,
        dest_path=str(dest_path),
        status="completed",
        progress=100.0,
        downloaded_bytes=dest_path.stat().st_size,
        total_bytes=dest_path.stat().st_size,
        completed_at=time.time(),
      )
      with self._lock:
        self._tasks[task.id] = task
      return task

    task = DownloadTask(
      id=str(uuid.uuid4())[:8],
      model_id=repo_id,
      quant="",
      filename=filename,
      repo=repo_id,
      dest_path=str(dest_path),
    )
    with self._lock:
      self._tasks[task.id] = task
    thread = threading.Thread(target=self._run_download, args=(task,), daemon=True)
    thread.start()
    return task

  def cancel_download(self, task_id: str) -> bool:
    with self._lock:
      task = self._tasks.get(task_id)
      if not task or task.status not in ("pending", "downloading"):
        return False
      task.status = "cancelled"
      return True

  def _run_download(self, task: DownloadTask) -> None:
    from .config import get_httpx_proxy
    fn = task.filename
    url = f"https://huggingface.co/{task.repo}/resolve/main/{fn}"
    proxy = get_httpx_proxy()
    proxy_url = proxy.get("https://") if proxy else None
    task.status = "downloading"
    dest = Path(task.dest_path)
    tmp = dest.with_suffix(dest.suffix + ".part")

    try:
      with httpx.stream(
        "GET", url, follow_redirects=True,
        timeout=httpx.Timeout(30.0, read=120.0),
        proxy=proxy_url,
      ) as resp:
        if resp.status_code == 404:
          url = f"https://huggingface.co/{task.repo}/resolve/main/{Path(fn).name}"
          resp.close()
          with httpx.stream("GET", url, follow_redirects=True, timeout=httpx.Timeout(30.0, read=120.0), proxy=proxy_url) as resp2:
            self._stream_to_file(resp2, task, tmp, dest)
          return
        self._stream_to_file(resp, task, tmp, dest)
    except Exception as e:
      task.status = "failed"
      task.error = str(e)
      tmp.unlink(missing_ok=True)
      logger.error("Download failed for %s: %s", task.model_id, e)

  def _stream_to_file(self, resp, task: DownloadTask, tmp: Path, dest: Path) -> None:
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    task.total_bytes = total
    downloaded = 0
    last_time = time.time()
    last_bytes = 0
    with open(tmp, "wb") as f:
      for chunk in resp.iter_bytes(chunk_size=1024 * 256):
        if task.status == "cancelled":
          tmp.unlink(missing_ok=True)
          return
        f.write(chunk)
        downloaded += len(chunk)
        task.downloaded_bytes = downloaded
        now = time.time()
        if now - last_time >= 0.5:
          task.speed_bps = (downloaded - last_bytes) / (now - last_time)
          last_time = now
          last_bytes = downloaded
        if total > 0:
          task.progress = downloaded / total * 100
    tmp.rename(dest)
    task.status = "completed"
    task.progress = 100.0
    task.completed_at = time.time()
    logger.info("Download completed: %s", dest)


download_manager = ModelDownloadManager()
