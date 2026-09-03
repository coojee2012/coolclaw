"""HuggingFace Hub search for GGUF models."""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from .config import get_httpx_proxy

logger = logging.getLogger(__name__)

_HF_API = "https://huggingface.co/api"
_GGUF_RE = re.compile(r"\.gguf$", re.IGNORECASE)


def _client() -> httpx.Client:
  proxy = get_httpx_proxy()
  return httpx.Client(
    timeout=httpx.Timeout(20.0, connect=10.0),
    follow_redirects=True,
    proxy=proxy.get("https://") if proxy else None,
  )


def search_gguf_models(query: str, limit: int = 25) -> list[dict]:
  """Search HuggingFace for GGUF models via public API."""
  q = (query or "").strip()
  if not q:
    return []

  search_q = q if "gguf" in q.lower() else f"{q} gguf"
  params = {
    "search": search_q,
    "filter": "text-generation-inference",  # broad; we filter by gguf files below
    "sort": "downloads",
    "direction": -1,
    "limit": min(limit * 3, 60),
  }

  try:
    with _client() as client:
      resp = client.get(f"{_HF_API}/models", params=params)
      resp.raise_for_status()
      items = resp.json()
  except Exception as e:
    logger.warning("HF search failed: %s", e)
    return _fallback_search(search_q, limit)

  results: list[dict] = []
  seen: set[str] = set()

  with _client() as client:
    for item in items:
      repo_id = item.get("modelId") or item.get("id", "")
      if not repo_id or repo_id in seen:
        continue
      # Prefer repos that mention gguf in id or tags
      tags = [t.lower() for t in (item.get("tags") or [])]
      repo_lower = repo_id.lower()
      if "gguf" not in repo_lower and "gguf" not in tags and not any("gguf" in t for t in tags):
        # Still check if repo has gguf files
        files = _list_gguf_files(client, repo_id, max_files=3)
        if not files:
          continue
      else:
        files = _list_gguf_files(client, repo_id, max_files=5)

      if not files:
        continue

      seen.add(repo_id)
      results.append({
        "repo_id": repo_id,
        "name": repo_id.split("/")[-1] if "/" in repo_id else repo_id,
        "author": repo_id.split("/")[0] if "/" in repo_id else "",
        "description": _truncate(item.get("pipeline_tag", "") or ", ".join(tags[:4]), 120),
        "downloads": item.get("downloads", 0),
        "likes": item.get("likes", 0),
        "tags": tags[:8],
        "gguf_files": files,
        "source": "huggingface",
      })
      if len(results) >= limit:
        break

  return results


def _fallback_search(query: str, limit: int) -> list[dict]:
  """Broader search without filter when primary search fails."""
  try:
    with _client() as client:
      resp = client.get(f"{_HF_API}/models", params={"search": query, "limit": limit * 2, "sort": "downloads", "direction": -1})
      resp.raise_for_status()
      items = resp.json()
  except Exception:
    return []

  results = []
  with _client() as client:
    for item in items:
      repo_id = item.get("modelId") or item.get("id", "")
      if not repo_id:
        continue
      files = _list_gguf_files(client, repo_id, max_files=3)
      if not files:
        continue
      results.append({
        "repo_id": repo_id,
        "name": repo_id.split("/")[-1],
        "author": repo_id.split("/")[0] if "/" in repo_id else "",
        "description": item.get("pipeline_tag", "GGUF model"),
        "downloads": item.get("downloads", 0),
        "likes": item.get("likes", 0),
        "tags": item.get("tags", [])[:8],
        "gguf_files": files,
        "source": "huggingface",
      })
      if len(results) >= limit:
        break
  return results


def _list_gguf_files(client: httpx.Client, repo_id: str, max_files: int = 10) -> list[dict]:
  try:
    resp = client.get(f"{_HF_API}/models/{repo_id}/tree/main")
    if resp.status_code == 404:
      resp = client.get(f"{_HF_API}/models/{repo_id}/tree/master")
    if resp.status_code != 200:
      return []
    entries = resp.json()
  except Exception:
    return []

  files = []
  for entry in entries:
    if entry.get("type") != "file":
      continue
    path = entry.get("path", "")
    if not _GGUF_RE.search(path):
      continue
    size = entry.get("size") or 0
    files.append({
      "filename": path.split("/")[-1],
      "path": path,
      "size_gb": round(size / (1024 ** 3), 2) if size else 0,
      "size_bytes": size,
    })
    if len(files) >= max_files:
      break
  return sorted(files, key=lambda x: x.get("size_bytes", 0))


def get_repo_gguf_files(repo_id: str) -> list[dict]:
  with _client() as client:
    return _list_gguf_files(client, repo_id, max_files=30)


def _truncate(s: str, n: int) -> str:
  return s if len(s) <= n else s[: n - 1] + "…"
