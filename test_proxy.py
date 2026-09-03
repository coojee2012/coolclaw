#!/usr/bin/env python3
"""Quick smoke test for the proxy — validates OMO compatibility."""

import json
import os
import sys
import time
import httpx

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

BASE = "http://127.0.0.1:8484"
API_KEY = "sk-local-dev"
client = httpx.Client(proxy=None)
async_client = httpx.AsyncClient(proxy=None)


def test_models():
    r = httpx.get(f"{BASE}/v1/models", headers={"Authorization": f"Bearer {API_KEY}"})
    assert r.status_code == 200, f"models: {r.status_code}"
    data = r.json()
    models = [m["id"] for m in data["data"]]
    print(f"[OK] /v1/models → {models}")
    return models


def test_chat_non_stream(model: str):
    t0 = time.time()
    r = httpx.post(
        f"{BASE}/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Say hi in 5 words"}],
            "max_tokens": 50,
            "stream": False,
        },
        timeout=30,
    )
    elapsed = (time.time() - t0) * 1000
    assert r.status_code == 200, f"chat non-stream: {r.status_code} {r.text[:200]}"
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    print(f"[OK] chat non-stream ({elapsed:.0f}ms) → {content[:80]}")


def test_chat_stream(model: str):
    t0 = time.time()
    chunks = []
    with client.stream(
        "POST",
        f"{BASE}/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Say bye in 3 words"}],
            "max_tokens": 30,
            "stream": True,
        },
        timeout=30,
    ) as r:
        assert r.status_code == 200, f"stream: {r.status_code}"
        for line in r.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks.append(json.loads(line[6:]))
    elapsed = (time.time() - t0) * 1000
    content = ""
    for c in chunks:
        if "choices" not in c:
            continue
        delta = c["choices"][0].get("delta", {})
        if "content" in delta:
            content += delta["content"]
    print(f"[OK] chat stream ({elapsed:.0f}ms, {len(chunks)} chunks) → {content.strip()[:80]}")


def test_proxy_status():
    r = httpx.get(f"{BASE}/proxy/status", headers={"Authorization": f"Bearer {API_KEY}"})
    assert r.status_code == 200, f"proxy status: {r.status_code}"
    data = r.json()
    rl = data.get("rate_limit_stats", {})
    for name, stats in rl.items():
        used = stats.get("rpd_used", 0)
        limit = stats.get("rpd_limit", 0)
        print(f"[OK] {name}: RPM={stats['rpm']} daily={used}/{limit} active={stats['active']}")


def main():
    models = test_models()
    test_chat_non_stream(models[0])
    test_chat_stream(models[0])
    test_proxy_status()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
