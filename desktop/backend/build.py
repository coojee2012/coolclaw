#!/usr/bin/env python3
"""Build the CoolClaw backend into a standalone binary using PyInstaller.

Usage:
    cd /Volumes/LynnData/Projects/coolclaw
    .venv/bin/python desktop/backend/build.py
"""
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "desktop", "backend")
OUTPUT_DIR = os.path.join(BACKEND_DIR, "dist")


def build():
    web_dir = os.path.join(PROJECT_ROOT, "web")
    config_file = os.path.join(PROJECT_ROOT, "config.yaml")
    main_py = os.path.join(PROJECT_ROOT, "main.py")
    entry = os.path.join(BACKEND_DIR, "main_wrapper.py")

    # Collect llama shared libraries
    llama_lib_dir = os.path.join(
        PROJECT_ROOT, ".venv", "lib", "python3.12",
        "site-packages", "llama_cpp", "lib"
    )
    # Place llama dylibs into llama_cpp/lib/ inside the bundle so the library
    # loader (which resolves relative to __file__) can find them.
    add_binary_args = []
    if os.path.isdir(llama_lib_dir):
        for fname in os.listdir(llama_lib_dir):
            if fname.endswith(".dylib"):
                src = os.path.join(llama_lib_dir, fname)
                add_binary_args += ["--add-binary", f"{src}:llama_cpp/lib"]

    pyinstaller_args = [
        sys.executable, "-m", "PyInstaller",
        "--name", "backend",
        "--onefile",
        "--clean",
        "--noconfirm",
        "--target-architecture", "arm64",
        "--distpath", OUTPUT_DIR,
        "--workpath", os.path.join(BACKEND_DIR, "build"),
        "--specpath", BACKEND_DIR,
        "--add-data", f"{web_dir}:web",
        "--add-data", f"{config_file}:.",
        *add_binary_args,
        "--hidden-import", "uvicorn",
        "--hidden-import", "uvicorn.logging",
        "--hidden-import", "uvicorn.loops",
        "--hidden-import", "uvicorn.loops.auto",
        "--hidden-import", "uvicorn.protocols",
        "--hidden-import", "uvicorn.protocols.http",
        "--hidden-import", "uvicorn.protocols.http.auto",
        "--hidden-import", "uvicorn.protocols.websockets",
        "--hidden-import", "uvicorn.protocols.websockets.auto",
        "--hidden-import", "uvicorn.lifespan",
        "--hidden-import", "uvicorn.lifespan.on",
        "--hidden-import", "fastapi",
        "--hidden-import", "starlette",
        "--hidden-import", "sse_starlette",
        "--hidden-import", "pydantic",
        "--hidden-import", "httpx",
        "--hidden-import", "yaml",
        "--hidden-import", "src",
        "--hidden-import", "src.api",
        "--hidden-import", "src.agent",
        "--hidden-import", "src.config",
        "--hidden-import", "src.dispatcher",
        "--hidden-import", "src.session",
        "--hidden-import", "src.memory",
        "--hidden-import", "src.models",
        "--hidden-import", "src.router",
        "--hidden-import", "src.proxy",
        "--hidden-import", "src.rate_limiter",
        "--hidden-import", "src.mcp_server",
        "--hidden-import", "src.mcp",
        "--hidden-import", "src.mcp.combined",
        "--hidden-import", "src.mcp.codegraph",
        "--hidden-import", "src.mcp.lsp",
        "--hidden-import", "src.mcp.websearch",
        "--hidden-import", "src.orchestrator",
        "--hidden-import", "src.storage",
        "--hidden-import", "src.task_manager",
        "--hidden-import", "src.knowledge_base",
        "--hidden-import", "src.local_llm",
        "--hidden-import", "src.gemini_client",
        "--hidden-import", "rich",
        "--hidden-import", "rich.console",
        "--hidden-import", "rich.markup",
        "--exclude-module", "tkinter",
        "--exclude-module", "matplotlib",
        "--exclude-module", "numpy.random._examples",
        "--exclude-module", "PIL",
        "--exclude-module", "pytest",
        "--exclude-module", "unittest",
        entry,
    ]

    print(f"Building backend binary...")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Entry point:  {entry}")
    print(f"  Web dir:      {web_dir}")
    print(f"  Config:       {config_file}")
    print(f"  Output:       {OUTPUT_DIR}")
    print()

    result = subprocess.run(pyinstaller_args, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"PyInstaller failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    binary = os.path.join(OUTPUT_DIR, "backend")
    if os.path.exists(binary):
        size_mb = os.path.getsize(binary) / (1024 * 1024)
        print(f"\nBuild succeeded: {binary} ({size_mb:.1f} MB)")
    else:
        print(f"\nBuild completed but binary not found at {binary}")
        sys.exit(1)


if __name__ == "__main__":
    build()
