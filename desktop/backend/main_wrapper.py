"""Entry point for PyInstaller-bundled backend.

When frozen (PyInstaller --onefile), resolves config.yaml and web/ from
sys._MEIPASS (the temp extraction directory). In dev mode, resolves
relative to the project root.
"""
import argparse
import logging
import os
import sys
import types


def _resolve_base() -> str:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _stub_heavy_imports():
    """Stub out heavy optional dependencies that fail in frozen builds.

    llama_cpp needs native dylibs with complex loading; in cloud_only mode it
    is never instantiated, so a stub is enough.  In frozen mode we skip the
    real import entirely because ctypes.CDLL can *hang* instead of raising
    when loading an incompatible or missing dylib.
    """
    for mod_name in ("llama_cpp",):
        if mod_name in sys.modules:
            continue
        if getattr(sys, "frozen", False):
            # Never attempt the real import in a frozen binary — it may hang.
            stub = types.ModuleType(mod_name)
            stub.Llama = type("Llama", (), {})  # type: ignore[attr-defined]
            sys.modules[mod_name] = stub
        else:
            try:
                __import__(mod_name)
            except Exception:
                stub = types.ModuleType(mod_name)
                stub.Llama = type("Llama", (), {})  # type: ignore[attr-defined]
                sys.modules[mod_name] = stub


def main():
    base = _resolve_base()

    # When frozen, set paths so ctypes/dlopen can find the llama shared
    # libraries bundled by PyInstaller into llama_cpp/lib/.
    if getattr(sys, "frozen", False):
        llama_lib = os.path.join(base, "llama_cpp", "lib")
        os.environ["LLAMA_CPP_LIB_PATH"] = llama_lib
        for var in ("DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"):
            existing = os.environ.get(var, "")
            os.environ[var] = (llama_lib + (":" + existing) if existing else llama_lib)
        sys.path.insert(0, base)

    default_config = os.path.join(base, "config.yaml")
    default_web = os.path.join(base, "web")

    _stub_heavy_imports()

    from src.api import create_app
    from src.router import create_router
    from src.config import get_config

    config_path = default_config
    web_dir = default_web

    parser = argparse.ArgumentParser(description="CoolClaw Backend")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8484)
    parser.add_argument("--config", default=config_path)
    parser.add_argument("--web-dir", default=web_dir)
    args = parser.parse_args()

    log_config = get_config(args.config)
    log_level = getattr(logging, log_config.logging.level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format=log_config.logging.format)

    router = create_router(log_config)
    app = create_app(router=router, web_dir=args.web_dir)

    import uvicorn
    print(f"[CoolClaw] Backend starting on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
