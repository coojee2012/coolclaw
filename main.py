import argparse
import logging
import os
import uvicorn
from src.api import create_app
from src.router import create_router
from src.config import get_config


def main():
    parser = argparse.ArgumentParser(description="CoolClaw - Local AI Agent Platform")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8484, help="Port to bind")
    parser.add_argument("--web-dir", default=None, help="Web static files directory")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    config = get_config(args.config)
    log_level = getattr(logging, config.logging.level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format=config.logging.format,
    )

    web_dir = args.web_dir or os.path.join(os.path.dirname(__file__), "web")

    # Initialize router for OpenAI-compatible API
    router = create_router(config)

    app = create_app(router=router, web_dir=web_dir)

    print(f"""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   🤖 CoolClaw - 本地 AI Agent 平台                          ║
║                                                           ║
║   🌐 Web UI: http://localhost:{args.port}                       ║
║   📖 API Docs: http://localhost:{args.port}/docs                ║
║                                                           ║
║   按 Ctrl+C 停止服务                                       ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """)

    from src.server_control import set_restart_argv
    import sys
    set_restart_argv(sys.argv)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
