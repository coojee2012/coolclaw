import sys
import argparse
import logging
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table

from .config import get_config
from .router import Router, Provider, create_router


console = Console()


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def cmd_chat(args, router: Router):
    console.print(
        Panel.fit(
            "[bold blue]CoolClaw - Chat Mode[/bold blue]\n"
            "Type your message and press Enter. Use /exit to quit, /clear to clear history.",
            border_style="blue",
        )
    )

    messages = []

    if args.system:
        messages.append({"role": "system", "content": args.system})

    if args.file:
        try:
            with open(args.file) as f:
                file_content = f.read()
                messages.append(
                    {
                        "role": "system",
                        "content": f"Context from file {args.file}:\n```\n{file_content}\n```",
                    }
                )
        except FileNotFoundError:
            console.print(f"[red]File not found: {args.file}[/red]")
            return

    provider = None
    if args.local:
        provider = Provider.LOCAL
    elif args.cloud:
        provider = Provider.GOOGLE_AI

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/bold green]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting...[/yellow]")
            break

        if user_input.strip() == "/exit":
            break

        if user_input.strip() == "/clear":
            messages = []
            if args.system:
                messages.append({"role": "system", "content": args.system})
            console.print("[yellow]History cleared[/yellow]")
            continue

        if user_input.strip() == "/status":
            status = router.get_status()
            table = Table(title="Router Status")
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="green")
            for key, value in status.items():
                table.add_row(key, str(value))
            console.print(table)
            continue

        if user_input.strip() == "/mode":
            new_mode = Prompt.ask(
                "Select mode",
                choices=["local_only", "cloud_only", "auto"],
                default="auto",
            )
            router.set_mode(new_mode)
            continue

        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            console.print("\n[bold cyan]Thinking...[/bold cyan]", end="")

            response = router.chat(
                messages=messages,
                provider=provider,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                stream=args.stream,
            )

            if args.stream:
                console.print()
                for chunk in response.chunks:
                    if hasattr(chunk, "content"):
                        print(chunk.content, end="", flush=True)
                print()
            else:
                console.print()
                md = Markdown(response.content)
                console.print(
                    Panel(
                        md,
                        title=f"[bold]Response ({response.provider.value})[/bold]",
                        border_style="green",
                    )
                )
                messages.append({"role": "assistant", "content": response.content})
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


def cmd_complete(args, router: Router):
    try:
        with open(args.input) as f:
            prompt = f.read()
    except FileNotFoundError:
        prompt = args.input

    console.print(
        f"[dim]Prompt: {prompt[:100]}...[/dim]"
        if len(prompt) > 100
        else f"[dim]Prompt: {prompt}[/dim]"
    )

    provider = (
        Provider.LOCAL if args.local else (Provider.GOOGLE_AI if args.cloud else None)
    )

    try:
        response = router.complete(
            prompt=prompt,
            provider=provider,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=args.stream,
        )

        if args.stream:
            for chunk in response.chunks:
                if hasattr(chunk, "content"):
                    print(chunk.content, end="", flush=True)
            print()
        else:
            console.print(
                Panel(
                    response.content,
                    title=f"Response ({response.provider.value})",
                    border_style="green",
                )
            )
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def cmd_server(args, router: Router):
    from .api import create_app

    if args.model:
        router._default_local_model = args.model

    console.print(
        Panel.fit(
            f"[bold green]Starting API Server[/bold green]\n"
            f"Host: {args.host}:{args.port}\n"
            f"Endpoint: {args.endpoint}",
            border_style="green",
        )
    )

    import os

    web_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
    app = create_app(router, web_dir=web_dir)

    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


def cmd_status(args, router: Router):
    status = router.get_status()

    table = Table(title="CoolClaw Status")
    table.add_column("Setting", style="cyan", justify="left")
    table.add_column("Value", style="green", justify="left")

    for key, value in status.items():
        if isinstance(value, bool):
            value = "✓" if value else "✗"
        table.add_row(key, str(value))

    console.print(table)


def cmd_download(args, router: Router):
    console.print("[yellow]Note: Model download not implemented yet.[/yellow]")
    console.print("Please download models manually from HuggingFace:")
    console.print(
        "  - Qwen2.5-Coder-7B-Instruct: https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
    )
    console.print("  - Place in ./models/ directory")


def main():
    parser = argparse.ArgumentParser(
        description="CoolClaw - Local AI Agent Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    chat_parser = subparsers.add_parser("chat", help="Interactive chat")
    chat_parser.add_argument(
        "--system", default="You are a helpful coding assistant.", help="System prompt"
    )
    chat_parser.add_argument("--file", help="Add file content as context")
    chat_parser.add_argument("--local", action="store_true", help="Force local model")
    chat_parser.add_argument("--cloud", action="store_true", help="Force Gemini")
    chat_parser.add_argument("--max-tokens", type=int, default=2048)
    chat_parser.add_argument("--temperature", type=float, default=0.7)
    chat_parser.add_argument("--stream", action="store_true", help="Stream output")

    complete_parser = subparsers.add_parser("complete", help="Single completion")
    complete_parser.add_argument("input", help="Prompt or @filename")
    complete_parser.add_argument("--local", action="store_true")
    complete_parser.add_argument("--cloud", action="store_true")
    complete_parser.add_argument("--max-tokens", type=int, default=2048)
    complete_parser.add_argument("--temperature", type=float, default=0.7)
    complete_parser.add_argument("--stream", action="store_true")

    server_parser = subparsers.add_parser("serve", help="Start API server")
    server_parser.add_argument("--host", default="127.0.0.1")
    server_parser.add_argument("--port", type=int, default=8484)
    server_parser.add_argument("--endpoint", default="/v1/chat/completions")
    server_parser.add_argument("--model", help="Override default model")

    subparsers.add_parser("status", help="Show status")
    subparsers.add_parser("download", help="Download models")

    args = parser.parse_args()

    setup_logging("DEBUG" if args.verbose else "INFO")

    config = get_config(args.config)
    router = create_router(config)

    if args.command == "chat":
        cmd_chat(args, router)
    elif args.command == "complete":
        cmd_complete(args, router)
    elif args.command == "serve":
        cmd_server(args, router)
    elif args.command == "status":
        cmd_status(args, router)
    elif args.command == "download":
        cmd_download(args, router)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
