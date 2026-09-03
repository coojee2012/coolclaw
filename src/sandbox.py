"""Cross-platform command sandbox — allowlist + OS-native confinement.

Replaces the weak blacklist in combined.py with:
1. Command allowlist (only whitelisted commands can run)
2. OS-native sandbox wrapping (Seatbelt on macOS, Landlock/bwrap on Linux)
3. Always-blocked dangerous patterns (rm -rf /, fork bombs, etc.)

Usage:
    from src.sandbox import is_command_allowed, confined_subprocess_run, get_capabilities

    # Check allowlist
    blocked = is_command_allowed("git status")
    if blocked:
        return f"Error: {blocked}"

    # Run with sandbox
    result = confined_subprocess_run("git status", workdir="/path/to/project")
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Allowlist data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AllowedPattern:
    """Pattern for allowed command arguments."""

    args_pattern: str | None = None  # regex; None = allow any args

    def matches_args(self, args_str: str) -> bool:
        if self.args_pattern is None:
            return True
        return bool(re.search(self.args_pattern, args_str))


# Core development commands — allow specific subcommands/patterns
_ALLOWED_COMMANDS: dict[str, AllowedPattern | bool] = {
    # Version control
    "git": AllowedPattern(
        r"^(status|log|diff|stat|commit|push|pull|branch|checkout|add|merge|rebase|"
        r"stash|reset|show|blame|remote|tag|fetch|clone|init|config|ls-files|shortlog)\b"
    ),
    "tig": True,

    # Python
    "python": True, "python3": True,
    "pip": True, "pip3": True,
    "pytest": True, "mypy": True, "ruff": True, "black": True,
    "uv": True, "uvicorn": True, "pyright": True, "basedpyright": True,

    # Node.js / TypeScript
    "node": True, "npm": True, "npx": True,
    "yarn": True, "pnpm": True, "bun": True,
    "tsc": True, "eslint": True, "prettier": True, "biome": True,

    # Package managers
    "apt": True, "apt-get": True, "yum": True, "dnf": True,
    "brew": True, "snap": True, "flatpak": True,

    # Build tools
    "make": True, "cmake": True, "ninja": True,
    "cargo": True, "rustc": True, "rustup": True,
    "go": True,
    "java": True, "javac": True, "gradle": True, "mvn": True,

    # Safe system utilities
    "ls": True, "ll": True, "la": True,
    "cat": True, "head": True, "tail": True,
    "grep": True, "rg": True, "ag": True, "ack": True,
    "find": True, "fd": True, "tree": True,
    "wc": True, "sort": True, "uniq": True, "cut": True,
    "diff": True, "file": True, "stat": True, "du": True, "df": True,
    "echo": True, "printf": True, "test": True,
    "pwd": True, "whoami": True, "date": True, "uptime": True,
    "env": True, "printenv": True,
    "which": True, "whereis": True, "command": True, "type": True,
    "uname": True, "hostname": True, "id": True,
    "realpath": True, "dirname": True, "basename": True,
    "mktemp": True, "tee": True, "xargs": True,

    # Process / diagnostics
    "ps": True, "top": True, "htop": True, "btop": True,
    "lsof": True, "netstat": True, "ss": True,
    "free": True, "vm_stat": True,

    # File operations (safe)
    "mkdir": True, "touch": True, "cp": True, "mv": True,
    "ln": True, "chmod": True, "chown": True,

    # Archive
    "tar": True, "zip": True, "unzip": True,
    "gzip": True, "gunzip": True, "zcat": True,

    # Text processing
    "sed": True, "awk": True, "tr": True, "column": True,
    "jq": True, "yq": True,

    # Network (read-only / safe)
    "curl": True, "wget": True,
    "dig": True, "nslookup": True, "host": True,
    "ssh": True, "scp": True, "rsync": True,

    # Dev tools
    "code": True,       # VS Code
    "open": True,       # macOS open
    "xdg-open": True,   # Linux open
    "pbcopy": True, "pbpaste": True,  # macOS clipboard
    "say": True,        # macOS TTS
    "npx": True,
}

# Prefix patterns that are allowed before the actual command
_CMD_PREFIXES = {"sudo", "env", "nohup", "nice", "time", "strace", "ltrace"}

# Dangerous patterns that are ALWAYS blocked regardless of allowlist
_ALWAYS_BLOCKED: list[re.Pattern[str]] = [
    re.compile(r"\brm\s+-[a-z]*r[a-z]*f[a-z]*\s+/\b"),    # rm -rf /
    re.compile(r"\bdd\s+if=/dev/(zero|random|sda)"),         # dd wipe
    re.compile(r":\(\)\s*\{\s*:\|:\&\s*\}\s*:"),            # fork bomb
    re.compile(r">\s*/dev/sd[a-z]"),                          # write to block device
    re.compile(r"\bmkfs\b"),                                  # format filesystem
    re.compile(r"\bchmod\s+777\b"),                           # chmod 777
    re.compile(r"\bshutdown\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\bhalt\b"),
]


# ---------------------------------------------------------------------------
# Allowlist check
# ---------------------------------------------------------------------------

def _extract_command_base(command: str) -> tuple[str, str]:
    """Extract the base command and remaining args from a shell command string.

    Handles prefixes like ``sudo``, ``env VAR=val``, ``nohup``, etc.
    Returns ``(base_command, args_string)``.
    """
    tokens = command.strip().split()
    if not tokens:
        return ("", "")

    idx = 0
    # Skip known prefixes
    while idx < len(tokens) and tokens[idx] in _CMD_PREFIXES:
        idx += 1
        # Skip env VAR=val pairs
        if idx > 0 and tokens[idx - 1] == "env":
            while idx < len(tokens) and "=" in tokens[idx]:
                idx += 1

    if idx >= len(tokens):
        return ("", "")

    base = tokens[idx]
    # Strip path: /usr/bin/git → git
    base = os.path.basename(base)
    # Strip extension: python3.12 → python3
    base = re.sub(r"\.\d+$", "", base)

    args = " ".join(tokens[idx + 1:])
    return (base, args)


def is_command_allowed(command: str) -> str | None:
    """Check if a command is in the allowlist.

    Returns ``None`` if allowed, error message string if blocked.
    """
    cmd_str = command.strip()
    if not cmd_str:
        return "Empty command"

    # Check always-blocked patterns first
    for pattern in _ALWAYS_BLOCKED:
        if pattern.search(cmd_str):
            return "Blocked: dangerous command pattern detected"

    # Check for pipe-to-shell (| sh, | bash, etc.)
    if re.search(r"\|\s*(ba)?sh\b", cmd_str):
        return "Blocked: piping to shell is not allowed"

    # Check for command substitution with shell operators
    if re.search(r"\$\(", cmd_str) or re.search(r"`[^`]+`", cmd_str):
        if re.search(r"\$\([^)]*[;&|]", cmd_str):
            return "Blocked: command substitution with shell operators not allowed"

    base, args = _extract_command_base(cmd_str)
    if not base:
        return "Could not parse command"

    pattern = _ALLOWED_COMMANDS.get(base)
    if pattern is None:
        return f"Blocked: command '{base}' is not in the allowlist"

    if isinstance(pattern, bool):
        if not pattern:
            return f"Blocked: command '{base}' is explicitly denied"
        return None  # Allowed unconditionally

    if isinstance(pattern, AllowedPattern):
        if not pattern.matches_args(args):
            return f"Blocked: arguments for '{base}' do not match allowed pattern"
        return None

    return None


# ---------------------------------------------------------------------------
# Platform detection & probing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SandboxCapabilities:
    """What sandbox features are available on this platform."""

    seatbelt: bool = False        # macOS sandbox-exec
    landlock: bool = False        # Linux Landlock (kernel 5.13+)
    bubblewrap: bool = False      # Linux bubblewrap (bwrap)
    restricted_token: bool = False  # Windows restricted token (not yet implemented)


_caps: SandboxCapabilities | None = None


def get_capabilities() -> SandboxCapabilities:
    """Get cached sandbox capabilities (probed once at startup)."""
    global _caps
    if _caps is None:
        _caps = _probe_sandbox()
        logger.info(
            "Sandbox capabilities: seatbelt=%s landlock=%s bubblewrap=%s",
            _caps.seatbelt, _caps.landlock, _caps.bubblewrap,
        )
    return _caps


def _probe_sandbox() -> SandboxCapabilities:
    """Probe what sandbox features are available on this platform."""
    if sys.platform == "darwin":
        return SandboxCapabilities(seatbelt=_probe_seatbelt())
    elif sys.platform == "linux":
        return SandboxCapabilities(
            landlock=_probe_landlock(),
            bubblewrap=_probe_bubblewrap(),
        )
    elif sys.platform == "win32":
        return SandboxCapabilities(restricted_token=False)
    return SandboxCapabilities()


def _probe_seatbelt() -> bool:
    """Test if sandbox-exec works on this macOS.

    On macOS 14+ (Sonoma/Sequoia), unprivileged processes may get
    'Operation not permitted' from sandbox_apply even though the binary
    exists. We probe with a trivial profile.
    """
    sandbox_exec = "/usr/bin/sandbox-exec"
    if not os.path.isfile(sandbox_exec):
        return False
    try:
        result = subprocess.run(
            [sandbox_exec, "-p", "(version 1)(allow default)", "/usr/bin/true"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _probe_landlock() -> bool:
    """Test if Landlock is available (Linux 5.13+)."""
    if sys.platform != "linux":
        return False
    try:
        uname = os.uname()
        parts = uname.release.split(".")
        major, minor = int(parts[0]), int(parts[1])
        return (major > 5) or (major == 5 and minor >= 13)
    except (ValueError, IndexError):
        return False


def _probe_bubblewrap() -> bool:
    """Test if bubblewrap (bwrap) is installed."""
    return shutil.which("bwrap") is not None


# ---------------------------------------------------------------------------
# Sandbox wrapping — Seatbelt (macOS)
# ---------------------------------------------------------------------------

def _wrap_seatbelt(
    cmd: str | list[str],
    workdir: str,
    allow_network: bool,
) -> list[str]:
    """Wrap command with macOS sandbox-exec.

    Writes profile to a temp file and uses ``-f`` (more reliable than ``-p``
    with ``-D`` params when invoked via ``subprocess.run(list)``).
    Profile: deny default, allow file-read everywhere,
    file-write ONLY in workdir.
    """
    profile_parts = [
        "(version 1)",
        "(deny default)",
        "(allow process-fork)",
        "(allow process-exec)",
        "(allow file-read*)",
        "(allow file-read-data (subpath \"/dev\"))",
        f'(allow file-write* (subpath "{workdir}"))',
        "(allow file-write* (subpath \"/dev\"))",
        "(allow sysctl-read)",
        "(allow mach-lookup)",            # TLS trust store + DNS
        "(allow signal (target same-sandbox))",
    ]

    if allow_network:
        profile_parts.append("(allow network*)")
    else:
        profile_parts.append("(allow local-network*)")
        profile_parts.append("(deny network*)")

    profile = " ".join(profile_parts)
    cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd

    # Write profile to temp file (mode 0600, deleted after use)
    profile_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".sb", delete=False, prefix="coolclaw_sandbox_",
    )
    try:
        profile_file.write(profile)
        profile_file.close()
        os.chmod(profile_file.name, 0o600)
        return [
            "sandbox-exec", "-f", profile_file.name,
            "/bin/sh", "-c", cmd_str,
        ]
    except Exception:
        # Fallback: try -p with -D (may not work in all cases)
        profile_file.close()
        try:
            os.unlink(profile_file.name)
        except OSError:
            pass
        return [
            "sandbox-exec", "-p", profile,
            "-D", f"WORKDIR={workdir}",
            "/bin/sh", "-c", cmd_str,
        ]


# ---------------------------------------------------------------------------
# Sandbox wrapping — bubblewrap (Linux)
# ---------------------------------------------------------------------------

def _wrap_bubblewrap(cmd: str | list[str], workdir: str) -> list[str]:
    """Wrap command with Linux bubblewrap."""
    cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
    return [
        "bwrap",
        "--ro-bind", "/", "/",
        "--bind", workdir, workdir,
        "--tmpfs", "/tmp",
        "--dev", "/dev",
        "--proc", "/proc",
        "--unshare-all",
        "--die-with-parent",
        "--chdir", workdir,
        "/bin/sh", "-c", cmd_str,
    ]


# ---------------------------------------------------------------------------
# Sandbox wrapping — Landlock (Linux, ctypes)
# ---------------------------------------------------------------------------

def make_landlock_preexec(workdir: str) -> Callable[[], None]:
    """Create a preexec_fn that applies Landlock filesystem restrictions.

    Used as ``subprocess.Popen(preexec_fn=...)``.
    Restrictions are irrevocable and inherited by all children.
    """

    def _apply_landlock() -> None:
        try:
            SYSCALL_CREATE_RULESET = 444
            SYSCALL_ADD_RULE = 445
            SYSCALL_RESTRICT_SELF = 446
            PR_SET_NO_NEW_PRIVS = 38

            libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

            # Access rights
            READ_FILE = 1 << 2
            READ_DIR = 1 << 3
            WRITE_FILE = 1 << 1
            REMOVE = 1 << 4
            MAKE_REG = 1 << 7
            TRUNCATE = 1 << 13

            all_read = READ_FILE | READ_DIR
            write_rights = WRITE_FILE | REMOVE | MAKE_REG | TRUNCATE

            # Create ruleset
            attr = struct.pack("QQ", all_read | write_rights, 0)
            ruleset_fd = libc.syscall(
                SYSCALL_CREATE_RULESET, attr, len(attr), 0
            )
            if ruleset_fd < 0:
                return

            # Allow read on root filesystem
            root_fd = os.open("/", os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW)
            rule = struct.pack("iQi", root_fd, all_read, 0)
            libc.syscall(SYSCALL_ADD_RULE, ruleset_fd, 1, rule, 0)
            os.close(root_fd)

            # Allow read+write in workdir
            workdir_fd = os.open(
                workdir, os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW
            )
            rule = struct.pack("iQi", workdir_fd, all_read | write_rights, 0)
            libc.syscall(SYSCALL_ADD_RULE, ruleset_fd, 1, rule, 0)
            os.close(workdir_fd)

            # PR_SET_NO_NEW_PRIVS (required for seccomp, good practice)
            libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)

            # Apply restrictions
            libc.syscall(SYSCALL_RESTRICT_SELF, ruleset_fd, 0, 0)
            os.close(ruleset_fd)
        except Exception as exc:
            logger.warning("Landlock apply failed: %s", exc)

    return _apply_landlock


# ---------------------------------------------------------------------------
# Public API — confine_command
# ---------------------------------------------------------------------------

def confine_command(
    cmd: str | list[str],
    workdir: str,
    *,
    allow_network: bool = True,
) -> list[str]:
    """Wrap a command with OS-native sandbox.

    Returns the wrapped command as a list of strings for
    ``subprocess.run(..., shell=False)``.

    If no sandbox is available, returns the command unchanged
    (allowlist check still applies separately).
    """
    caps = get_capabilities()

    if sys.platform == "darwin" and caps.seatbelt:
        return _wrap_seatbelt(cmd, workdir, allow_network)
    elif sys.platform == "linux":
        if caps.bubblewrap:
            return _wrap_bubblewrap(cmd, workdir)
        if caps.landlock:
            # Landlock must be applied in child via preexec_fn
            cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
            return ["__SANDBOX_LANDLOCK__", workdir, "/bin/sh", "-c", cmd_str]

    # Fallback: plain shell command
    if isinstance(cmd, str):
        return ["/bin/sh", "-c", cmd]
    return list(cmd)


# ---------------------------------------------------------------------------
# Public API — confined_subprocess_run
# ---------------------------------------------------------------------------

def confined_subprocess_run(
    command: str | list[str],
    workdir: str,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    """Run a command with allowlist check + OS-native sandbox.

    1. Check allowlist → ``PermissionError`` if blocked
    2. Apply OS sandbox wrapping
    3. Execute via ``subprocess.run``

    Raises ``PermissionError`` if command is not in the allowlist.
    """
    cmd_str = command if isinstance(command, str) else " ".join(command)
    blocked = is_command_allowed(cmd_str)
    if blocked:
        raise PermissionError(blocked)

    wrapped = confine_command(command, workdir)

    # Handle Landlock special case (needs preexec_fn)
    if wrapped and wrapped[0] == "__SANDBOX_LANDLOCK__":
        wd = wrapped[1]
        actual_cmd = wrapped[2:]
        kwargs["preexec_fn"] = make_landlock_preexec(wd)
        return subprocess.run(actual_cmd, shell=False, **kwargs)

    # For Seatbelt: clean up temp profile file after execution
    profile_to_cleanup: str | None = None
    if sys.platform == "darwin" and len(wrapped) >= 3 and wrapped[1] == "-f":
        profile_to_cleanup = wrapped[2]

    try:
        return subprocess.run(wrapped, shell=False, **kwargs)
    finally:
        if profile_to_cleanup:
            try:
                os.unlink(profile_to_cleanup)
            except OSError:
                pass
