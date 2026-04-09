# OpenCode Agent Conventions

## Project Overview

OpenCode Helper is a lightweight local AI assistant combining llama.cpp local inference with Gemini API fallback.

## Agent Definitions

### Primary Agents

| Agent | Purpose | Trigger |
|-------|---------|---------|
| `build` | General implementation, debugging | Default for most tasks |
| `explore` | Codebase understanding, pattern finding | "Find how X works", "Where is Y" |
| `librarian` | External docs, library research | Unfamiliar libraries, API lookups |
| `oracle` | Architecture, high-level design | Complex decisions, multi-system tradeoffs |

### Domain Agents

| Agent | Purpose | When to Use |
|-------|---------|--------------|
| `visual-engineering` | UI/frontend work | Any UI/UX/styling tasks |
| `ultrabrain` | Hard logic/algorithms | Complex problem-solving |
| `deep` | Autonomous research + implementation | Hairy problems requiring research |
| `quick` | Trivial single-file changes | Typos, simple modifications |

## Workflow Guidelines

### Before Implementing

1. Check `AGENTS.md` for project conventions
2. Review existing patterns in `src/`
3. For external libraries → fire `librarian` first

### During Implementation

1. Follow existing code style (check 2-3 similar files first)
2. Use Pydantic for config/data models
3. Prefer async patterns for API/server code
4. Keep functions small and focused

### After Implementing

1. Run `lsp_diagnostics` on changed files
2. Verify build/test passes
3. Update relevant documentation

## Code Style

- Type hints required
- Docstrings only for complex public APIs
- Prefer dataclasses for simple data structures
- Use `logging` module, not print statements

## Testing

- No formal test suite yet (contribution opportunity)
- Manual testing via `python -m src.cli chat --local`

## Resources

- Models: `/Volumes/LynnData/myclaw_models`
- Config: `config.yaml`
- API: `src/api.py` (FastAPI)
