# OpenCode Helper Skills

## Project-Specific Skills

### opencode-helper-dev

Purpose: Development and maintenance of OpenCode Helper

Trigger: Any code changes to this project

Guidelines:
- Follow AGENTS.md conventions
- Models stored on external SSD: `/Volumes/LynnData/myclaw_models`
- Config management via `config.yaml`, not hardcoded values
- Test locally with `python -m src.cli chat --local`

### llama-cpp-expert

Purpose: llama.cpp integration and optimization

Trigger: Working with `src/local_llm.py`, model loading, quantization

Guidelines:
- Use GGUF format for all models
- Q4_K_M is recommended quantization level
- For Mac MPS fallback, check `n_gpu_layers` setting
- Streaming support via `stream=True` parameter

### gemini-api-expert

Purpose: Google Gemini API integration

Trigger: Working with `src/gemini_client.py`, API server

Guidelines:
- Use `google-genai` SDK (not deprecated `google-generativeai`)
- API key via `GEMINI_API_KEY` env var
- Models: gemini-2.5-flash (fast), gemini-3-pro (best quality)
- Streaming support available

## General Skills

- `git-master`: Version control operations
- `frontend-ui-ux`: Any UI/CLI presentation layer work
- `ai-slop-remover`: Code quality review
