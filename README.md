# OpenCode Helper

Lightweight local AI assistant with Gemini fallback. Designed for OpenCode workflow.

## Features

- **Local Inference**: Run Qwen2.5-Coder locally with llama.cpp
- **Gemini Fallback**: Seamlessly use Google Gemini for complex tasks
- **OpenAI Compatible API**: Works with existing tools via `/v1/chat/completions`
- **Auto Routing**: Automatically selects local or cloud based on task complexity

## Quick Start

### 1. Install Dependencies

```bash
cd opencode_helper
pip install -r requirements.txt
```

### 2. Download a Model

Download a GGUF model from HuggingFace:

```bash
# Qwen2.5-Coder 7B (recommended for Mac)
mkdir -p models
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct-GGUF \
    Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf \
    --local-dir models/
```

### 3. Configure

Set your Gemini API key (optional):

```bash
export GEMINI_API_KEY="your-key-here"
```

Or edit `config.yaml`.

### 4. Run

```bash
# Interactive chat
python -m opencode_helper.main chat

# Start API server
python -m opencode_helper.main serve

# Single completion
python -m opencode_helper.main complete "Write a hello world in Python"
```

## Usage

### CLI Chat

```bash
python -m opencode_helper.main chat \
    --system "You are a coding assistant" \
    --local  # Force local model
```

### API Server

```bash
python -m opencode_helper.main serve --port 8080
```

Then use with OpenAI client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-local-dev"
)

response = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Architecture

```
opencode_helper/
├── src/
│   ├── config.py       # Configuration management
│   ├── local_llm.py    # llama.cpp integration
│   ├── gemini_client.py # Gemini API wrapper
│   ├── router.py       # Local/cloud routing
│   ├── cli.py         # Command-line interface
│   ├── api.py         # FastAPI server
│   └── turboquant.py  # Optional KV cache compression
├── config.yaml        # User configuration
└── requirements.txt   # Dependencies
```

## TurboQuant (Experimental)

TurboQuant KV cache compression is available but experimental. For local inference, GGUF quantization is recommended.

## Model Recommendations

| Model | Size (Q4) | RAM Needed | Best For |
|-------|-----------|------------|----------|
| Qwen2.5-Coder-7B | 4.7 GB | 8 GB | General coding |
| Qwen2.5-Coder-14B | 9 GB | 16 GB | Better quality |
| Qwen3-Coder-30B-A3B | 18 GB | 24 GB | MoE, agentic |

## License

MIT
