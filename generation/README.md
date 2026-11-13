# Prome RAG Plugins — Generation Ecosystem

## Overview

The Generation and Model Routing ecosystem for Prome RAG Plugins.
Supports local Ollama Qwen models, Groq generation with 12-key rotation,
adaptive Prompt Guard integration, optional GPT-OSS-Safeguard policy evaluation,
optional query enhancement, and model-aware rate limiting.

## Model Inventory

### Groq Generation Models (default fallback chain)
1. **Primary**: `qwen/qwen3.8-27b` — RPM=30, RPD=1000, TPM=8000, TPD=200000
2. **First fallback**: `openai/gpt-oss-120b` — same limits
3. **Second fallback**: `openai/gpt-oss-20b` — same limits

### Local Ollama Models (manual selection)
- **Qwen 8B** — 4-bit quantization: `OLLAMA_QWEN_8B_MODEL`
- **Qwen 4B** — 4-bit quantization: `OLLAMA_QWEN_4B_MODEL`

### Prompt Guard Classification Models
- `meta-llama/llama-prompt-guard-2-22m` — English
- `meta-llama/llama-prompt-guard-2-86m` — Multilingual / non-English

### Safeguard Model (optional, not in fallback chain)
- `openai/gpt-oss-safeguard-20b` — policy-based evaluation
- Dedicated limits: RPM=3, TPM=2000

## Environment Variables

```bash
# Groq 12-key rotation (never hardcode secrets)
GROQ_API_KEY_1=...
GROQ_API_KEY_2=...
...
GROQ_API_KEY_12=...
GROQ_BASE_URL=https://api.groq.com/openai/v1

# Ollama local models
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_QWEN_8B_MODEL=qwen:8b
OLLAMA_QWEN_4B_MODEL=qwen:4b

# Generation configuration
GENERATION_DEFAULT_PROVIDER=groq
GENERATION_DEFAULT_MODE=auto
QUERY_ENHANCEMENT_ENABLED=false
PROMPT_GUARD_ENABLED=true
PROMPT_GUARD_LANGUAGE_MODE=adaptive
SAFEGUARD_MODE=off
```

## Architecture

```
Incoming request
  → Request validation
  → Input and token caps
  → Prompt Guard (adaptive routing)
  → Optional query enhancement
  → Manual Ollama selection OR Groq auto-route (12-key load balanced)
  → Generation response
  → Optional Safeguard evaluation
  → Structured answer and diagnostics
```

## Components

- `contracts/` — Shared data models, protocols, capabilities, exceptions (dependency-free)
- `providers/ollama/` — Local Qwen inference (lazy init, HTTP health checks)
- `providers/groq/` — Groq API with parameter compatibility validation
- `routing/` — Generation router, fallback manager, model selector
- `rate_limiting/` — Per-key RPM/RPD/TPM/TPD tracking, key rotation pool
- `input_management/` — Input caps, token counting, context budgeting
- `safety/prompt_guard/` — Adaptive prompt-injection detection
- `safety/safeguard/` — Policy-based content safety evaluation
- `query_enhancement/` — Optional query rewriting

## Testing

```bash
python -m pytest tests/generation -q
ruff check generation
python -m pip check
```

See `/.env.example` for all environment placeholders.

## License

MIT
