# Browser Agent

Self-hosted browser automation agent powered by open-weight LLMs.

## Quickstart

```bash
# Install
uv sync

# Configure: copy .env.example → .env, fill in a model API key
cp .env.example .env

# CLI — run a natural-language task
uv run browser-agent "go to example.com and return the H1 heading"

# Web UI — interactive SSE-powered chat at http://127.0.0.1:8000
uv run browser-agent-ui

# Tests
uv run pytest                          # all tests (LLM-gated ones autoskip without API key)
uv run pytest tests/test_injection.py -v  # single file
uv run ruff check .                    # lint
```

## Configuration

All knobs are env vars (see `.env.example`). Key ones:

| Variable | Default | Purpose |
|---|---|---|
| `PROVIDER` | `kimi` | `kimi` (Moonshot K2.6) or `openai` (generic OpenAI-compat) |
| `MOONSHOT_API_KEY` | — | Kimi/Moonshot API key |
| `LLM_MODEL` | — | Model name (for openai provider) |
| `LLM_API_KEY` | — | API key (for openai provider) |
| `LLM_BASE_URL` | — | Base URL for openai provider (blank → api.openai.com) |
| `HEADLESS` | `true` | Run browser in headless mode |
| `MAX_STEPS` | `25` | Max agent steps per task |
| `VISION_MODE` | `auto` | `auto` (heuristic), `dom` (never), or `vision` (always) |
| `VISION_MODELS` | — | Comma-separated vision-capable model names |
| `ALLOWLIST` | — | Comma-separated host substrings (allow) |
| `BLOCKLIST` | — | Comma-separated host substrings (block) |
| `KILL_SWITCH` | `false` | Block all agent actions immediately |

## Architecture

```
src/browser_agent/
├── main.py            CLI entrypoint
├── config.py          Env-driven settings (pydantic-settings)
├── agent/
│   ├── loop.py            Task run loop + streaming
│   └── safe_message_manager.py  DOM injection sanitization
├── models/
│   ├── base.py            ModelAdapter ABC
│   ├── kimi.py            Kimi K2.6 via Moonshot
│   ├── openai_compat.py   Generic OpenAI-compatible adapter
│   ├── registry.py        Provider → adapter lookup
│   └── discovery.py       /v1/models fetcher
├── safety/
│   ├── layer.py           Single choke point: guard()
│   ├── classifier.py      Sensitivity heuristics
│   ├── injection.py       DOM prompt-injection filter
│   ├── gate.py            Human-in-the-loop confirmation (CLI + streaming)
│   ├── policy.py          Site allow/block list
│   └── types.py           PendingAction, SafetyDecision
├── perception/
│   └── vision_router.py   Vision mode routing (dom/auto/vision)
├── tools/
│   └── actions.py         Gated browser actions (navigate, click, type, scroll, extract, done)
└── ui/
    └── server.py          FastAPI + SSE server
```

**Safety pipeline** (every action goes through this, in order):
1. Kill switch → 2. Site policy (allow/block list) → 3. Sensitivity classifier → 4. Human confirmation gate

## Adding a model provider

1. Subclass `ModelAdapter` in `src/browser_agent/models/` — implement `chat_model()` returning a browser-use `BaseChatModel`
2. Register the class in `_ADAPTERS` dict in `registry.py`
3. Set `supports_vision` manually (no auto-detection)

## Dev workflow

```bash
uv run ruff check .          # lint
uv run pytest                # test (no typecheck step)
```

See `AGENTS.md` for gotchas and conventions.
