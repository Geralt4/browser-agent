# AGENTS.md

## Quickstart

```bash
uv run browser-agent "go to example.com and return the H1"   # CLI task
uv run browser-agent-ui                                       # Web UI on 127.0.0.1:8000
uv run pytest                                                 # all tests
uv run pytest tests/test_injection.py -v                       # single file
uv run ruff check .                                            # lint (no typecheck step)
```

API-key gated tests (`test_hello_world.py`, `test_interaction_llm.py`, `test_injection.py`)
skip automatically when neither `MOONSHOT_API_KEY` nor `LLM_API_KEY` is set in `.env`.

## Architecture summary

Six packages under `src/browser_agent/`: `models/`, `agent/`, `safety/`, `perception/`,
`tools/`, `ui/`. Full design brief is in `CLAUDE.md`. The main CLI entrypoint is
`src/browser_agent/main.py`; the API server is `src/browser_agent/ui/server.py`.

## Critical gotchas

### `from __future__ import annotations` is banned in `tools/actions.py`

browser-use's action registry compares the runtime class of the `browser_session`
parameter. PEP 563 string annotations break that check. The file has a loud comment
at line 11. Any new action file must follow the same rule.

### `_wrap_message_manager()` in `agent/loop.py` is fragile

Lines 16-38 monkey-patch `agent._message_manager` with `InjectionSafeMessageManager`,
re-listing every constructor param. A browser-use upgrade that changes
`MessageManager` params breaks here only. If injection sanitization silently stops
working after a dependency bump, this is the first place to check.

### Don't add new tools without reading the safety pipeline

Every tool routes through `SafetyLayer.guard()` in `safety/layer.py`. If you add a
new tool and don't call `safety.guard()`, that action bypasses the kill switch,
blocklist, sensitivity classifier, and confirmation gate entirely.

## Model adapters

Providers live in `src/browser_agent/models/`. To add one:
1. Subclass `ModelAdapter` (`base.py`) — implement `chat_model()` returning a
   browser-use `BaseChatModel`.
2. Register the class in the `_ADAPTERS` dict in `registry.py`.
3. Set `supports_vision` manually (no auto-detection). For the openai-compat path,
   the user configures `VISION_MODELS` as a comma-separated list.

## Testing conventions

- `conftest.py` loads `dotenv` so both `os.getenv()` and `pydantic-settings` see
  `.env` values — tests use `os.getenv` for skip gates.
- `fixture_url()` serves static HTML from `tests/fixtures/` on a loopback port.
  No test should hit a live external site.
- Use `Config(...)` with kwargs for unit tests; `load_config()` reads `.env` and is
  only for e2e/integration.

## Safety layer

- `SafetyLayer.guard()` is the single choke point: kill switch → allow/block list
  → sensitivity check → confirmation gate. Order matters — kill switch checked first.
- `Config.with_overrides()` returns a **new copy**; it never mutates the original.
- The sensitivity classifier in `classifier.py` is keyword-based only. If you add
  new sensitive action categories, add tests to `test_classifier.py`.

## Extension (Phase 5 — secondary)

The Chrome extension in `extension/` uses Manifest V3 with a native messaging host
for keychain access. The primary dev workflow is the Python CLI/UI; the extension is
a later form factor and not the default path for agent development.
