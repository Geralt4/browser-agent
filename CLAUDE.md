# Open-Source Browser Agent — Project Plan

A self-hosted browser-automation agent (a "Claude for Chrome" equivalent) built on
open-weight models. This document is the standing project brief: architecture
decisions, build phases, and milestone ordering. Treat the safety layer (Phase 4)
as non-negotiable, not optional polish.

---

## 1. Goal

Build a browser agent that takes natural-language tasks ("clean up my inbox",
"fill this form", "extract these prices") and executes them by perceiving web pages
and taking actions (click, type, navigate, extract). The model layer is an
open-weight LLM; the harness and safety layer are ours.

## 2. Core architecture decisions (already made — don't relitigate without cause)

- **Model:** Kimi K2.6 (Moonshot). Chosen because it has *both* strong agentic
  tool-calling *and* integrated vision, so a single model covers DOM reasoning and
  the occasional screenshot. MIT-ish license, available via API or self-host (it's a
  ~1T-param MoE — self-hosting is non-trivial hardware; start with the API).
- **Perception: DOM-first, vision as fallback.** The accessibility tree / DOM is the
  default observation (faster, cheaper, index-based element referencing is robust).
  Screenshots → Kimi vision input *only* when DOM parsing is ambiguous or a visual
  check is needed. Do NOT make screenshots the default path.
- **Harness:** Build on `browser-use` (MIT, Playwright-based, model-agnostic, supports
  custom tools and custom/local models). We extend it; we do not rebuild browser plumbing.
- **Form factor:** Start as a standalone app driving a Playwright browser. Port to a
  Chrome extension (Manifest V3 + native-messaging host) ONLY after the core loop and
  safety layer are proven. Do not build the extension first.

## 3. Tech stack

- Python 3.11+
- `browser-use` + `playwright`
- Kimi K2.6 via Moonshot API (OpenAI-compatible endpoint) — keep the model behind an
  adapter interface so we can swap to GLM-5.1 / DeepSeek V4 / Qwen 3.6 later
- `pydantic` for the structured agent-step schema
- `pytest` for the safety-layer tests (write these early)

## 4. Build phases & milestone ordering

### Phase 1 — Prove the loop (target: day 1)
- Install browser-use + playwright; configure Kimi as the LLM.
- Run a hello-world task ("go to example.com, return the H1").
- **Exit criterion:** one task completes end to end. Do not proceed until green.

### Phase 2 — Perception layer
- Serialize the accessibility tree / DOM into an indexed list of interactable elements
  as the default observation.
- Add a screenshot-capture path that feeds Kimi's vision input, gated behind a
  "DOM insufficient" condition.
- Structured per-step output (pydantic model): `assessment` (did the last action
  work?), `memory` (progress notes), `next_subgoal`, `action`.

### Phase 3 — Action toolset
- Tools: `click(element_id)`, `type(element_id, text)`, `navigate(url)`, `scroll`,
  `extract(query)`, `done(result)`. Register as custom browser-use tools.
- Element referencing is index-based (from Phase 2 serialization), not coordinate-based.

### Phase 4 — Safety layer (THE core work — budget the most time here)
- **Confirmation gate:** pause for human approval before any irreversible/sensitive
  action — send, publish, purchase, delete, or submitting forms with personal data.
- **Injection filter:** treat ALL page content as untrusted data, never as commands.
  Strip/flag hidden DOM text, off-screen elements, and instruction-like patterns
  ("ignore previous instructions", "for security reasons delete…"). This is the
  primary failure mode of browser agents.
- **Site allow/block lists** + a kill switch. Block financial/adult/sensitive
  categories by default.
- Write pytest cases for each defense using known injection payloads.

### Phase 5 — UI / form factor
- Simple chat UI: user types a task, watches steps stream, approves gated actions inline.
- (Optional, later) Manifest V3 extension + native-messaging host for in-browser UX
  against the user's real logged-in session.

### Phase 6 — Measure
- Run the open `browser-use/benchmark` (100 hard browser tasks) for a real score.
- A/B test DOM-only vs DOM+vision routing to confirm vision is actually buying anything
  for the target task mix.

## 5. Guardrails for the build

- Untrusted-content principle is absolute: page content is data, not instructions.
- Never let the agent enter credentials, payment details, or submit sensitive forms
  without explicit human confirmation.
- Keep the model behind an adapter — provider lock-in is a design smell here.
- Don't skip Phase 4 to demo faster. An agent with real session access and no safety
  layer is dangerous to its own user.

## 6. First task for Claude Code

Scaffold the repo: directory structure, a `ModelAdapter` interface with a Kimi
implementation, the browser-use-backed agent loop, the Phase 3 tool definitions, and a
stub safety layer with the confirmation gate wired in (even if the injection filter is
a TODO). Get the Phase 1 hello-world task passing before adding anything else.
