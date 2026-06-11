# swe-agent

An autonomous coding agent that resolves real GitHub issues, benchmarked on
**SWE-bench Lite** (300 real Python issues, each with hidden verifying tests).

This is a portfolio project. The goal is a **citable success-rate number plus a
thorough failure analysis** on a free, open model — not a state-of-the-art
score. A modest score is expected and fine.

> **Status:** Phase 0 complete (project skeleton + provider-agnostic LLM
> client). The agent itself is not built yet. See the [phase plan](#phase-plan).

---

## High-level design

```
                ┌──────────────────────────────────────────────┐
                │                  Agent loop                    │   (Phase 2)
                │  plan → choose tool → act → observe → repeat   │
                └───────────────┬───────────────┬───────────────┘
                                │               │
                  calls tools   │               │  calls the model
                                ▼               ▼
                ┌──────────────────────┐  ┌──────────────────────┐
                │      Tool layer       │  │     LLM layer         │  ← Phase 0
                │ read_file, list_dir,  │  │ LLMClient (interface) │
                │ search_code,          │  │   └─ GroqClient       │
                │ edit_file, run_tests  │  │ provider-agnostic     │
                │      (Phase 1)        │  │ swap via config only  │
                └───────────┬──────────┘  └──────────────────────┘
                            │ runs inside
                            ▼
                ┌──────────────────────┐
                │   Docker sandbox      │   (Phase 3)
                │ one clean container   │
                │   per task            │
                └───────────┬──────────┘
                            │ scored by
                            ▼
                ┌──────────────────────┐
                │  Evaluation runner    │   (Phase 4-5)
                │ run subset → log      │
                │ pass/fail + reason →  │
                │ failure categories →  │
                │ summary table         │
                └──────────────────────┘
```

**The central design requirement is the provider-agnostic LLM layer.** Nothing
above the LLM layer knows it's talking to Groq. The agent loop depends only on
the `LLMClient` interface and a handful of neutral dataclasses (`Message`,
`ToolSpec`, `ToolCall`, `LLMResponse`). Swapping Groq's free Llama for a
stronger paid API later is a `config.toml` edit, not a code change.

---

## Repository structure

```
agent/
├── README.md
├── pyproject.toml          # deps + tooling config (pytest, mypy)
├── config.toml             # provider/model/rate-limit settings — edit to swap providers
├── .env.example            # template for the gitignored .env (holds GROQ_API_KEY)
├── .gitignore
├── src/swe_agent/
│   ├── config.py           # loads config.toml + API key from env
│   ├── llm/                # ← Phase 0 (this phase)
│   │   ├── base.py         #   provider-neutral interface + dataclasses
│   │   ├── groq_client.py  #   Groq implementation (only file that knows OpenAI wire format)
│   │   └── factory.py      #   build_llm_client(config) — the one switch point
│   ├── utils/
│   │   └── retry.py        # retry-with-exponential-backoff (rate-limit handling)
│   ├── tools/              # Phase 1: the five agent tools
│   ├── agent/              # Phase 2: the agent loop
│   ├── sandbox/            # Phase 3: Docker isolation
│   └── eval/               # Phase 4-5: SWE-bench harness, runner, scoring, analysis
└── tests/                  # pytest; LLM tests use fakes, never the network
```

---

## Dependencies & justification

Kept deliberately small.

| Dependency       | Why it's here | Alternative rejected |
|------------------|---------------|----------------------|
| `openai`         | Groq exposes an OpenAI-compatible endpoint; the SDK gives us robust tool-call serialization for free, and pointing it at a different `base_url` is most of "swap the provider". | Raw `httpx` calls — more code to maintain (request building, error types, tool-call parsing) for little gain, since we already wrap it behind our own interface. |
| `python-dotenv`  | Load `GROQ_API_KEY` from a gitignored `.env` in local dev. | Manual `export` only — works, but easy to forget; dotenv is tiny and standard. |
| `pytest` (dev)   | Test runner. | stdlib `unittest` — more boilerplate, weaker fixtures. |
| `mypy` (dev)     | Enforce the strict type hints used throughout. | None; type checking is a stated requirement. |

Things we intentionally **did not** add:
- **No YAML lib** — config is TOML, read with the stdlib `tomllib` (Python 3.11+).
- **No `tenacity`** — the backoff logic we need is ~30 lines and we want an
  injectable `sleep` for instant tests, so it's hand-rolled in `utils/retry.py`.
- **No `requests`** — `openai` brings its own HTTP stack.

---

## Configuration & secrets

- **`config.toml`** holds non-secret settings: provider, model, base URL,
  temperature, token limits, and rate-limit knobs. **Editing this file is how
  you swap providers** — no code change.
- **API keys are never in the repo.** `config.toml` stores only the *name* of
  the env var (`api_key_env`); the value is read from the environment (and
  `.env` locally, which is gitignored). `LLMConfig.__repr__` masks the key so it
  can't leak into logs.

```bash
cp .env.example .env       # then paste your key
# get a free key at https://console.groq.com/keys
```

To swap to a different OpenAI-compatible provider later, edit `config.toml`:
```toml
[llm]
provider    = "groq"          # add a new impl + registry entry for non-OpenAI APIs
model       = "..."
base_url    = "..."
api_key_env = "SOME_OTHER_KEY"
```

---

## Where the rate limits bite

Groq's free tier limits both **requests per minute (RPM)** and **tokens per
minute (TPM)** per model. An agent loop is bursty: a single SWE-bench task can
take 10–30+ model calls (plan, then many tool-use turns), and a benchmark sweep
multiplies that across tasks. Three mitigations are already wired into the LLM
layer:

1. **Fixed inter-call delay** (`delay_between_calls_s`) — a pause after every
   successful call, the simplest reliable way to stay under an RPM cap.
2. **Retry with exponential backoff** (`max_retries`, `initial_backoff_s`,
   `max_backoff_s`) — on HTTP 429 and transient 5xx/network errors. 4xx errors
   like bad-request/auth are *not* retried (that would just burn budget).
3. **Run a tiny subset first** — the Phase 4 runner is built to score 3–5 tasks
   before any full sweep, to bound time and token spend while iterating.

Likely pain points to watch as later phases land: TPM exhaustion from large file
contents stuffed into prompts (mitigation: read/grep with offsets, not whole
files), and long tool-call chains on hard tasks (mitigation: a per-task step
budget in Phase 2).

---

## Main risks (and how we'll manage them)

- **Weak open model → low solve rate.** Expected. The deliverable is the number
  + failure analysis, not SOTA. Phase 5 categorizes *why* tasks fail.
- **Docker not installed locally.** Phase 3 needs it; `docker` is currently
  absent on this machine. Install Docker Desktop before Phase 3.
- **SWE-bench environment setup is fiddly.** Each task needs the right repo at
  the right commit with deps installed. Phase 3 leans on the official
  per-task Docker images to keep this reproducible.
- **Rate limits stretching wall-clock time.** A throttled sweep is slow; we
  iterate on small subsets and only do a full run once the loop is stable.
- **Non-determinism.** `temperature = 0` reduces but doesn't eliminate it; runs
  are logged with enough detail (Phase 4) to make results reproducible/citable.

---

## Setup & running tests

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest          # run the suite (no network/API key needed — LLM tests use fakes)
mypy            # static type check
```

---

## Phase plan

- **Phase 0 — ✅ skeleton + provider-agnostic LLM client.** (this phase)
- **Phase 1 — tool layer** (read_file, list_dir, search_code, edit_file,
  run_tests) with unit tests against a small dummy repo. No LLM.
- **Phase 2 — agent loop** that plans, chooses tools, and iterates; solve ONE
  hand-picked issue end-to-end.
- **Phase 3 — Docker sandbox** + SWE-bench Lite task loading/harness.
- **Phase 4 — evaluation runner** scoring a batch, logging structured
  pass/fail + reason; easy to run on a 3–5 task subset first.
- **Phase 5 — failure categorization** (couldn't localize / wrong fix / broke
  other tests / out of steps) + a summary table for a blog post.
