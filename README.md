# swe-agent

An autonomous coding agent that resolves real GitHub issues, benchmarked on
**SWE-bench Lite** (300 real Python issues, each with hidden verifying tests).

This is a portfolio project. The goal is a **citable success-rate number plus a
thorough failure analysis** on a free, open model — not a state-of-the-art
score. A modest score is expected and fine.

> **Status:** the full pipeline (Phases 0–5) is built and tested — agent loop,
> per-instance Docker environment, official SWE-bench grading, a batch runner,
> and a failure-analysis report. What remains is *running* the benchmark at
> scale, which is gated by the free-tier token budget (see
> [Evaluation workflow](#evaluation-workflow)). See the [phase plan](#phase-plan).

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
                │   Docker sandbox      │   (Phase 3) — the agent edits/tests
                │ one container per     │   inside the real SWE-bench instance
                │   task (/testbed)     │   image, run under amd64 emulation
                └───────────┬──────────┘
                            │ patch graded by
                            ▼
                ┌──────────────────────┐
                │  Evaluation pipeline  │   (Phases 3c–5)
                │ official grader →     │
                │ batch runner → log    │
                │ per-instance JSON →   │
                │ failure analysis →    │
                │ summary report        │
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
autonomous-issue-resolver/
├── README.md
├── LICENSE                 # MIT
├── pyproject.toml          # deps + tooling config (pytest, mypy)
├── config.toml             # provider/model/rate-limit settings — edit to swap providers
├── .env.example            # template for the gitignored .env (one API key)
├── .gitignore
├── .github/workflows/ci.yml  # CI: mypy --strict + pytest on 3.11/3.12
├── scripts/
│   └── smoke_test.py       # cheap end-to-end check; most stages cost 0 tokens
├── src/swe_agent/
│   ├── config.py           # loads config.toml + API key from env
│   ├── task.py             # Task: the unit of work (id + problem statement)
│   ├── dataset.py          # SWE-bench Lite loading (SWEBenchInstance, load_swebench_lite)
│   ├── llm/                # Phase 0: provider-agnostic LLM layer
│   │   ├── base.py         #   provider-neutral interface + dataclasses
│   │   ├── groq_client.py  #   Groq implementation (only file that knows OpenAI wire format)
│   │   └── factory.py      #   build_llm_client(config) — the one switch point
│   ├── utils/retry.py      # retry-with-exponential-backoff (rate-limit handling)
│   ├── tools/              # Phase 1: the agent tools + CommandExecutor seam
│   ├── agent/              # Phase 2: the agent loop + CLI
│   │   ├── loop.py         #   the ReAct loop (plan → act → observe → repeat)
│   │   └── compaction.py   #   elide old tool outputs to cut tokens per call
│   ├── sandbox/            # Phase 3: Docker isolation
│   │   ├── docker.py       #   DockerSandbox + DockerExecutor (CommandExecutor over a container)
│   │   └── environment.py  #   SWEBenchEnvironment: provision a real instance to solve in
│   └── eval/               # Phases 3c–5: grading, runners, analysis
│       ├── grading.py      #   official SWE-bench grading (grade(instance, patch))
│       ├── runner.py       #   solve_and_grade(): one instance, end to end
│       ├── batch.py        #   run_batch(): many instances, resumable + scored
│       ├── analysis.py     #   classify outcomes → AnalysisReport → Markdown
│       ├── dashboard.py    #   render a self-contained offline dashboard.html
│       ├── cli.py          #   run + grade ONE SWE-bench instance
│       ├── batch_cli.py    #   run + grade a BATCH
│       └── analyze_cli.py  #   turn result records into the report
└── tests/                  # pytest; LLM tests use fakes, Docker tests skip without a daemon
```

---

## Dependencies & justification

Kept deliberately small.

| Dependency       | Why it's here | Alternative rejected |
|------------------|---------------|----------------------|
| `openai`         | Groq exposes an OpenAI-compatible endpoint; the SDK gives us robust tool-call serialization and 429-retry (honouring `Retry-After`) for free, and pointing it at a different `base_url` is most of "swap the provider". | Raw `httpx` calls — more code to maintain for little gain, since we wrap it behind our own interface. |
| `python-dotenv`  | Load `GROQ_API_KEY` from a gitignored `.env` in local dev. | Manual `export` only — easy to forget; dotenv is tiny and standard. |
| `datasets`       | Load SWE-bench Lite (the 300 tasks + their gold tests) from the Hugging Face hub. | Hand-fetching parquet — reinventing what `datasets` does well. |
| `swebench`       | The **authoritative** source of per-instance Docker image names, the `eval_script`, and the official grader. Used so our pass/fail is the *same* number the SWE-bench leaderboard uses. | Reimplementing image naming + grading — fragile and non-citable; the whole point is a faithful score. |
| `pytest` (dev)   | Test runner. | stdlib `unittest` — more boilerplate, weaker fixtures. |
| `mypy` (dev)     | Enforce the strict type hints used throughout. | None; type checking is a stated requirement. |

Things we intentionally **did not** add:
- **No YAML lib** — config is TOML, read with the stdlib `tomllib` (Python 3.11+).
- **No `tenacity`** — the backoff logic we need is ~30 lines and we want an
  injectable `sleep` for instant tests, so it's hand-rolled in `utils/retry.py`.
- **No `docker` SDK** — we shell out to the `docker` CLI from `sandbox/docker.py`,
  consistent with how the local executor uses `subprocess`.

---

## Configuration & secrets

- **`config.toml`** holds non-secret settings: provider, model, base URL,
  temperature, token limits, rate-limit knobs, and the agent step budget.
  **Editing this file is how you swap providers** — no code change.
- **API keys are never in the repo.** `config.toml` stores only the *name* of
  the env var (`api_key_env`); the value is read from the environment (and
  `.env` locally, which is gitignored). `LLMConfig.__repr__` masks the key.

```bash
cp .env.example .env       # then paste ONE key — whichever provider config.toml names
# a free Groq key (the default) : https://console.groq.com/keys
```

### Using a different provider

Any **OpenAI-compatible** endpoint is a `config.toml` edit — no code change.
`provider` selects the client implementation; `groq`, `gemini`, and `openai` all
map to the same generic OpenAI-compatible client, so they differ only in
`base_url`/`api_key_env`. Use `openai` for OpenAI itself, OpenRouter, DeepSeek,
Together, or a local vLLM/Ollama server:

```toml
[llm]
provider    = "openai"                      # groq | gemini | openai
model       = "gpt-4o-mini"
base_url    = "https://api.openai.com/v1"   # your provider's endpoint
api_key_env = "OPENAI_API_KEY"              # must match the var set in .env
# temperature = 0.0                         # see the caveat below
```

**Sampling-parameter caveat.** `temperature` is only sent when it is present in
`config.toml`. Some providers (notably current Anthropic models) reject sampling
parameters with a 400, so **comment the line out** for those. Leaving it set is
what you want for Groq/Gemini/OpenAI, where a fixed temperature makes runs more
reproducible.

**Anthropic (Claude) is deliberately not wired up.** Its API is not
OpenAI-compatible, so it is *not* aliased to the generic client — setting
`provider = "anthropic"` fails immediately with an explanatory error rather than
breaking mid-run. Adding it means writing a small `AnthropicClient` implementing
the `LLMClient` interface in `src/swe_agent/llm/`, registering it in
`factory.py`, and adding the `anthropic` dependency. Everything above the LLM
layer stays untouched — that's the point of the interface.

A misconfigured provider or a missing key is caught at startup by every CLI,
which prints a one-line error and exits `2` — no traceback, nothing half-run.

---

## Setup & running tests

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest          # the suite (no API key needed; Docker/grading tests skip without a daemon)
mypy            # static type check
```

The benchmark also needs **Docker**. This project was developed on an Apple
Silicon (arm64) Mac: SWE-bench's prebuilt instance images are published for
**x86_64 only**, so we run them under emulation (`docker run --platform
linux/amd64`). That works out of the box with Docker Desktop / colima; the first
run of each instance pulls a **~3.8 GB** image (measured across django, sympy,
sphinx, and flask).

### Smoke test — verify the whole pipeline cheaply

Before spending a token budget on a real sweep, check that everything actually
works end to end:

```bash
python scripts/smoke_test.py --stage free   # 0 tokens: needs Docker, no API key
python scripts/smoke_test.py                # adds a short live agent run
```

The `free` stages cost **nothing**, because the dataset ships each task's *gold*
patch: grading that patch must produce `RESOLVED_FULL`, which exercises Docker
provisioning, amd64 emulation, the official `eval_script`, and the log parser —
the fragile half of the pipeline — without an API key. The `agent` stage then
makes a few real model calls (`--max-steps`, default 4) to confirm the LLM is
reachable and tools execute in-container, and prints exact token usage plus a
projected cost per full run, so you can size a sweep against your quota.

This is the recommended first command after adding a key to `.env`.

---

## Running the agent on a local repo

Point the agent at any local git repository and describe the issue:

```bash
python -m swe_agent.agent.cli --repo ./path/to/repo \
    --issue "subtract(5, 3) returns 8 but should return 2"
# or:  --issue-file bug.txt    --max-steps 15
```

It streams each tool call, then prints the diff it produced. **Note:** this mode
runs commands on the host (no sandbox) — it's for quick local use. The sandboxed,
graded SWE-bench path is below.

---

## Evaluation workflow

The benchmark runs each task inside its official SWE-bench Docker image. The
agent edits a bind-mounted checkout of the repo at `/testbed` (so its host-side
file edits and the container's test runs share files), then the resulting patch
is graded in a **fresh** container using swebench's own `eval_script` + grader.

**Per-repo test runners.** Only **93 of the 300** SWE-bench Lite instances use
pytest. django (114) runs `./tests/runtests.py`, sympy (77) runs `bin/test`, and
sphinx (16) runs `tox`. The agent's `run_tests` tool therefore takes its command
from swebench's own repo/version table (`repo_test_command()`) rather than
assuming pytest — so the agent verifies its fix and checks for regressions using
the same runner the official grader will use. Outside SWE-bench (`agent.cli` on
a local repo) it falls back to pytest.

**One instance, end to end:**
```bash
python -m swe_agent.eval.cli --instance-id pallets__flask-4045 --max-steps 20
# add --results-dir runs/lite to also save a JSON record for analysis
```

**A batch (resumable, scored):**
```bash
python -m swe_agent.eval.batch_cli --limit 5 --open     # first 5 instances
python -m swe_agent.eval.batch_cli --instance-ids a,b   # specific ids
# writes one runs/lite/<instance_id>.json per instance,
# then (re)writes runs/lite/dashboard.html
```
Re-running the same command **resumes** — instances that already have a result
file are skipped. If a run hits a rate/quota limit it **aborts the rest** and
leaves them for the next resume (pass `--keep-going` to override). Every run
regenerates the **HTML dashboard** (below); `--open` pops it open in your browser.

**The report (no API key / Docker needed):**
```bash
python -m swe_agent.eval.analyze_cli --results-dir runs/lite \
    --out runs/lite/report.md --html runs/lite/dashboard.html
```
This classifies every instance into one outcome — `resolved`, or a failure mode
(`regression`, `incomplete_fix`, `no_patch`, `patch_apply_failed`,
`eval_incomplete`, `llm_error`, `run_error`) — and prints the resolve rate, an
outcome breakdown, and a per-repo table (`--out` also saves it as Markdown).

**The dashboard.** `--html` (and every batch run) writes a **self-contained
`dashboard.html`** — inline CSS/JS, data embedded, no server, no external
requests. Open it in any browser: a resolve-rate headline, an outcome-breakdown
chart, a per-repo table, and expandable per-instance rows showing the agent's
diagnosis, tool-call trace, per-test pass/fail, and the diff it wrote.
```bash
open runs/lite/dashboard.html          # macOS  (xdg-open on Linux)
```

### Free-tier reality

Groq's free tier caps **tokens per day (≈100k)** as well as per-minute rates.
Measured on `pallets__flask-4045` with `llama-3.3-70b-versatile`: about
**2,200 tokens per step**, so a full 25-step attempt costs **~55k tokens** — the
transcript is re-sent every step and grows with each file read and tool result.

That means the free daily budget is **less than two complete attempts**, and in
practice runs have died mid-way rather than finishing. The harness is built for
this: it's resumable, so you continue the next day, and a paid tier (or a
different provider via `config.toml`) lifts the ceiling. Use
`scripts/smoke_test.py` to measure the per-step cost of *your* provider before
committing to a sweep.

Disk is the other cost, and it's the bigger one. Instance images are **~3.8 GB
each**, and they're per-*instance*, not per-repo — so a 30-instance sweep pulls
roughly **110 GB** and the full 300 would be over a terabyte. Containers are
removed after each instance, but the images are not, so prune between chunks:

```bash
docker image prune -a -f     # between chunks of ~25 instances
```

Run in chunks and the peak disk stays near 25–30 GB; the download volume is the
part you can't avoid.

---

## Where the rate limits bite

Groq's free tier limits **requests/min (RPM)**, **tokens/min (TPM)**, and
**tokens/day (TPD)** per model. An agent loop is bursty — a single task can take
10–30+ model calls. Mitigations wired into the LLM layer:

1. **Fixed inter-call delay** (`delay_between_calls_s`) — a pause after every
   successful call, easing RPM/TPM pressure.
2. **Two layers of retry on 429/5xx** — the `openai` SDK retries first
   (honouring Groq's `Retry-After`; configured from `max_retries`), and
   `utils/retry.py` is the outer exponential-backoff net. 4xx auth/bad-request
   are *not* retried (that would just burn budget).
3. **Graceful exhaustion** — when retries are spent (e.g. the daily cap), the
   agent loop ends the run with `StopReason.ERROR` instead of crashing, captures
   any partial patch, and the batch aborts cleanly so one dead instance never
   kills the sweep.

TPD is the binding constraint for free-tier sweeps; TPM 429s are common but the
retries absorb them. The deeper lever is fewer tokens per call: the agent loop
**compacts the transcript** before each request, sending only the most recent
tool outputs in full and eliding older ones (`keep_recent_tool_results` in
`config.toml`, see `agent/compaction.py`) — roughly halving tokens on long runs,
which directly raises how many instances a daily budget covers.

---

## Main risks (and how we manage them)

- **Weak open model → low solve rate.** Expected. The deliverable is the number
  + failure analysis, not SOTA; Phase 5 categorizes *why* tasks fail.
- **Free-tier token budget caps sweep size.** ~a few instances/day; the runner
  is resumable, and the provider is swappable via `config.toml`.
- **arm64 host, x86_64 images.** Handled by `--platform linux/amd64` emulation;
  slower but correct.
- **Non-determinism.** `temperature = 0` reduces but doesn't eliminate it; every
  run is logged to a JSON record so results are reproducible/citable.

---

## Phase plan

- **Phase 0 — ✅ skeleton + provider-agnostic LLM client.**
- **Phase 1 — ✅ tool layer** (read_file, list_dir, search_code, edit_file,
  run_shell, run_tests) with unit tests against a dummy repo. No LLM.
- **Phase 2 — ✅ agent loop** that plans, chooses tools, iterates, finishes; CLI;
  verified end-to-end on a local repo.
- **Phase 3a — ✅ Docker sandbox primitive** (`DockerExecutor` over the Phase-1
  `CommandExecutor` seam).
- **Phase 3b — ✅ SWE-bench Lite loading** (`SWEBenchInstance`, `load_swebench_lite`).
- **Phase 3c — ✅ instance environment + official grading.** Provision a real
  instance container the agent solves in (bind-mounted `/testbed`, amd64
  emulation); grade the patch with swebench's own `eval_script` + grader.
- **Phase 4 — ✅ batch runner** — score a subset, persist a JSON record per
  instance; resumable and quota-aware.
- **Phase 5 — ✅ failure analysis** — classify outcomes and render the citable
  resolve-rate + breakdown report.
- **Remaining — run the benchmark** at scale (gated by the free-tier token
  budget) and write up the resulting number + failure analysis.
