# circuitry — project instructions for Claude

## Project goal

`circuitry` is a standalone Python library providing mechanistic-interpretability diagnostics for PyTorch (weight / activation / gradient / spectral primitives + a `Recorder` workflow for live training-time capture + a `scan` workflow for post-hoc analysis on saved checkpoints). Modality-agnostic core; per-modality recipes for LLM, vision, and recsys. MIT-licensed; public on GitHub.

**The design contract is `docs/design.md`. Read it before substantive work.** Sections of particular note:

- §3 — Repository structure and CI-enforced layering rules.
- §4 — Public API (primitives, Recorder, Recipe escape hatches, MetricWriter protocol).
- §10 — Performance budget (≤10% wall-clock at default settings).
- §11 — Multi-process design notes (v1 single-process; v2 additive path).

`CHANGELOG.md` is the release log. Active implementation plans, when in flight, live under `docs/` (e.g. `docs/plan-m3.md` for any future milestone). Historical plans are kept in git history rather than the working tree.

## Agent delegation strategy

Claude (Opus) orchestrates, implementation goes to Haiku/Sonnet subagents, large-context reads and reviews go to Gemini.

| Task | Agent | Tool / model |
|------|-------|---------------|
| Implement a plan task (write files, run tests) | **Claude subagent** | `Agent` with `subagent_type: "claude"`, `model: "haiku"` for well-specified tasks, `"sonnet"` for tasks that need judgment |
| Read + summarize large code sections | **Gemini** | `mcp__gemini-bridge__consult_gemini_with_files` |
| Spec / code review | **Gemini** | `mcp__gemini-bridge__consult_gemini_with_files`, `model="flash"` for quick, `"pro"` for deep |
| Locate code by symbol / pattern | **Explore agent** | `Agent` with `subagent_type: "Explore"` |
| Orchestration, planning, user dialog | **Claude (this session)** | direct |

Use the `superpowers:subagent-driven-development` skill for plan execution.

## Library invariants (CI-enforced)

These are non-negotiable; don't propose changes without amending `docs/design.md` first:

1. **`core/` MUST NOT import from `recorder/`, `recipes/`, `writers/`, or `cli/`.** Primitives are pure functions with no I/O.
2. **`recipes/` MUST NOT import from `cli/`.**
3. **The package MUST NOT import from any consumer codebase.** Reverse-dependency rule — circuitry is the consumed dep, never the consumer. `tests/test_layering.py` enforces this with an explicit forbidden-imports list.
4. **No hidden `.cuda()` calls in `core/`.** Primitives stay device-deterministic.

## Environment

- Python: 3.12.
- Venv: `/home/vishsangale/workspace/circuitry/venv/`. **Always use full venv paths** — never `source venv/bin/activate`:
  - `venv/bin/python` instead of `python` or `python3`
  - `venv/bin/pytest` instead of `pytest`
  - `venv/bin/pip` instead of `pip`
- Working directory: `/home/vishsangale/workspace/circuitry`.

## Workflow

### Before suggesting next steps
1. `git log --oneline -10` for recent history.
2. Read `docs/design.md` (and any active plan under `docs/`).
3. Only then propose work — do not duplicate something already done.

### Before committing
1. Check whether `docs/design.md` (or any active plan under `docs/`) needs updating to match the code change.
2. Update docs first, include in the same commit if both changed.
3. Commit message scope is the area being touched (`feat(core)`, `feat(recorder)`, `feat(recipes)`, `test(...)`, `docs(...)`, `chore(...)`).

### Output conventions
- Always print the full absolute path of any file you create or write.
