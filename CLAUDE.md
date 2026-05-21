# circuitry — project instructions for Claude

## Project goal

`circuitry` is a standalone Python library being extracted from `~/workspace/mendu` (and partially from `~/workspace/latent-superpowers-inspect`) into a reusable, public, MIT-licensed package. The library provides mechanistic-interpretability diagnostics (weight / activation / gradient / spectral primitives + a `Recorder` workflow for live use + a `scan` workflow for post-hoc analysis) that works across LLM, vision, and recsys models.

**The design contract is `docs/design.md`. Read it before substantive work.** Sections of particular note:

- §3 — Repository structure and CI-enforced layering rules.
- §4 — Public API (primitives, Recorder, Recipe escape hatches, MetricWriter protocol).
- §7 — Migration plan for cutting mendu over with parity tolerances.
- §10 — Performance budget (≤10% wall-clock at default settings).
- §11 — Multi-process design notes (v1 single-process; v2 additive path).

The implementation plan (when written) lives at `docs/plan.md`.

## Agent delegation strategy

Same pattern as mendu: Claude (Opus) orchestrates, implementation goes to Haiku/Sonnet subagents, large-context reads and reviews go to Gemini.

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
3. **The package MUST NOT import from `mendu`, `rl-recsys`, or any sibling workspace project.** Reverse-dependency rule — circuitry is the consumed dep, never the consumer.
4. **No hidden `.cuda()` calls in `core/`.** Primitives stay device-deterministic.

## Environment

- Python: 3.12 (matches mendu).
- Venv: `/home/vishsangale/workspace/circuitry/venv/`. **Always use full venv paths** — never `source venv/bin/activate`:
  - `venv/bin/python` instead of `python` or `python3`
  - `venv/bin/pytest` instead of `pytest`
  - `venv/bin/pip` instead of `pip`
- Working directory: `/home/vishsangale/workspace/circuitry`.

## Workflow

### Before suggesting next steps
1. `git log --oneline -10` for recent history.
2. Read `docs/design.md` (and `docs/plan.md` once it exists).
3. Only then propose work — do not duplicate something already done.

### Before committing
1. Check whether `docs/design.md` or `docs/plan.md` needs updating to match the code change.
2. Update docs first, include in the same commit if both changed.
3. Commit message scope is the area being touched (`feat(core)`, `feat(recorder)`, `feat(recipes)`, `test(...)`, `docs(...)`, `chore(...)`).

### Output conventions
- Always print the full absolute path of any file you create or write.

## Relationship to mendu

The library is being **extracted from** mendu, not developed alongside it. Once `v0.1.0` is tagged, mendu will install it editable (`venv/bin/pip install -e ~/workspace/circuitry`) and remove its in-tree copies of the inspector + diagnostics. The migration plan is in `docs/design.md` §7.

Until then, mendu's existing code at `mendu/tools/inspect_checkpoint/` and `mendu/paper2/bet2_daleian/analysis/` is the **reference implementation** — when porting numerics into circuitry, parity must hold within the tolerances stated in §7 Phase M2.
