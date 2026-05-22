# Provenance log — `lit-review.md`

- **Topic:** Frontier of mechanistic interpretability tooling and diagnostics for transformer LMs, 2024-2026.
- **Slug:** `mech-interp-tooling-2024-2026`
- **Run date:** 2026-05-22
- **Plan file:** `/home/vishsangale/workspace/circuitry/docs/v0.9-research/.plan-mech-interp-tooling-2024-2026.md`

## Tools used
- `WebSearch` — broad discovery across all six themes.
- `WebFetch` — direct retrieval of arXiv abstract pages and key GitHub READMEs for primary-source anchoring.
- (Considered but not used: `arxiv` skill, `huggingface-skills:huggingface-papers` — WebSearch + WebFetch on arxiv.org/abs URLs returned enough metadata and abstract content for the cited claims; deeper paper-body extraction was not needed within the 20-minute budget.)

## URLs retrieved

### Fetched via WebFetch (primary anchoring)
- https://arxiv.org/abs/2404.16014  — Gated SAEs abstract; authors, date, shrinkage claim, "half as many firing features" Pareto result.
- https://arxiv.org/abs/2408.05147  — Gemma Scope abstract; authors, JumpReLU architecture, Gemma 2 2B/9B/27B coverage.
- https://github.com/jbloomAus/SAELens — SAELens README; confirmed `SAE.from_pretrained()` API, backend-agnostic claim.
- https://github.com/openai/transformer-debugger — Transformer Debugger README; GPT-2 only, components shipped.

### Searched and surfaced (snippet-level, claims cross-checked across multiple result URLs in the same search)
- arxiv.org/abs/2407.14435 — JumpReLU SAEs (Rajamanoharan et al. 2024).
- arxiv.org/abs/2406.04093 — Scaling and evaluating SAEs / TopK (Gao et al., OpenAI 2024).
- arxiv.org/abs/2403.00745 — AtP* (Kramár et al. 2024).
- arxiv.org/abs/2403.19647 — Sparse Feature Circuits (Marks et al. 2024).
- arxiv.org/abs/2304.14997 — ACDC (Conmy et al. 2023).
- arxiv.org/abs/2303.08112 — Tuned Lens (Belrose et al. 2023).
- arxiv.org/abs/2311.04897 — Future Lens (Pal et al. 2023).
- arxiv.org/abs/2209.11895 — Induction Heads (Olsson et al. 2022).
- arxiv.org/abs/2310.04625 — Copy Suppression (McDougall et al. 2023).
- arxiv.org/abs/2301.05217 — Progress Measures for Grokking (Nanda et al. 2023).
- arxiv.org/abs/2407.14561 — nnsight / NDIF (Fiotto-Kaufman et al. 2024).
- arxiv.org/abs/2403.07809 — pyvene (Wu et al. 2024).
- arxiv.org/abs/2412.17626 — SAE-Track (Xu et al. 2024).
- arxiv.org/abs/2508.15841 — Developmental Interpretability review (2025).
- github.com/AlignmentResearch/tuned-lens — tuned-lens library.
- github.com/ArthurConmy/Automatic-Circuit-Discovery — ACDC code.
- github.com/Aaquib111/edge-attribution-patching — EAP code (Syed et al.).
- github.com/koayon/atp_star — AtP* PyTorch/nnsight port.
- github.com/ndif-team/nnsight — nnsight repo.
- github.com/stanfordnlp/pyvene — pyvene repo.
- huggingface.co/google/gemma-scope — Gemma Scope SAE checkpoints.
- huggingface.co/papers/2403.00745 — AtP* HF paper page.
- transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html — Anthropic Induction Heads writeup.

## Sources: consulted / accepted / rejected

**Accepted** (cited inline in `lit-review.md`): all arXiv IDs and GitHub URLs listed above.

**Rejected / not used**
- aimodels.fyi summary of 2404.16014 — secondary aggregator; superseded by direct arXiv fetch.
- emergentmind.com summaries — secondary; useful for orientation only.
- themoonlight.io "Literature Review" pages — auto-generated summaries; not authoritative.
- syncedreview.com Gemma Scope writeup — popular-press; superseded by arXiv + HF Hub.
- analyticsindiamag.com and aibase.com Transformer Debugger writeups — popular-press; superseded by GitHub README.
- arxiv.org/abs/2511.14465 (nnterp) — surfaced but not cited; mentioned only in narrative context (nnsight/pyvene gap), not as a primary claim.

## Verification status
- **FATAL issues found and fixed:** 0.
- **MAJOR issues noted:** 1 — exact Gemma Scope SAE count: the search snippet says "more than 400 SAEs / 30M features"; the WebFetch of the abstract did not confirm a precise total. Report uses "400+" qualifier rather than a hard number.
- All inline arXiv URLs in `lit-review.md` are of the form `arxiv.org/abs/<id>` where `<id>` was confirmed to exist via WebSearch result listings or direct WebFetch.

## Notes on coverage tradeoffs
- Depth was prioritized on themes 1 (SAEs), 2 (patching), and 6 (tooling landscape), per the user's "depth over breadth" instruction.
- Themes 3-5 received single-pass coverage with 2-3 anchors each, which is sufficient for the user's stated "2-4 citations per theme" target.
- Anthropic's "Towards Monosemanticity" is cited from memory (transformer-circuits.pub, no arXiv ID); marked as orientation/prior-work rather than a primary v0.9 driver.
