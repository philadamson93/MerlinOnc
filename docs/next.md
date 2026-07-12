# MerlinOnc — Next

## Active

- [ ] 2026-05-19 — Structure each batch to have a better positive/negative pair ratio (false-positive control during contrastive training).

## Backlog

- [ ] 2026-07-12 — **Re-lock `uv.lock` for the peft/transformers import incompatibility (Finding F1).** Running MerlinOnc standalone via its own `uv.lock` hits an import error (peft/transformers version incompat); surfaced during the TotalSeg organ-localization Phase-2 GPU Step-3 landing. The vista-ct consumer path imports `merlin` from its *unlocked* env, which floated to a working set and ran clean — so this is confined to MerlinOnc's own locked env, not a consumer gate. **Proven-working set: transformers 4.53.3 + peft 0.19.1** — re-lock to that (or a newer compatible pair). Context: `vista-ct/docs/vm-status/2026-07-10-totalseg-organ-localization-phase2.md` (Finding F1).

## Done
