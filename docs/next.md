# MerlinOnc — Next

## Active

- [ ] 2026-05-19 — Structure each batch to have a better positive/negative pair ratio (false-positive control during contrastive training).
- [ ] 2026-07-21 — **MerlinOnc v2.2 promotion, Track B — code done + Codex `/review-implementation`'d clean, HF upload still pending.** `merlin/models/load.py` `MODEL_CONFIGS`: bumped `merlin_onc` alias to v2.2, pinned `merlin_onc_v2_1`/`merlin_onc_v2_2`, made `class_nb` per-entry (1876), added a `checkpoint_key` param so a caller can actually select a pinned version. Canonical plan: `vista-ct/docs/plans/promote-merlinonc-v2.2-checkpoint.md` (Track B section). **Blocked on:** uploading both `.pt` files (v2.1 `i3_resnet_clinical_longformer_best_clip_04-03-2026_04-45-12_epoch_99.pt`, v2.2 `..._07-01-2026_18-38-12_epoch_66.pt`) to HF `philadamson93/MerlinOnc` — non-code, needs HF credentials, not done this session. Track B smoke (`Merlin(MerlinOnc=True)` + `checkpoint_key="merlin_onc_v2_1"`) can't run until that upload lands.

## Backlog

- [ ] 2026-07-12 — **Re-lock `uv.lock` for the peft/transformers import incompatibility (Finding F1).** Running MerlinOnc standalone via its own `uv.lock` hits an import error (peft/transformers version incompat); surfaced during the TotalSeg organ-localization Phase-2 GPU Step-3 landing. The vista-ct consumer path imports `merlin` from its *unlocked* env, which floated to a working set and ran clean — so this is confined to MerlinOnc's own locked env, not a consumer gate. **Proven-working set: transformers 4.53.3 + peft 0.19.1** — re-lock to that (or a newer compatible pair). Context: `vista-ct/docs/vm-status/2026-07-10-totalseg-organ-localization-phase2.md` (Finding F1).

## Done
