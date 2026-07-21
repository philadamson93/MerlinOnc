Reference: docs/claude_ops.md

# Implementation Feedback: Promote MerlinOnc v2.2 Checkpoint (MerlinOnc / Track B)

## Verdict
Ready to commit. The Track B code and docs match the plan: `merlin_onc` now points at v2.2, v2.1/v2.2 are pinned, per-entry `class_nb` is honored with a 1692 fallback, and `checkpoint_key` is applied before config resolution. The only remaining risk is execution-time verification against the uploaded HuggingFace checkpoint files, which the plan already calls out as pending VM/HF smoke work.

## Plan Coverage
| Slice / section | Status | Evidence: path:line | Notes |
|---|---|---|---|
| Track B in scope now, not deferred | Done | `.CROSS_REPO_PLAN_vista-ct.md:40` | This repo only needed Track B; absence of vista-ct / vista-eval Track A changes is not a MerlinOnc repo gap. |
| Set bare `merlin_onc` alias to v2.2 | Done | `merlin/models/load.py:32` | `merlin_onc` now uses `i3_resnet_clinical_longformer_best_clip_07-01-2026_18-38-12_epoch_66.pt` with `repo_id: philadamson93/MerlinOnc`. |
| Pin `merlin_onc_v2_1` | Done | `merlin/models/load.py:38` | Pinned to `i3_resnet_clinical_longformer_best_clip_04-03-2026_04-45-12_epoch_99.pt` with `class_nb: 1876`. |
| Pin `merlin_onc_v2_2` | Done | `merlin/models/load.py:44` | Pinned to the same July 2026 checkpoint as the latest alias, with `class_nb: 1876`. |
| Drop older Oct-2025 checkpoint from `MODEL_CONFIGS` | Done | `merlin/models/load.py:16` | The old `i3_resnet_clinical_longformer_best_clip_10-08-2025_03-41-48_epoch_99.pt` is absent from `MODEL_CONFIGS`; it appears only in docs as retired-history text at `docs/models.md:58`. |
| Make `class_nb` per-entry for v2.1/v2.2 | Done | `merlin/models/load.py:36` | Both v2.1/v2.2 entries set `class_nb: 1876`; non-MerlinOnc entries intentionally omit it and fall back to 1692. |
| Preserve explicit caller `class_nb` override | Done | `merlin/models/load.py:96` | Resolution order is explicit argument first, then config value, then 1692 fallback. |
| Preserve boolean-flag task behavior | Done | `merlin/models/load.py:73` | Existing flag-derived tasks remain: `MerlinOnc=True` -> `merlin_onc`, `RadiologyReport=True` -> `report_generation`, `FiveYearPred=True` -> `five_year_disease_prediction`, otherwise `default`. |
| Add a callable way to select pinned entries | Done | `merlin/models/load.py:62` | This is a defensible addition: before the change, `MerlinOnc=True` always resolved to the single `merlin_onc` key (`git show 1107622:merlin/models/load.py:53`), so there was no constructor path to `merlin_onc_v2_1`. |
| Apply `checkpoint_key` before `_config` lookup | Done | `merlin/models/load.py:86` | Ordering is correct: `self.task` is overridden at lines 86-87, then `_config = MODEL_CONFIGS[self.task]` occurs at line 91. |
| Update `docs/models.md` registry and constructor docs | Done | `docs/models.md:34` | Docs include the v2.2 latest alias, pinned v2.1/v2.2 entries, `checkpoint_key`, and `class_nb` behavior. |
| Add pinned MerlinOnc usage example | Done | `docs/models.md:123` | Example shows both `checkpoint_key="merlin_onc_v2_1"` and `checkpoint_key="merlin_onc_v2_2"`. |

## Critical Drift
- None.

## Missing Pieces
- None.

## Contract Violations
- None.

## Test Gaps
- Track B smoke remains unrun | `.CROSS_REPO_PLAN_vista-ct.md:306` | The plan calls for `Merlin(MerlinOnc=True)` loading v2.2 and `checkpoint_key="merlin_onc_v2_1"` loading the April-2026 weights. This audit verifies the code path statically, but did not prove HF artifact availability or checkpoint shape compatibility at runtime.
- No automated regression around `checkpoint_key` / `class_nb` resolution | `merlin/models/load.py:86` | Not required by the plan, but a lightweight unit test with a stubbed builder/download would catch future ordering regressions before `_config` lookup.

## Defensible Deviations
- Added `checkpoint_key: Optional[str] = None` beyond the literal Files-to-Modify text | `merlin/models/load.py:62` | This is necessary and correctly scoped. The prior constructor derived `self.task` only from boolean flags, so named registry entries could not be selected directly; the new override makes the plan's v2.1 smoke callable.
- Changed `class_nb` from `int = 1692` to `Optional[int] = None` | `merlin/models/load.py:63` | This preserves existing behavior for default/report/five-year paths through the 1692 fallback, while allowing MerlinOnc registry entries to supply 1876 without requiring callers to pass it.

## Suggested Code Edits
- None required. Optional cleanup: `docs/models.md:16` and `docs/models.md:62` have stale approximate "Location" line numbers after the added registry entries; they do not affect the constructor/registry documentation added for this plan.

## Questions For The Author
- Confirm before runtime smoke that both `i3_resnet_clinical_longformer_best_clip_04-03-2026_04-45-12_epoch_99.pt` and `i3_resnet_clinical_longformer_best_clip_07-01-2026_18-38-12_epoch_66.pt` have been uploaded to `philadamson93/MerlinOnc`; the code assumes those filenames are available from that repo.

## Audit Trail
- `.CROSS_REPO_PLAN_vista-ct.md`
- `docs/claude_ops.md`
- `docs/inference.md`
- `docs/models.md`
- `documentation/demo.py`
- `documentation/inference.md`
- `documentation/report_generation.md`
- `documentation/report_generation_demo.py`
- `merlin/models/build.py`
- `merlin/models/i3res.py`
- `merlin/models/load.py`
- `merlin/models/radiology_report_generation.py`
- `README.md`
- `CLAUDE.md`
