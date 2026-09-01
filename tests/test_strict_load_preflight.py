"""RD5 gate: exact strict-load preflight against the real driver checkpoint.

This is the empirical resolution of Resolved-Decision 5 -- build the architecture
at the checkpoint's ``class_nb`` and confirm the state-dict key audit is entirely
clean (no missing / unexpected / mismatched keys). If it is, no adapter machinery
is needed and ``strict_load_with_audit`` can load the checkpoint as-is.

It is **opt-in** (guarded by ``RUN_CHECKPOINT_PREFLIGHT=1``) because it:
  * reads a ~1 GB checkpoint off the shared mount, and
  * constructs ``MerlinArchitecture``, which downloads resnet152 (torchvision) and
    ``yikuan8/Clinical-Longformer`` (HuggingFace) -- neither belongs in the fast
    unit suite / offline CI.

Run explicitly:
    RUN_CHECKPOINT_PREFLIGHT=1 uv run --with pytest pytest \
        tests/test_strict_load_preflight.py -s
"""

import os

import pytest
import torch

from merlin.models.build import MerlinArchitecture
from merlin.models.load import audit_state_dict

# Driver checkpoint (contrastive-3d-onc i3_resnet_clinical_longformer, epoch 97,
# pixdim (1,1,1.5) / spatial 336x336x320 / top_centered / class_nb 1876).
DRIVER_CHECKPOINT = (
    "/mnt/su-vista-uscentral1/chaudhari_lab/merlin/results/models/"
    "i3_resnet_clinical_longformer_best_clip_08-05-2026_17-24-06_epoch_97.pt"
)
DRIVER_CLASS_NB = 1876

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_CHECKPOINT_PREFLIGHT") != "1",
    reason="set RUN_CHECKPOINT_PREFLIGHT=1 (needs mount checkpoint + HF/torchvision downloads)",
)


def test_driver_checkpoint_audit_is_clean():
    assert os.path.exists(DRIVER_CHECKPOINT), f"missing checkpoint {DRIVER_CHECKPOINT}"

    model = MerlinArchitecture(ImageEmbedding=True, class_nb=DRIVER_CLASS_NB)
    state_dict = torch.load(DRIVER_CHECKPOINT, map_location="cpu")

    audit = audit_state_dict(model, state_dict)
    # Print for the runner log so the RD5 result is captured even on success.
    print("\nRD5 strict-load audit against driver checkpoint:")
    for name, keys in audit.items():
        print(f"  {name}: {len(keys)}" + (f" -> {keys[:10]}" if keys else ""))

    assert audit["missing"] == [], f"missing keys: {audit['missing']}"
    assert audit["unexpected"] == [], f"unexpected keys: {audit['unexpected']}"
    assert audit["mismatched"] == [], f"mismatched keys: {audit['mismatched']}"
