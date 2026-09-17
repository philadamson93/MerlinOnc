"""The one narrow exception to the fail-closed strict load: an absent text tower.

A CLIP model trained against PRECOMPUTED text vectors (e.g. frozen Gemini findings
embeddings) ships no Clinical-Longformer body, so its checkpoint legitimately lacks
every ``encode_text.text_encoder.*`` key and carries a differently-shaped
``encode_text.linear_layer.weight``. In image-embedding mode that tower never runs
(``MerlinArchitecture.forward`` returns ``encode_image(image)`` and raises if given
text), so leaving it at init cannot change an output value.

These tests pin BOTH halves of that: the exception applies where it should, and --
more importantly -- it does not widen into tolerance for anything that actually runs.
A toy module keeps them in the fast suite (the real architecture pulls resnet152 +
Clinical-Longformer; see test_strict_load_preflight.py for the opt-in real-checkpoint
gate).
"""

import pytest
import torch
from torch import nn

from merlin.models.load import (
    audit_state_dict,
    audit_state_dict_excluding_text_tower,
    strict_load_with_audit,
)


class _ToyImageEncoder(nn.Module):
    def __init__(self, class_nb=4):
        super().__init__()
        self.backbone = nn.Linear(8, 8, bias=False)
        self.classifier = nn.Linear(8, class_nb, bias=False)


class _ToyTextBody(nn.Module):
    """Stands in for the Clinical-Longformer body: the part a precomputed-text
    checkpoint does not ship, nested under ``encode_text.text_encoder.*`` exactly
    as the real one is."""

    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Linear(6, 6, bias=False)


class _ToyTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = _ToyTextBody()


class _ToyClip(nn.Module):
    """Mirrors MerlinArchitecture's shape: encode_image + encode_text + logit_scale."""

    def __init__(self, class_nb=4, text_proj_in=6):
        super().__init__()
        self.encode_image = _ToyImageEncoder(class_nb)
        self.encode_text = _ToyTextEncoder()
        self.encode_text.linear_layer = nn.Linear(text_proj_in, 5, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]))


def _full_checkpoint(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


def _precomputed_text_checkpoint(model, text_proj_in=12):
    """What a precomputed-text-tower run actually writes: no text_encoder body, and a
    linear_layer sized for the external embedding width."""
    sd = _full_checkpoint(model)
    sd = {k: v for k, v in sd.items() if not k.startswith("encode_text.text_encoder.")}
    sd["encode_text.linear_layer.weight"] = torch.randn(5, text_proj_in)
    return sd


def test_clean_checkpoint_still_loads_strictly():
    """The common path is untouched: a matching checkpoint loads with no exception."""
    model = _ToyClip()
    strict_load_with_audit(model, _full_checkpoint(_ToyClip()), context="clean")


def test_absent_text_tower_is_refused_by_default():
    """Default stays fail-closed; the exception must be asked for explicitly."""
    model = _ToyClip()
    with pytest.raises(RuntimeError, match="key audit is non-empty"):
        strict_load_with_audit(
            model, _precomputed_text_checkpoint(_ToyClip()), context="gemini-like"
        )


def test_absent_text_tower_is_allowed_in_image_mode():
    model = _ToyClip()
    ckpt = _precomputed_text_checkpoint(_ToyClip())
    strict_load_with_audit(
        model, ckpt, context="gemini-like", allow_unloaded_text_tower=True
    )


def test_image_weights_are_actually_loaded_when_the_text_tower_is_skipped():
    """The point of the exception is that the image tower still gets the checkpoint.

    Without this, 'it loaded' could mean 'it silently loaded nothing'.
    """
    model = _ToyClip()
    ckpt = _precomputed_text_checkpoint(_ToyClip())
    before = model.encode_image.backbone.weight.clone()

    strict_load_with_audit(
        model, ckpt, context="gemini-like", allow_unloaded_text_tower=True
    )

    assert not torch.equal(before, model.encode_image.backbone.weight)
    torch.testing.assert_close(
        model.encode_image.backbone.weight, ckpt["encode_image.backbone.weight"]
    )
    torch.testing.assert_close(model.logit_scale, ckpt["logit_scale"])


def test_text_tower_is_left_at_init_not_zeroed():
    """Unloaded means untouched — the module keeps its constructed weights."""
    model = _ToyClip()
    before = model.encode_text.text_encoder.word_embeddings.weight.clone()
    strict_load_with_audit(
        model,
        _precomputed_text_checkpoint(_ToyClip()),
        context="gemini-like",
        allow_unloaded_text_tower=True,
    )
    torch.testing.assert_close(model.encode_text.text_encoder.word_embeddings.weight, before)


def test_image_side_mismatch_still_raises_in_image_mode():
    """The exception must NOT widen. A wrong classifier width is the realistic
    class_nb error, and it has to keep failing even with the flag on."""
    model = _ToyClip(class_nb=4)
    ckpt = _precomputed_text_checkpoint(_ToyClip(class_nb=7))
    with pytest.raises(RuntimeError, match="key audit is non-empty"):
        strict_load_with_audit(
            model, ckpt, context="wrong class_nb", allow_unloaded_text_tower=True
        )


def test_missing_image_key_still_raises_in_image_mode():
    model = _ToyClip()
    ckpt = _precomputed_text_checkpoint(_ToyClip())
    del ckpt["encode_image.backbone.weight"]
    with pytest.raises(RuntimeError, match="key audit is non-empty"):
        strict_load_with_audit(
            model, ckpt, context="missing backbone", allow_unloaded_text_tower=True
        )


def test_unexpected_key_still_raises_in_image_mode():
    """An unexpected key means the checkpoint has something the model does not —
    never tolerated, text tower or not."""
    model = _ToyClip()
    ckpt = _precomputed_text_checkpoint(_ToyClip())
    ckpt["encode_image.some_new_head.weight"] = torch.randn(3, 3)
    with pytest.raises(RuntimeError, match="key audit is non-empty"):
        strict_load_with_audit(
            model, ckpt, context="extra head", allow_unloaded_text_tower=True
        )


def test_excluding_text_tower_audit_sees_only_image_side():
    model = _ToyClip()
    ckpt = _precomputed_text_checkpoint(_ToyClip())

    full = audit_state_dict(model, ckpt)
    assert full["missing"], "toy fixture should mimic the real missing text keys"
    assert full["mismatched"] == ["encode_text.linear_layer.weight"]

    image_only = audit_state_dict_excluding_text_tower(model, ckpt)
    assert image_only == {"missing": [], "unexpected": [], "mismatched": []}
