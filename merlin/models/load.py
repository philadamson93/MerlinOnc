import os

import torch
from torch import nn

from merlin.models.build import MerlinArchitecture
from merlin.utils import download_file
from typing import Dict, Any

# NOTE: ``Clip3DForTextGeneration`` (the report-generation builder) is imported
# lazily inside ``_load_model`` rather than at module top. It pulls in ``peft``,
# whose currently-resolved version is incompatible with the pinned
# ``transformers``; importing it eagerly would break the far more common
# image-embedding / phenotype paths (and every downstream consumer, incl.
# vista-ct) even though they never touch report generation.

DEFAULT_REPO_ID = "stanfordmimi/Merlin"


def audit_state_dict(model: nn.Module, state_dict: Dict[str, Any]) -> Dict[str, list]:
    """Return the ``{missing, unexpected, mismatched}`` key sets for a strict load.

    * ``missing``    -- in the model but absent from the checkpoint
    * ``unexpected`` -- in the checkpoint but absent from the model
    * ``mismatched`` -- present in both but with a different tensor shape

    A clean load has all three empty. Exposed (not underscore-private) so the
    strict-preflight test can assert emptiness against the real checkpoint (RD5).
    """
    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())
    ckpt_keys = set(state_dict.keys())
    mismatched = sorted(
        k
        for k in (model_keys & ckpt_keys)
        if tuple(getattr(model_sd[k], "shape", ())) != tuple(getattr(state_dict[k], "shape", ()))
    )
    return {
        "missing": sorted(model_keys - ckpt_keys),
        "unexpected": sorted(ckpt_keys - model_keys),
        "mismatched": mismatched,
    }


TEXT_TOWER_PREFIX = "encode_text."


def audit_state_dict_excluding_text_tower(
    model: nn.Module, state_dict: Dict[str, Any]
) -> Dict[str, list]:
    """``audit_state_dict`` over everything except the ``encode_text.*`` tower.

    Used only in image-embedding mode, where the text tower never runs (see
    ``MerlinArchitecture.forward``: ``ImageEmbedding`` returns ``encode_image(image)``
    and raises if given text). Strictness over every tensor that DOES participate in
    the forward pass is unchanged.
    """
    keep = lambda k: not k.startswith(TEXT_TOWER_PREFIX)  # noqa: E731
    model_sd = {k: v for k, v in model.state_dict().items() if keep(k)}
    ckpt = {k: v for k, v in state_dict.items() if keep(k)}
    model_keys, ckpt_keys = set(model_sd), set(ckpt)
    mismatched = sorted(
        k
        for k in (model_keys & ckpt_keys)
        if tuple(getattr(model_sd[k], "shape", ()))
        != tuple(getattr(ckpt[k], "shape", ()))
    )
    return {
        "missing": sorted(model_keys - ckpt_keys),
        "unexpected": sorted(ckpt_keys - model_keys),
        "mismatched": mismatched,
    }


def _discrepancies_confined_to_text_tower(audit: Dict[str, list]) -> bool:
    """Every missing/mismatched key is a text-tower key, and nothing is unexpected."""
    flagged = audit["missing"] + audit["mismatched"]
    return (
        bool(flagged)
        and not audit["unexpected"]
        and all(k.startswith(TEXT_TOWER_PREFIX) for k in flagged)
    )


def strict_load_with_audit(
    model: nn.Module,
    state_dict: Dict[str, Any],
    context: str,
    allow_unloaded_text_tower: bool = False,
) -> None:
    """Fail-closed strict load: audit key sets first, stop on any discrepancy,
    then ``load_state_dict(strict=True)``.

    ``strict=True`` alone stops at the first problem class; this reports all three
    sets up front so a checkpoint/``class_nb`` mismatch is diagnosable in one shot.
    We do not fall back to a lenient ``strict=False`` + size-filter -- that can
    silently drop backbone tensors. Fix the config instead.

    ``allow_unloaded_text_tower`` (set only in image-embedding mode) permits ONE
    narrow exception: a checkpoint whose discrepancies are confined entirely to
    ``encode_text.*``. A CLIP model trained against precomputed text vectors has no
    Clinical-Longformer body to ship, so those keys are legitimately absent -- and in
    image-embedding mode the text tower never runs, so leaving it at its init cannot
    change a single output value. Everything that DOES run is still loaded strictly:
    an unexpected key, or any image-side discrepancy, still raises.
    """
    audit = audit_state_dict(model, state_dict)
    if any(audit.values()):
        def _fmt(keys):
            if not keys:
                return "(none)"
            head = ", ".join(keys[:10])
            return head + (f" (+{len(keys) - 10} more)" if len(keys) > 10 else "")

        if allow_unloaded_text_tower and _discrepancies_confined_to_text_tower(audit):
            image_audit = audit_state_dict_excluding_text_tower(model, state_dict)
            if not any(image_audit.values()):
                loadable = {
                    k: v
                    for k, v in state_dict.items()
                    if not k.startswith(TEXT_TOWER_PREFIX)
                }
                # strict=False only because the text tower is deliberately absent;
                # the audit above already proved every loaded key matches exactly.
                model.load_state_dict(loadable, strict=False)
                print(
                    f"Loaded {context} in image-embedding mode WITHOUT its text tower: "
                    f"{len(audit['missing'])} missing + {len(audit['mismatched'])} "
                    f"mismatched keys, all under '{TEXT_TOWER_PREFIX}', left at init. "
                    "The text tower does not run in this mode. Every image-side tensor "
                    "was loaded and audited strictly."
                )
                return
            audit = image_audit
            context += " (image-side audit, text tower excluded)"

        raise RuntimeError(
            f"Refusing to load {context}: state-dict key audit is non-empty.\n"
            f"  missing ({len(audit['missing'])}): {_fmt(audit['missing'])}\n"
            f"  unexpected ({len(audit['unexpected'])}): {_fmt(audit['unexpected'])}\n"
            f"  mismatched ({len(audit['mismatched'])}): {_fmt(audit['mismatched'])}\n"
            "This usually means the checkpoint was trained with a different "
            "architecture / class_nb than requested. Fix the config rather than "
            "loosening strictness."
        )
    model.load_state_dict(state_dict, strict=True)
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "default": {
        "builder": MerlinArchitecture,
        "checkpoint": "i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt",
    },
    "report_generation": {
        # Builder imported lazily in _load_model (see module-top note re: peft).
        "builder": None,
        "checkpoint": "resnet_gpt2_best_stanford_report_generation_average.pt",
    },
    "five_year_disease_prediction": {
        "builder": MerlinArchitecture,
        "checkpoint": "resnet_clinical_longformer_five_year_disease_prediction.pt",
    },
    "merlin_onc": {
        "builder": MerlinArchitecture,
        "checkpoint": "i3_resnet_clinical_longformer_best_clip_10-08-2025_03-41-48_epoch_99.pt",
        "repo_id": "philadamson93/MerlinOnc",
    },
}


class Merlin(nn.Module):
    def __init__(
        self,
        ImageEmbedding: bool = False,
        PhenotypeCls: bool = False,
        RadiologyReport: bool = False,
        FiveYearPred: bool = False,
        MerlinOnc: bool = False,
        local_checkpoint_path: str = None,
        class_nb: int = 1692,
    ):
        super(Merlin, self).__init__()

        # If more than one output mode is True, raise an error
        if sum([ImageEmbedding, PhenotypeCls, FiveYearPred]) > 1:
            raise ValueError(
                "ImageEmbedding and PhenotypeCls and FiveYearPred cannot be True at the same time."
            )

        # Determine task based on flags
        if MerlinOnc:
            self.task = "merlin_onc"
        elif RadiologyReport:
            self.task = "report_generation"
        elif FiveYearPred:
            self.task = "five_year_disease_prediction"
        else:
            self.task = "default"

        self.local_checkpoint_path = local_checkpoint_path

        self._config = MODEL_CONFIGS[self.task]

        # Pass through the flags needed by the underlying model builders
        model_kwargs = (
            {
                "ImageEmbedding": ImageEmbedding,
                "PhenotypeCls": PhenotypeCls,
                "FiveYearPred": FiveYearPred,
                "class_nb": class_nb,
            }
            if not RadiologyReport
            else {}
        )
        self.model = self._load_model(**model_kwargs)

    def _load_model(self, **kwargs) -> nn.Module:
        """
        Downloads the correct checkpoint and constructs the appropriate model.
        If local_checkpoint_path is provided, uses that instead of downloading.
        """
        model_builder = self._config["builder"]
        if model_builder is None and self.task == "report_generation":
            # Lazy: only the report-generation path needs peft (see module top).
            from merlin.models.radiology_report_generation import (
                Clip3DForTextGeneration,
            )

            model_builder = Clip3DForTextGeneration

        # Determine checkpoint path
        if self.local_checkpoint_path is not None:
            # Use user-provided local path
            checkpoint_path = self.local_checkpoint_path
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        else:
            # Download checkpoint to local directory
            checkpoint_name = self._config["checkpoint"]
            repo_id = self._config.get("repo_id", DEFAULT_REPO_ID)
            local_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "checkpoints"
            )
            checkpoint_path = os.path.join(local_dir, checkpoint_name)
            self._download_checkpoint(
                filename=checkpoint_name, local_dir=local_dir, repo_id=repo_id
            )

        # Build the model
        model = model_builder(**kwargs)

        print(f"Loading checkpoint for '{self.task}' task from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        if self.task == "five_year_disease_prediction":
            model.encode_image.i3_resnet.load_state_dict(state_dict, strict=True)
        else:
            strict_load_with_audit(
                model,
                state_dict,
                context=f"'{self.task}' checkpoint {checkpoint_path}",
                allow_unloaded_text_tower=bool(kwargs.get("ImageEmbedding")),
            )

        return model

    def _download_checkpoint(self, filename: str, local_dir: str, repo_id: str):
        if not os.path.exists(os.path.join(local_dir, filename)):
            print(f"Downloading {filename} from {repo_id}...")
            download_file(repo_id=repo_id, filename=filename, local_dir=local_dir)

    def forward(self, *args, **kwargs):
        """Delegates the forward call to the underlying model."""
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """
        Generates text if the model is in RadiologyReport mode.
        Passes all arguments to the underlying model's generate method.
        """
        if self.task != "report_generation":
            raise AttributeError(
                "The 'generate' method is only available when RadiologyReport=True."
            )
        # Delegate the call to the actual text generation model
        return self.model.generate(*args, **kwargs)
