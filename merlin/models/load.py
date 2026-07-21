import os

import torch
from torch import nn

from merlin.models.build import MerlinArchitecture
from merlin.models.radiology_report_generation import Clip3DForTextGeneration
from merlin.utils import download_file
from typing import Dict, Any, Optional

DEFAULT_REPO_ID = "stanfordmimi/Merlin"
# class_nb is per-entry here (not a single global default) because MerlinOnc
# checkpoints trained against a wider label set (v2.1/v2.2) need a different
# classifier-head width than the original 1692-class checkpoint. A caller that
# doesn't pass class_nb explicitly gets this entry's value (see Merlin.__init__).
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "default": {
        "builder": MerlinArchitecture,
        "checkpoint": "i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt",
    },
    "report_generation": {
        "builder": Clip3DForTextGeneration,
        "checkpoint": "resnet_gpt2_best_stanford_report_generation_average.pt",
    },
    "five_year_disease_prediction": {
        "builder": MerlinArchitecture,
        "checkpoint": "resnet_clinical_longformer_five_year_disease_prediction.pt",
    },
    # "merlin_onc" is the latest-checkpoint alias (currently v2.2). To pin an
    # exact version instead, pass checkpoint_key="merlin_onc_v2_1" /
    # "merlin_onc_v2_2" to Merlin(...).
    "merlin_onc": {
        "builder": MerlinArchitecture,
        "checkpoint": "i3_resnet_clinical_longformer_best_clip_07-01-2026_18-38-12_epoch_66.pt",
        "repo_id": "philadamson93/MerlinOnc",
        "class_nb": 1876,
    },
    "merlin_onc_v2_1": {
        "builder": MerlinArchitecture,
        "checkpoint": "i3_resnet_clinical_longformer_best_clip_04-03-2026_04-45-12_epoch_99.pt",
        "repo_id": "philadamson93/MerlinOnc",
        "class_nb": 1876,
    },
    "merlin_onc_v2_2": {
        "builder": MerlinArchitecture,
        "checkpoint": "i3_resnet_clinical_longformer_best_clip_07-01-2026_18-38-12_epoch_66.pt",
        "repo_id": "philadamson93/MerlinOnc",
        "class_nb": 1876,
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
        checkpoint_key: Optional[str] = None,
        class_nb: Optional[int] = None,
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

        # checkpoint_key pins an exact MODEL_CONFIGS entry (e.g. "merlin_onc_v2_1"),
        # overriding the flag-derived task -- lets a caller select a specific
        # MerlinOnc version instead of always getting the "merlin_onc" latest alias.
        if checkpoint_key is not None:
            self.task = checkpoint_key

        self.local_checkpoint_path = local_checkpoint_path

        self._config = MODEL_CONFIGS[self.task]

        # class_nb defaults to the checkpoint's own entry (per-entry, since
        # different MerlinOnc checkpoints were trained against different label
        # counts); an explicit caller-supplied value still wins.
        resolved_class_nb = (
            class_nb if class_nb is not None else self._config.get("class_nb", 1692)
        )

        # Pass through the flags needed by the underlying model builders
        model_kwargs = (
            {
                "ImageEmbedding": ImageEmbedding,
                "PhenotypeCls": PhenotypeCls,
                "FiveYearPred": FiveYearPred,
                "class_nb": resolved_class_nb,
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
            model.load_state_dict(state_dict)

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
