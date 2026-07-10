import csv
import math
import os
import warnings
from typing import List, Sequence, Tuple

import numpy as np
import torch
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    MapTransform,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
    CenterSpatialCropd,
)

ROI_SIZE = [224, 224, 160]


class TopCenteredSpatialCropd(MapTransform):
    """Crop by centering in x,y and taking the superior-most region in z.

    In RAS orientation the highest z-indices correspond to the superior
    (head) direction, so this preserves the lung apex for chest CTs.
    """

    def __init__(self, keys: Sequence[str], roi_size: Sequence[int]):
        super().__init__(keys)
        self.roi_size: Tuple[int, int, int] = tuple(int(v) for v in roi_size)  # type: ignore[assignment]

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            if not isinstance(img, (np.ndarray, torch.Tensor)):
                raise TypeError(f"Unsupported data type for key '{key}': {type(img)}")

            spatial_size = img.shape[-3:]
            size_x, size_y, size_z = spatial_size
            req_x, req_y, req_z = self.roi_size

            roi_x = size_x if req_x <= 0 else req_x
            roi_y = size_y if req_y <= 0 else req_y
            roi_z = size_z if req_z <= 0 else req_z

            start_x = max((size_x - roi_x) // 2, 0)
            start_y = max((size_y - roi_y) // 2, 0)
            # Superior-most along z (highest indices in RAS)
            start_z = max(size_z - roi_z, 0)

            end_x = min(start_x + roi_x, size_x)
            end_y = min(start_y + roi_y, size_y)
            end_z = min(start_z + roi_z, size_z)

            slicer = [slice(None)] * (img.ndim - 3) + [
                slice(start_x, end_x),
                slice(start_y, end_y),
                slice(start_z, end_z),
            ]
            d[key] = img[tuple(slicer)]
        return d


def _parse_fraction(value):
    """Parse a localization-CSV fraction cell to float; blank / ``nan`` -> NaN."""
    if value is None:
        return float("nan")
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


class OrganCenteredCropd(MapTransform):
    """Crop centered in x,y with the z-window anchored on an organ's location.

    z is anchored using a per-series organ-localization CSV produced by
    vista-ct's ``python -m vista_ct.segment`` (frozen column contract:
    vista-ct ``docs/organ-localization.md``). The producer computes the
    fractions on the RAS-oriented TotalSegmentator mask, so they live in the
    same frame this transform sees -- the pipeline runs
    ``Orientationd(axcodes="RAS")`` before the crop -- meaning fraction 0 ->
    inferior, 1 -> superior on the z (dim_2) axis, read directly with no affine
    mapping (OQ15).

    Anchor modes (``anchor``):
      * ``"apex"`` (default; OQ14) -- the crop's superior edge sits
        ``superior_buffer`` voxels above the organ's superior bbox edge
        (``dim_2_max_fraction``) and the window extends inferiorly for the full
        ``roi_size`` ("top of the lungs + a small buffer, go down"); robust to
        organ-size variation and scan extent.
      * ``"center"`` -- centered on the organ's bbox center
        (``dim_2_fraction``), matching Malte's original ``OrganCenteredCropd``.

    x,y are always geometric-center-cropped (as in ``TopCenteredSpatialCropd``).
    A missing image key, missing organ row, or NaN fraction falls back to a
    z-center of 0.5 with a warning and never raises (OQ13) -- matching the
    upstream never-fail contract.

    PARITY: the body of this class (and ``_parse_fraction`` above) is duplicated
    VERBATIM in
      * contrastive-3d-onc  contrastive_3d/datasets/monai_transforms.py
      * MerlinOnc            merlin/data/monai_transforms.py
    and pinned by a SHARED-fixture golden crop-center test in both repos. Keep
    the two copies byte-identical. ``roi_size`` is passed in by each repo's
    ``build_image_transform`` (contrastive: config ``spatial_size``; MerlinOnc:
    ``ROI_SIZE``). Relies on an upstream ``SpatialPadd(spatial_size=roi_size)``
    to guarantee ``size >= roi`` (no internal zero-pad, same as
    ``TopCenteredSpatialCropd``). ``superior_buffer`` is in voxels of the
    post-``Spacingd`` grid; its default is provisional pending the deferred
    OQ14 apex-vs-liver-center A/B (rendered eyeball crops).
    """

    def __init__(
        self,
        keys: Sequence[str],
        roi_size: Sequence[int],
        organ_coordinates_path: str,
        organ: str = "lungs",
        anchor: str = "apex",
        superior_buffer: int = 5,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        if anchor not in ("apex", "center"):
            raise ValueError(
                f"OrganCenteredCropd anchor must be 'apex' or 'center', got {anchor!r}"
            )
        self.roi_size: Tuple[int, int, int] = tuple(int(v) for v in roi_size)  # type: ignore[assignment]
        self.organ_coordinates_path = str(organ_coordinates_path)
        self.organ = organ
        self.anchor = anchor
        self.superior_buffer = int(superior_buffer)
        self._center_frac, self._max_frac = self._load_fractions()

    def _load_fractions(self):
        center: dict = {}
        max_edge: dict = {}
        required = {"image_file", "structure", "dim_2_fraction", "dim_2_max_fraction"}
        with open(self.organ_coordinates_path, newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"Organ-coordinates CSV {self.organ_coordinates_path} is missing "
                    f"columns {sorted(missing)}; expected the vista-ct localization "
                    f"schema (see vista-ct docs/organ-localization.md)."
                )
            for row in reader:
                if row.get("structure") != self.organ:
                    continue
                image_file = row.get("image_file")
                if not image_file:
                    continue
                center[image_file] = _parse_fraction(row.get("dim_2_fraction"))
                max_edge[image_file] = _parse_fraction(row.get("dim_2_max_fraction"))
        return center, max_edge

    @staticmethod
    def _filename_or_obj(d, key):
        """Robust meta accessor: MetaTensor ``.meta`` first, ``{key}_meta_dict`` fallback."""
        img = d.get(key)
        meta = getattr(img, "meta", None)
        if isinstance(meta, dict):
            value = meta.get("filename_or_obj")
            if value:
                return value
        meta_dict = d.get(f"{key}_meta_dict")
        if isinstance(meta_dict, dict):
            value = meta_dict.get("filename_or_obj")
            if value:
                return value
        return None

    def _resolve_z_fraction(self, d, key):
        filename = self._filename_or_obj(d, key)
        image_file = os.path.basename(str(filename)) if filename else None
        lut = self._max_frac if self.anchor == "apex" else self._center_frac
        fraction = lut.get(image_file) if image_file is not None else None
        if fraction is None or not math.isfinite(fraction):
            warnings.warn(
                f"OrganCenteredCropd: no finite {self.anchor} z-fraction for organ "
                f"'{self.organ}' / image '{image_file}' in "
                f"{self.organ_coordinates_path}; falling back to z-center 0.5.",
                stacklevel=2,
            )
            return 0.5
        return float(fraction)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            if not isinstance(img, (np.ndarray, torch.Tensor)):
                raise TypeError(f"Unsupported data type for key '{key}': {type(img)}")

            fraction = self._resolve_z_fraction(d, key)

            size_x, size_y, size_z = img.shape[-3:]
            req_x, req_y, req_z = self.roi_size
            roi_x = size_x if req_x <= 0 else req_x
            roi_y = size_y if req_y <= 0 else req_y
            roi_z = size_z if req_z <= 0 else req_z

            start_x = max((size_x - roi_x) // 2, 0)
            start_y = max((size_y - roi_y) // 2, 0)
            if self.anchor == "apex":
                # Superior edge of the organ (+ buffer above it), extend inferiorly.
                start_z = int(round(size_z * fraction)) + self.superior_buffer - roi_z
            else:
                start_z = int(round(size_z * fraction)) - roi_z // 2
            # Clamp so the full roi_z window stays in bounds. Upstream SpatialPadd
            # guarantees size_z >= roi_z, so the output is exactly roi_z (no pad here).
            start_z = max(0, min(start_z, size_z - roi_z))

            end_x = min(start_x + roi_x, size_x)
            end_y = min(start_y + roi_y, size_y)
            end_z = start_z + roi_z

            slicer = [slice(None)] * (img.ndim - 3) + [
                slice(start_x, end_x),
                slice(start_y, end_y),
                slice(start_z, end_z),
            ]
            d[key] = img[tuple(slicer)]
        return d


def build_image_transform(
    crop_mode: str = "center",
    organ_coordinates_path: str = None,
    organ: str = "lungs",
    crop_anchor: str = "apex",
    superior_buffer: int = 5,
) -> Compose:
    """Build the MONAI preprocessing pipeline with the specified crop mode.

    Args:
        crop_mode: ``"center"`` for standard CenterSpatialCropd,
                   ``"top_centered"`` for TopCenteredSpatialCropd, or
                   ``"organ_centered"`` for OrganCenteredCropd (z anchored on an
                   organ from ``organ_coordinates_path``).
        organ_coordinates_path: path to the vista-ct localization CSV (required
                   when crop_mode == "organ_centered").
        organ / crop_anchor / superior_buffer: OrganCenteredCropd parameters.
    """
    if crop_mode == "center":
        crop = CenterSpatialCropd(roi_size=ROI_SIZE, keys=["image"])
    elif crop_mode == "top_centered":
        crop = TopCenteredSpatialCropd(keys=["image"], roi_size=ROI_SIZE)
    elif crop_mode == "organ_centered":
        if not organ_coordinates_path:
            raise ValueError(
                "crop_mode='organ_centered' requires organ_coordinates_path "
                "(the vista-ct localization CSV)."
            )
        crop = OrganCenteredCropd(
            keys=["image"],
            roi_size=ROI_SIZE,
            organ_coordinates_path=organ_coordinates_path,
            organ=organ,
            anchor=crop_anchor,
            superior_buffer=superior_buffer,
        )
    else:
        raise ValueError(
            f"Unknown crop_mode '{crop_mode}'. Options: center, top_centered, organ_centered"
        )

    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear")),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
        ),
        SpatialPadd(keys=["image"], spatial_size=ROI_SIZE),
        crop,
        ToTensord(keys=["image"]),
    ])


def build_preprocess_transform() -> Compose:
    """Build a MONAI pipeline that preprocesses but skips the z-crop.

    Same steps as :func:`build_image_transform` (load → orient → resample →
    scale → pad) but center-crops only x,y to 224 and leaves z at full extent.
    Intended for multi-crop workflows where the caller extracts z-subvolumes.
    """
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear")),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
        ),
        SpatialPadd(keys=["image"], spatial_size=ROI_SIZE),
        CenterSpatialCropd(roi_size=[ROI_SIZE[0], ROI_SIZE[1], -1], keys=["image"]),
        ToTensord(keys=["image"]),
    ])


def compute_z_crop_positions(z_size: int, roi_z: int, num_crops: int) -> List[int]:
    """Return equally-spaced z-start indices for extracting subvolumes.

    Args:
        z_size: Total z-extent of the preprocessed volume.
        roi_z: Z-extent of each crop (e.g. 160).
        num_crops: Desired number of crops.

    Returns:
        List of integer z-start positions.  If the volume fits in a single
        crop (``z_size <= roi_z``), returns ``[0]`` regardless of *num_crops*.
    """
    if z_size <= roi_z:
        return [0]
    return np.linspace(0, z_size - roi_z, num_crops).astype(int).tolist()


# Default transform (backward compat)
ImageTransforms = build_image_transform("center")
