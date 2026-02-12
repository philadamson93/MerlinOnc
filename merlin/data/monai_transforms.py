from typing import Sequence, Tuple

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


def build_image_transform(crop_mode: str = "center") -> Compose:
    """Build the MONAI preprocessing pipeline with the specified crop mode.

    Args:
        crop_mode: ``"center"`` for standard CenterSpatialCropd,
                   ``"top_centered"`` for TopCenteredSpatialCropd.
    """
    if crop_mode == "center":
        crop = CenterSpatialCropd(roi_size=ROI_SIZE, keys=["image"])
    elif crop_mode == "top_centered":
        crop = TopCenteredSpatialCropd(keys=["image"], roi_size=ROI_SIZE)
    else:
        raise ValueError(f"Unknown crop_mode '{crop_mode}'. Options: center, top_centered")

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


# Default transform (backward compat)
ImageTransforms = build_image_transform("center")
