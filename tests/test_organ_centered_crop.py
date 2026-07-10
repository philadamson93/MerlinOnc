"""OrganCenteredCropd — golden crop-center parity + RAS-backend equivalence.

CPU-only, download-free (synthetic MetaTensors; no NIfTI on disk, no model).

Two purposes, both from Phase 2 of the TotalSeg organ-localization port
(vista-ct docs/plans/totalseg-organ-localization-port.md):

1. GOLDEN PARITY (OQ2). The ``SHARED FIXTURE`` block + ``test_apex_golden`` /
   ``test_center_golden`` below are duplicated VERBATIM in the sibling repo
   (contrastive-3d-onc/tests/test_organ_centered_crop.py). Same synthetic volume
   + same CSV + same ROI must yield the SAME crop start/end indices in both repos
   — the drift guard for the two hand-duplicated OrganCenteredCropd bodies. The
   ONLY intended difference between the two files is the import line just below.

2. RAS BACKEND EQUIVALENCE (Phase-1 VM Step-5 carry-forward). The producer
   orients masks to RAS with nibabel ``as_closest_canonical``; this consumer
   pipeline orients with MONAI ``Orientationd(axcodes="RAS")``. OQ15's
   "read dim_2_fraction directly, no affine map" holds only if those two RAS
   backends agree — ``test_nibabel_and_monai_ras_agree`` proves it.
"""

import numpy as np
import pytest

from merlin.data.monai_transforms import OrganCenteredCropd

from monai.data import MetaTensor


# ── SHARED FIXTURE (keep byte-identical with the sibling repo's copy) ─────────
IMAGE_FILE = "STUDY__SERIES.nii.gz"
ROI_SIZE = [4, 4, 6]
SUPERIOR_BUFFER = 2
VOL_SHAPE = (1, 10, 8, 20)  # channel-first ramp so slicing is exactly checkable
DIM_2_FRACTION = 0.5        # organ bbox center (center anchor)
DIM_2_MAX_FRACTION = 0.8    # organ superior edge (apex anchor)
# x: (10-4)//2 = 3 -> 3:7 ; y: (8-4)//2 = 2 -> 2:6.
# apex z:  round(20*0.8)=16 ; +buffer 2 - roi_z 6 = 12 (clamp [0,14]) -> 12:18.
# center z: round(20*0.5)=10 ; -roi_z//2 3 = 7 (clamp [0,14]) -> 7:13.
EXPECTED_APEX = (slice(None), slice(3, 7), slice(2, 6), slice(12, 18))
EXPECTED_CENTER = (slice(None), slice(3, 7), slice(2, 6), slice(7, 13))


def _write_csv(tmp_path, extra_rows=""):
    csv_path = tmp_path / "localization.csv"
    csv_path.write_text(
        "image_file,structure,dim_2_fraction,dim_2_max_fraction\n"
        f"{IMAGE_FILE},lungs,{DIM_2_FRACTION},{DIM_2_MAX_FRACTION}\n"
        f"{IMAGE_FILE},liver,0.30,0.40\n" + extra_rows
    )
    return str(csv_path)


def _ramp_volume():
    return np.arange(int(np.prod(VOL_SHAPE)), dtype=np.float32).reshape(VOL_SHAPE)


def _meta_tensor(arr, image_file=IMAGE_FILE):
    return MetaTensor(arr, meta={"filename_or_obj": f"/data/{image_file}"})


def test_apex_golden(tmp_path):
    """Apex anchor lands the SHARED golden crop window [.., 3:7, 2:6, 12:18]."""
    vol = _ramp_volume()
    crop = OrganCenteredCropd(
        keys=["image"], roi_size=ROI_SIZE, organ_coordinates_path=_write_csv(tmp_path),
        organ="lungs", anchor="apex", superior_buffer=SUPERIOR_BUFFER,
    )
    out = np.asarray(crop({"image": _meta_tensor(vol)})["image"])
    assert out.shape == (1, 4, 4, 6)
    np.testing.assert_array_equal(out, vol[EXPECTED_APEX])


def test_center_golden(tmp_path):
    """Center anchor lands the SHARED golden crop window [.., 3:7, 2:6, 7:13]."""
    vol = _ramp_volume()
    crop = OrganCenteredCropd(
        keys=["image"], roi_size=ROI_SIZE, organ_coordinates_path=_write_csv(tmp_path),
        organ="lungs", anchor="center", superior_buffer=SUPERIOR_BUFFER,
    )
    out = np.asarray(crop({"image": _meta_tensor(vol)})["image"])
    assert out.shape == (1, 4, 4, 6)
    np.testing.assert_array_equal(out, vol[EXPECTED_CENTER])


def test_meta_dict_fallback(tmp_path):
    """Robust accessor also reads the ``{key}_meta_dict`` convention (no MetaTensor)."""
    vol = _ramp_volume()
    data = {"image": vol, "image_meta_dict": {"filename_or_obj": f"/data/{IMAGE_FILE}"}}
    crop = OrganCenteredCropd(
        keys=["image"], roi_size=ROI_SIZE, organ_coordinates_path=_write_csv(tmp_path),
        organ="lungs", anchor="apex", superior_buffer=SUPERIOR_BUFFER,
    )
    out = np.asarray(crop(data)["image"])
    np.testing.assert_array_equal(out, vol[EXPECTED_APEX])


def test_missing_organ_row_falls_back_to_center(tmp_path):
    """A filename absent from the CSV → z-center 0.5 + warning, never raises (OQ13)."""
    vol = _ramp_volume()
    crop = OrganCenteredCropd(
        keys=["image"], roi_size=ROI_SIZE, organ_coordinates_path=_write_csv(tmp_path),
        organ="lungs", anchor="apex", superior_buffer=SUPERIOR_BUFFER,
    )
    img = _meta_tensor(vol, image_file="UNKNOWN__SERIES.nii.gz")
    with pytest.warns(UserWarning):
        out = np.asarray(crop({"image": img})["image"])
    # fallback frac 0.5, apex: round(20*0.5)+2-6 = 6 -> z 6:12.
    assert out.shape == (1, 4, 4, 6)
    np.testing.assert_array_equal(out, vol[(slice(None), slice(3, 7), slice(2, 6), slice(6, 12))])


def test_nan_fraction_falls_back_to_center(tmp_path):
    """A present-but-NaN fraction (absent-anatomy row) → 0.5 fallback + warning."""
    vol = _ramp_volume()
    csv_path = _write_csv(tmp_path, extra_rows="BLANK__SERIES.nii.gz,lungs,,\n")
    crop = OrganCenteredCropd(
        keys=["image"], roi_size=ROI_SIZE, organ_coordinates_path=csv_path,
        organ="lungs", anchor="apex", superior_buffer=SUPERIOR_BUFFER,
    )
    img = _meta_tensor(vol, image_file="BLANK__SERIES.nii.gz")
    with pytest.warns(UserWarning):
        out = np.asarray(crop({"image": img})["image"])
    np.testing.assert_array_equal(out, vol[(slice(None), slice(3, 7), slice(2, 6), slice(6, 12))])


def test_missing_required_columns_raises(tmp_path):
    """A CSV missing the frozen localization columns fails loudly at construction."""
    bad = tmp_path / "bad.csv"
    bad.write_text("image_file,structure,dim_2_fraction\nSTUDY__SERIES.nii.gz,lungs,0.5\n")
    with pytest.raises(ValueError, match="missing"):
        OrganCenteredCropd(
            keys=["image"], roi_size=ROI_SIZE, organ_coordinates_path=str(bad),
            organ="lungs", anchor="apex", superior_buffer=SUPERIOR_BUFFER,
        )


def test_invalid_anchor_raises(tmp_path):
    with pytest.raises(ValueError, match="anchor"):
        OrganCenteredCropd(
            keys=["image"], roi_size=ROI_SIZE, organ_coordinates_path=_write_csv(tmp_path),
            organ="lungs", anchor="middle", superior_buffer=SUPERIOR_BUFFER,
        )


def test_nibabel_and_monai_ras_agree():
    """Carry-forward 1: nibabel as_closest_canonical == MONAI Orientationd(RAS).

    Producer orients masks with nibabel; this consumer pipeline orients with
    MONAI. OQ15 (read dim_2_fraction directly, no affine map) holds only if the
    two RAS backends produce the SAME array (axis order + direction). Proven on a
    deliberately permuted + z-flipped affine, which a same-axis test would miss.
    """
    import torch

    nib = pytest.importorskip("nibabel")
    from monai.transforms import Orientation

    arr = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    # array axis0 -> world A, axis1 -> world R, axis2 -> world I (flipped S).
    affine = np.array(
        [
            [0.0, 2.0, 0.0, -10.0],
            [3.0, 0.0, 0.0, -20.0],
            [0.0, 0.0, -4.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    canon = nib.as_closest_canonical(nib.Nifti1Image(arr, affine))
    nib_arr = np.asarray(canon.get_fdata(), dtype=np.float32)

    monai_out = Orientation(axcodes="RAS")(MetaTensor(arr[None], affine=torch.as_tensor(affine)))
    monai_arr = np.asarray(monai_out[0], dtype=np.float32)

    assert nib.aff2axcodes(canon.affine) == ("R", "A", "S")
    assert nib.aff2axcodes(np.asarray(monai_out.affine)) == ("R", "A", "S")
    np.testing.assert_array_equal(nib_arr, monai_arr)
