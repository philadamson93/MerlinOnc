"""Tests for configurable pixdim / spatial_size and the fail-closed strict load.

Covers the three MerlinOnc changes that let a checkpoint trained at a non-default
preprocessing geometry be served without disturbing the warmed default caches:

  * ``build_image_transform`` / ``build_preprocess_transform`` thread the resolved
    ``pixdim`` / ``spatial_size`` into Spacingd + the pad/crop, keyword-only so the
    historical positional callers stay valid.
  * ``dataloaders`` folds the full geometry into the persistent-cache dir identity
    (reusing contrastive-3d-onc's canonical suffix spelling), so distinct
    resolutions never collide and the same geometry shares one cache across repos.
  * ``models.load`` audits the state-dict key sets and refuses a lenient fallback.

All synthetic — no GPU, model weights, NIfTI, or mount access needed.
"""

import numpy as np
import pytest
import torch
from torch import nn
from monai.transforms import CenterSpatialCropd, Spacingd, SpatialPadd

from merlin.data.monai_transforms import (
    ROI_SIZE,
    OrganCenteredCropd,
    TopCenteredSpatialCropd,
    build_image_transform,
    build_preprocess_transform,
)
from merlin.data import dataloaders
from merlin.data.dataloaders import (
    DEFAULT_PIXDIM,
    DEFAULT_SPATIAL_SIZE,
    DataLoader,
    _fmt_pixdim_component,
    _geometry_cache_suffix,
)
from merlin.models.load import audit_state_dict, strict_load_with_audit


def _spacing(transform):
    return next(t for t in transform.transforms if isinstance(t, Spacingd))


def _pad(transform):
    return next(t for t in transform.transforms if isinstance(t, SpatialPadd))


# ---------------------------------------------------------------------------
# build_image_transform: geometry threading + positional-compat
# ---------------------------------------------------------------------------


class TestBuildImageTransformGeometry:

    def test_default_pixdim_and_spatial_size(self):
        """No geometry args -> Merlin's historical (1.5,1.5,3.0) / ROI_SIZE."""
        transform = build_image_transform("center")
        spacing = _spacing(transform)
        pad = _pad(transform)
        assert tuple(spacing.spacing_transform.pixdim) == (1.5, 1.5, 3.0)
        assert list(pad.padder.spatial_size) == ROI_SIZE

    def test_custom_pixdim_threaded_into_spacing(self):
        transform = build_image_transform("top_centered", pixdim=(1.0, 1.0, 1.5))
        spacing = _spacing(transform)
        assert tuple(spacing.spacing_transform.pixdim) == (1.0, 1.0, 1.5)

    def test_custom_spatial_size_threaded_into_pad_and_crop(self):
        transform = build_image_transform(
            "top_centered", spatial_size=[336, 336, 320]
        )
        pad = _pad(transform)
        crop = next(
            t for t in transform.transforms if isinstance(t, TopCenteredSpatialCropd)
        )
        assert list(pad.padder.spatial_size) == [336, 336, 320]
        assert list(crop.roi_size) == [336, 336, 320]

    def test_custom_spatial_size_center_crop(self):
        transform = build_image_transform("center", spatial_size=[336, 336, 320])
        crop = next(
            t for t in transform.transforms if isinstance(t, CenterSpatialCropd)
        )
        assert list(crop.cropper.roi_size) == [336, 336, 320]

    def test_int_spatial_size_components_coerced(self):
        """Numpy ints / floats in spatial_size don't break the crop/pad."""
        transform = build_image_transform(
            "top_centered", spatial_size=[np.int64(336), np.int64(336), np.int64(320)]
        )
        crop = next(
            t for t in transform.transforms if isinstance(t, TopCenteredSpatialCropd)
        )
        assert list(crop.roi_size) == [336, 336, 320]

    def test_positional_center_still_valid(self):
        """crop_mode stays first-positional (backward compat)."""
        build_image_transform("center")

    def test_positional_organ_centered_signature_unchanged(self, tmp_path):
        """The full historical positional organ_centered call still works."""
        csv = tmp_path / "loc.csv"
        csv.write_text(
            "image_file,structure,dim_2_fraction,dim_2_max_fraction\n"
            "a.nii.gz,lungs,0.5,0.8\n"
        )
        transform = build_image_transform(
            "organ_centered", str(csv), "lungs", "apex", 5
        )
        crop = next(
            t for t in transform.transforms if isinstance(t, OrganCenteredCropd)
        )
        assert list(crop.roi_size) == ROI_SIZE

    def test_organ_centered_honors_custom_geometry(self, tmp_path):
        csv = tmp_path / "loc.csv"
        csv.write_text(
            "image_file,structure,dim_2_fraction,dim_2_max_fraction\n"
            "a.nii.gz,lungs,0.5,0.8\n"
        )
        transform = build_image_transform(
            "organ_centered", str(csv), spatial_size=[288, 288, 256]
        )
        crop = next(
            t for t in transform.transforms if isinstance(t, OrganCenteredCropd)
        )
        assert list(crop.roi_size) == [288, 288, 256]


class TestBuildPreprocessTransformGeometry:

    def test_default_geometry(self):
        preprocess = build_preprocess_transform()
        spacing = _spacing(preprocess)
        assert tuple(spacing.spacing_transform.pixdim) == (1.5, 1.5, 3.0)

    def test_custom_geometry_threaded(self):
        preprocess = build_preprocess_transform(
            pixdim=(1.0, 1.0, 1.5), spatial_size=[336, 336, 320]
        )
        spacing = _spacing(preprocess)
        crop = next(
            t for t in preprocess.transforms if isinstance(t, CenterSpatialCropd)
        )
        assert tuple(spacing.spacing_transform.pixdim) == (1.0, 1.0, 1.5)
        # x,y cropped to spatial_size; z left unconstrained (-1)
        assert list(crop.cropper.roi_size) == [336, 336, -1]


# ---------------------------------------------------------------------------
# Cache-identity suffix (collision avoidance across geometry / crop / organ)
# ---------------------------------------------------------------------------


class TestGeometryCacheSuffix:

    def test_fmt_component_normalizes_like_c3d(self):
        # 1.0 and 1 collapse to "1"; 1.5 -> "1p5"; 3.0 -> "3".
        assert _fmt_pixdim_component(1.0) == "1"
        assert _fmt_pixdim_component(1) == "1"
        assert _fmt_pixdim_component(1.5) == "1p5"
        assert _fmt_pixdim_component(3.0) == "3"

    def test_driver_geometry_matches_c3d_warmed_dir(self):
        """The 1x1x1.5 / 336x336x320 / top_centered driver reuses the c3d dir name."""
        suffix = _geometry_cache_suffix(
            pixdim=(1.0, 1.0, 1.5),
            spatial_size=[336, 336, 320],
            crop_mode="top_centered",
            organ="lungs",
            crop_anchor="apex",
            superior_buffer=5,
            organ_coordinates_path=None,
        )
        assert suffix == "pix_1_1_1p5__size_336_336_320__crop_top_centered"

    def test_distinct_pixdim_distinct_suffix(self):
        a = _geometry_cache_suffix(
            (1.0, 1.0, 1.5), [336, 336, 320], "top_centered",
            "lungs", "apex", 5, None,
        )
        b = _geometry_cache_suffix(
            (1.5, 1.5, 3.0), [336, 336, 320], "top_centered",
            "lungs", "apex", 5, None,
        )
        assert a != b

    def test_distinct_crop_mode_distinct_suffix(self):
        a = _geometry_cache_suffix(
            (1.0, 1.0, 1.5), [336, 336, 320], "top_centered",
            "lungs", "apex", 5, None,
        )
        b = _geometry_cache_suffix(
            (1.0, 1.0, 1.5), [336, 336, 320], "center",
            "lungs", "apex", 5, None,
        )
        assert a != b

    def test_organ_centered_extends_suffix(self):
        suffix = _geometry_cache_suffix(
            (1.0, 1.0, 1.5), [336, 336, 320], "organ_centered",
            "lungs", "apex", 5, None,
        )
        assert suffix.startswith("pix_1_1_1p5__size_336_336_320__crop_organ_centered")
        assert "__organ_lungs__anchor_apex__buf_5__csv_" in suffix


# ---------------------------------------------------------------------------
# DataLoader cache-dir resolution (branch selection)
# ---------------------------------------------------------------------------


class TestDataLoaderCacheDir:
    """Instantiating with an empty datalist exercises the branch logic without I/O."""

    def _make(self, tmp_path, **kwargs):
        return DataLoader(
            datalist=[],
            cache_dir=str(tmp_path),
            batchsize=1,
            shuffle=False,
            num_workers=0,
            **kwargs,
        )

    def test_default_center_uses_base_cache_dir(self, tmp_path):
        loader = self._make(tmp_path, crop_mode="center")
        assert loader.cache_dir == str(tmp_path)

    def test_default_top_centered_uses_base_cache_dir(self, tmp_path):
        """Default-geometry top_centered keeps the base dir (preserves 299k cache)."""
        loader = self._make(tmp_path, crop_mode="top_centered")
        assert loader.cache_dir == str(tmp_path)

    def test_default_organ_centered_keeps_preresolution_suffix(self, tmp_path):
        csv = tmp_path / "loc.csv"
        csv.write_text(
            "image_file,structure,dim_2_fraction,dim_2_max_fraction\n"
            "a.nii.gz,lungs,0.5,0.8\n"
        )
        loader = self._make(
            tmp_path, crop_mode="organ_centered", organ_coordinates_path=str(csv)
        )
        # Pre-resolution organ suffix (organ/anchor/buffer/csv-hash), no geometry
        # prefix -> the warmed default-geometry apex caches still hit.
        assert "organ_centered__organ_lungs__anchor_apex__buf_5__csv_" in loader.cache_dir
        assert "pix_" not in loader.cache_dir

    def test_nondefault_geometry_namespaces_cache(self, tmp_path):
        loader = self._make(
            tmp_path,
            crop_mode="top_centered",
            pixdim=(1.0, 1.0, 1.5),
            spatial_size=[336, 336, 320],
        )
        assert loader.cache_dir.endswith(
            "pix_1_1_1p5__size_336_336_320__crop_top_centered"
        )

    def test_default_constants(self):
        assert DEFAULT_PIXDIM == (1.5, 1.5, 3.0)
        assert DEFAULT_SPATIAL_SIZE == (224, 224, 160)


# ---------------------------------------------------------------------------
# Fail-closed strict load
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    def __init__(self, out=4):
        super().__init__()
        self.fc = nn.Linear(3, out)


class TestStrictLoadAudit:

    def test_clean_audit_all_empty(self):
        model = _TinyModel()
        audit = audit_state_dict(model, model.state_dict())
        assert audit == {"missing": [], "unexpected": [], "mismatched": []}

    def test_missing_key_detected(self):
        model = _TinyModel()
        sd = dict(model.state_dict())
        del sd["fc.bias"]
        audit = audit_state_dict(model, sd)
        assert audit["missing"] == ["fc.bias"]
        assert audit["unexpected"] == []

    def test_unexpected_key_detected(self):
        model = _TinyModel()
        sd = dict(model.state_dict())
        sd["extra.weight"] = torch.zeros(2, 2)
        audit = audit_state_dict(model, sd)
        assert audit["unexpected"] == ["extra.weight"]

    def test_shape_mismatch_detected(self):
        model = _TinyModel(out=4)
        other = _TinyModel(out=8)
        audit = audit_state_dict(model, other.state_dict())
        # fc.weight (4x3 vs 8x3) and fc.bias (4 vs 8) both mismatch.
        assert set(audit["mismatched"]) == {"fc.weight", "fc.bias"}

    def test_strict_load_succeeds_on_clean(self):
        model = _TinyModel()
        strict_load_with_audit(model, dict(model.state_dict()), context="unit")

    def test_strict_load_raises_on_mismatch(self):
        model = _TinyModel(out=4)
        other = _TinyModel(out=8)
        with pytest.raises(RuntimeError, match="key audit is non-empty"):
            strict_load_with_audit(model, other.state_dict(), context="unit")

    def test_strict_load_error_reports_all_sets(self):
        model = _TinyModel()
        sd = dict(model.state_dict())
        del sd["fc.bias"]
        sd["ghost.weight"] = torch.zeros(1)
        with pytest.raises(RuntimeError) as exc:
            strict_load_with_audit(model, sd, context="unit")
        msg = str(exc.value)
        assert "missing (1)" in msg
        assert "unexpected (1)" in msg
