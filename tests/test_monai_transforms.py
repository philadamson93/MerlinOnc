"""Tests for TopCenteredSpatialCropd, build_image_transform, and multi-crop utilities.

All tests use synthetic numpy/torch tensors — no GPU, model weights, or NIfTI needed.
"""

import numpy as np
import pytest
import torch
from monai.transforms import CenterSpatialCropd

from merlin.data.monai_transforms import (
    ROI_SIZE,
    TopCenteredSpatialCropd,
    build_image_transform,
    build_preprocess_transform,
    compute_z_crop_positions,
    ImageTransforms,
)


# ---------------------------------------------------------------------------
# TopCenteredSpatialCropd
# ---------------------------------------------------------------------------


class TestTopCenteredSpatialCropd:

    def test_centers_xy_takes_superior_z(self):
        """Core behavior: center in x,y; take highest z indices (superior in RAS)."""
        vol = np.zeros((1, 300, 300, 200), dtype=np.float32)
        for z in range(200):
            vol[0, :, :, z] = z

        crop = TopCenteredSpatialCropd(keys=["image"], roi_size=[224, 224, 160])
        result = crop({"image": vol})["image"]

        assert result.shape == (1, 224, 224, 160)
        # z should start at 200 - 160 = 40
        assert result[0, 0, 0, 0] == 40.0
        assert result[0, 0, 0, -1] == 199.0

    def test_xy_centering_precise(self):
        """Verify exact x,y center and z-superior slicing coordinates."""
        vol = np.arange(100 * 80 * 50, dtype=np.float32).reshape(1, 100, 80, 50)
        crop = TopCenteredSpatialCropd(keys=["image"], roi_size=[60, 40, 30])
        result = crop({"image": vol})["image"]

        # x: (100-60)//2=20 -> 20:80, y: (80-40)//2=20 -> 20:60, z: 50-30=20 -> 20:50
        expected = vol[:, 20:80, 20:60, 20:50]
        np.testing.assert_array_equal(result, expected)

    def test_z_preserves_superior_region(self):
        """Superior half (high z) should be fully preserved in the crop."""
        vol = np.zeros((1, 224, 224, 320), dtype=np.float32)
        vol[0, :, :, 160:] = 1.0  # superior half

        crop = TopCenteredSpatialCropd(keys=["image"], roi_size=[224, 224, 160])
        result = crop({"image": vol})["image"]

        # z start = 320-160 = 160, so entire crop is from the superior half
        assert np.all(result == 1.0)

    def test_volume_exactly_roi_size(self):
        """No-op when volume matches ROI exactly."""
        vol = np.random.randn(1, 224, 224, 160).astype(np.float32)
        crop = TopCenteredSpatialCropd(keys=["image"], roi_size=[224, 224, 160])
        result = crop({"image": vol})["image"]

        assert result.shape == (1, 224, 224, 160)
        np.testing.assert_array_equal(result, vol)

    def test_volume_smaller_than_roi(self):
        """No-op when volume is smaller than ROI in all dims."""
        vol = np.random.randn(1, 50, 50, 40).astype(np.float32)
        crop = TopCenteredSpatialCropd(keys=["image"], roi_size=[224, 224, 160])
        result = crop({"image": vol})["image"]

        assert result.shape == (1, 50, 50, 40)
        np.testing.assert_array_equal(result, vol)

    def test_only_z_exceeds_roi(self):
        """When only z exceeds ROI, x,y untouched, z cropped from top."""
        vol = np.zeros((1, 100, 100, 300), dtype=np.float32)
        for z in range(300):
            vol[0, :, :, z] = z

        crop = TopCenteredSpatialCropd(keys=["image"], roi_size=[100, 100, 160])
        result = crop({"image": vol})["image"]

        assert result.shape == (1, 100, 100, 160)
        assert result[0, 0, 0, 0] == 140.0
        assert result[0, 0, 0, -1] == 299.0

    def test_torch_tensor_input(self):
        """Works with torch tensors, returns torch tensor."""
        vol = torch.randn(1, 256, 256, 180)
        crop = TopCenteredSpatialCropd(keys=["image"], roi_size=[224, 224, 160])
        result = crop({"image": vol})["image"]

        assert result.shape == (1, 224, 224, 160)
        assert isinstance(result, torch.Tensor)

    def test_extra_leading_dims(self):
        """Handles (B, C, X, Y, Z) — extra leading dims preserved."""
        vol = np.random.randn(2, 1, 300, 300, 200).astype(np.float32)
        crop = TopCenteredSpatialCropd(keys=["image"], roi_size=[224, 224, 160])
        result = crop({"image": vol})["image"]

        assert result.shape == (2, 1, 224, 224, 160)

    def test_invalid_type_raises(self):
        """Non-array input raises TypeError."""
        crop = TopCenteredSpatialCropd(keys=["image"], roi_size=[224, 224, 160])
        with pytest.raises(TypeError, match="Unsupported data type"):
            crop({"image": "not_a_tensor"})

    def test_multiple_keys(self):
        """Operates on all specified keys."""
        vol_a = np.random.randn(1, 300, 300, 200).astype(np.float32)
        vol_b = np.random.randn(1, 300, 300, 200).astype(np.float32)
        crop = TopCenteredSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 160])
        result = crop({"image": vol_a, "label": vol_b})

        assert result["image"].shape == (1, 224, 224, 160)
        assert result["label"].shape == (1, 224, 224, 160)


# ---------------------------------------------------------------------------
# build_image_transform factory
# ---------------------------------------------------------------------------


class TestBuildImageTransform:

    def test_center_mode_uses_center_crop(self):
        transform = build_image_transform("center")
        crops = [t for t in transform.transforms if isinstance(t, CenterSpatialCropd)]
        assert len(crops) == 1

    def test_top_centered_mode_uses_top_centered_crop(self):
        transform = build_image_transform("top_centered")
        crops = [t for t in transform.transforms if isinstance(t, TopCenteredSpatialCropd)]
        assert len(crops) == 1

    def test_center_mode_excludes_top_centered(self):
        transform = build_image_transform("center")
        assert not any(isinstance(t, TopCenteredSpatialCropd) for t in transform.transforms)

    def test_top_centered_mode_excludes_center(self):
        transform = build_image_transform("top_centered")
        assert not any(isinstance(t, CenterSpatialCropd) for t in transform.transforms)

    def test_both_modes_same_pipeline_length(self):
        center = build_image_transform("center")
        top = build_image_transform("top_centered")
        assert len(center.transforms) == len(top.transforms)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown crop_mode"):
            build_image_transform("bogus")

    def test_image_transforms_constant_is_center(self):
        """Module-level ImageTransforms uses center crop (backward compat)."""
        crops = [t for t in ImageTransforms.transforms if isinstance(t, CenterSpatialCropd)]
        assert len(crops) == 1

    def test_roi_size_constant(self):
        assert ROI_SIZE == [224, 224, 160]


# ---------------------------------------------------------------------------
# build_preprocess_transform factory (multi-crop support)
# ---------------------------------------------------------------------------


class TestBuildPreprocessTransform:

    def test_uses_center_spatial_crop(self):
        """Pipeline includes a CenterSpatialCropd (for x,y only)."""
        preprocess = build_preprocess_transform()
        crops = [t for t in preprocess.transforms if isinstance(t, CenterSpatialCropd)]
        assert len(crops) == 1

    def test_no_top_centered_crop(self):
        """Pipeline should not include TopCenteredSpatialCropd."""
        preprocess = build_preprocess_transform()
        assert not any(isinstance(t, TopCenteredSpatialCropd) for t in preprocess.transforms)

    def test_crop_roi_leaves_z_unconstrained(self):
        """x,y are cropped to 224 but z is preserved at full extent."""
        # Synthetic volume: x,y larger than 224, z=200 (larger than ROI_SIZE[2]=160)
        vol = np.random.randn(1, 300, 300, 200).astype(np.float32)
        crop = CenterSpatialCropd(roi_size=[ROI_SIZE[0], ROI_SIZE[1], -1], keys=["image"])
        result = crop({"image": vol})["image"]

        assert result.shape[1] == ROI_SIZE[0]  # x cropped to 224
        assert result.shape[2] == ROI_SIZE[1]  # y cropped to 224
        assert result.shape[3] == 200          # z preserved


# ---------------------------------------------------------------------------
# compute_z_crop_positions
# ---------------------------------------------------------------------------


class TestComputeZCropPositions:

    def test_single_crop(self):
        """N=1 returns a single position at 0."""
        positions = compute_z_crop_positions(z_size=200, roi_z=160, num_crops=1)
        assert positions == [0]

    def test_two_crops_endpoints(self):
        """N=2 returns inferior-most and superior-most positions."""
        positions = compute_z_crop_positions(z_size=200, roi_z=160, num_crops=2)
        assert positions == [0, 40]

    def test_three_crops_evenly_spaced(self):
        """N=3 gives evenly spaced positions."""
        positions = compute_z_crop_positions(z_size=220, roi_z=160, num_crops=3)
        # linspace(0, 60, 3) = [0, 30, 60]
        assert positions == [0, 30, 60]

    def test_small_volume_fallback(self):
        """If z_size <= roi_z, always return [0] regardless of num_crops."""
        positions = compute_z_crop_positions(z_size=100, roi_z=160, num_crops=5)
        assert positions == [0]

    def test_exact_fit_returns_zero(self):
        """If z_size == roi_z, return [0]."""
        positions = compute_z_crop_positions(z_size=160, roi_z=160, num_crops=3)
        assert positions == [0]

    def test_barely_exceeds_roi(self):
        """z_size = roi_z + 1: N=2 gives [0, 1]."""
        positions = compute_z_crop_positions(z_size=161, roi_z=160, num_crops=2)
        assert positions == [0, 1]

    def test_returns_list_of_ints(self):
        """Return type is a plain list of Python ints (not numpy)."""
        positions = compute_z_crop_positions(z_size=300, roi_z=160, num_crops=4)
        assert isinstance(positions, list)
        for p in positions:
            assert isinstance(p, int)
