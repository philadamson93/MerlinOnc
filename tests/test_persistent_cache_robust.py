"""Tests for CTPersistentDataset's cache handling when cache files are broken or cannot be written.

On 2026-10-28 a full disk cut cache writes short and left truncated ``.pt`` files under their
real names; every later read of them failed the scan. Covered here:

  * a truncated cache file is dropped and recomputed (with a warning), not raised;
  * a cache file that is a complete zip but still fails to load keeps raising, so good
    shared cache files are never deleted by mistake;
  * a failed cache write warns and still returns the computed image, leaving neither a
    file under the real name nor a staged ``.part`` file.

All synthetic — no GPU, model weights, NIfTI, or mount access needed.
"""

import zipfile

import pytest
import torch
from monai.transforms import Compose, Lambdad

from merlin.data import dataloaders
from merlin.data.dataloaders import CTPersistentDataset


def _dataset(cache_dir):
    transform = Compose([Lambdad(keys="image", func=lambda _: torch.ones(1, 2, 2, 2))])
    return CTPersistentDataset(data=[{"image": "scan.nii.gz"}], transform=transform, cache_dir=cache_dir)


def _hashfile(ds):
    return ds.cache_dir / f"{ds.hash_func({'image': 'scan.nii.gz'}).decode('utf-8')}.pt"


def test_writes_cache_and_leaves_no_part_file(tmp_path):
    ds = _dataset(tmp_path)
    out = ds._cachecheck(ds.data[0])
    assert torch.equal(out["image"], torch.ones(1, 2, 2, 2))
    assert zipfile.is_zipfile(_hashfile(ds))
    assert not list(tmp_path.glob(".*.part"))


def test_truncated_cache_file_is_recomputed(tmp_path):
    ds = _dataset(tmp_path)
    ds._cachecheck(ds.data[0])
    hashfile = _hashfile(ds)
    data = hashfile.read_bytes()
    hashfile.write_bytes(data[: len(data) // 2])
    with pytest.warns(UserWarning, match="Truncated cache file"):
        out = ds._cachecheck(ds.data[0])
    assert torch.equal(out["image"], torch.ones(1, 2, 2, 2))
    assert zipfile.is_zipfile(hashfile)
    assert torch.equal(torch.load(hashfile, weights_only=False), torch.ones(1, 2, 2, 2))


def test_complete_zip_that_fails_to_load_still_raises(tmp_path):
    ds = _dataset(tmp_path)
    hashfile = _hashfile(ds)
    with zipfile.ZipFile(hashfile, "w") as z:
        z.writestr("not_a_tensor.txt", "hello")
    with pytest.raises(Exception):
        ds._cachecheck(ds.data[0])
    assert hashfile.is_file()


def test_failed_cache_write_still_returns_image(tmp_path, monkeypatch):
    ds = _dataset(tmp_path)

    def full_disk(*args, **kwargs):
        raise RuntimeError("PytorchStreamWriter failed writing file data: file write failed")

    monkeypatch.setattr(dataloaders.torch, "save", full_disk)
    with pytest.warns(UserWarning, match="Could not cache"):
        out = ds._cachecheck(ds.data[0])
    assert torch.equal(out["image"], torch.ones(1, 2, 2, 2))
    assert not _hashfile(ds).exists()
    assert not list(tmp_path.glob(".*.part"))
