import torch
import monai
from copy import deepcopy
import shutil
import tempfile
from pathlib import Path
from typing import List
from monai.utils import look_up_option
from monai.data.utils import SUPPORTED_PICKLE_MOD

from merlin.data.monai_transforms import ImageTransforms, ROI_SIZE, build_image_transform

# Merlin's historical default preprocessing geometry. A DataLoader built at this
# geometry keeps using the pre-existing (un-suffixed / organ-only-suffixed) cache
# dirs, so the large warmed default-resolution cache is never orphaned by adding
# configurable pixdim/spatial_size.
DEFAULT_PIXDIM = (1.5, 1.5, 3.0)
DEFAULT_SPATIAL_SIZE = tuple(ROI_SIZE)  # (224, 224, 160)


def _localization_csv_hash(path):
    """Short content hash of the organ-localization CSV (OQ17 cache key)."""
    import hashlib

    if not path:
        return "nocsv"
    try:
        with open(path, "rb") as handle:
            return hashlib.md5(handle.read()).hexdigest()[:8]
    except OSError:
        return "missing"


def _fmt_pixdim_component(value):
    """Format a spacing component for a cache key: ``1.0`` -> ``'1'``, ``1.5`` -> ``'1p5'``.

    Mirrors contrastive-3d-onc's canonical suffix (``%g`` then ``.`` -> ``p``) so a
    MerlinOnc run and a c3d run at the same geometry share one persistent-transform
    cache dir -- the ``%g`` spelling normalizes ``1`` == ``1.0``.
    """
    try:
        return ("%g" % float(value)).replace(".", "p")
    except (TypeError, ValueError):
        return str(value)


def _geometry_cache_suffix(
    pixdim, spatial_size, crop_mode, organ, crop_anchor, superior_buffer,
    organ_coordinates_path,
):
    """Canonical cache-dir suffix folding pixdim + spatial_size + crop_mode (all
    modes) + organ inputs, matching contrastive-3d-onc ``datasets/dataloaders.py``.

    CTPersistentDataset hashes only ``{"image": path}``, so without this namespace
    two configs that differ only in geometry/crop would silently serve each other's
    cached tensors. Using the c3d spelling means the same geometry shares one cache
    across both repos.
    """
    pix_s = "_".join(_fmt_pixdim_component(v) for v in pixdim)
    size_s = "_".join(str(int(v)) for v in spatial_size)
    suffix = f"pix_{pix_s}__size_{size_s}__crop_{crop_mode}"
    if crop_mode == "organ_centered":
        suffix += (
            f"__organ_{organ}__anchor_{crop_anchor}"
            f"__buf_{superior_buffer}__csv_{_localization_csv_hash(organ_coordinates_path)}"
        )
    return suffix


class CTPersistentDataset(monai.data.PersistentDataset):
    def __init__(self, data, transform, cache_dir=None):
        super().__init__(data=data, transform=transform, cache_dir=cache_dir)

        print(f"Size of dataset: {self.__len__()}\n")

    def _cachecheck(self, item_transformed):
        hashfile = None
        _item_transformed = deepcopy(item_transformed)
        image_data = {
            "image": item_transformed.get("image")
        }  # Assuming the image data is under the 'image' key

        if self.cache_dir is not None and image_data is not None:
            data_item_md5 = self.hash_func(image_data).decode(
                "utf-8"
            )  # Hash based on image data
            hashfile = self.cache_dir / f"{data_item_md5}.pt"

        if hashfile is not None and hashfile.is_file():
            cached_image = torch.load(hashfile, weights_only=False)
            _item_transformed["image"] = cached_image
            return _item_transformed

        _image_transformed = self._pre_transform(image_data)["image"]
        _item_transformed["image"] = _image_transformed
        if hashfile is None:
            return _item_transformed
        try:
            # NOTE: Writing to a temporary directory and then using a nearly atomic rename operation
            #       to make the cache more robust to manual killing of parent process
            #       which may leave partially written cache files in an incomplete state
            with tempfile.TemporaryDirectory() as tmpdirname:
                temp_hash_file = Path(tmpdirname) / hashfile.name
                torch.save(
                    obj=_image_transformed,
                    f=temp_hash_file,
                    pickle_module=look_up_option(
                        self.pickle_module, SUPPORTED_PICKLE_MOD
                    ),
                    pickle_protocol=self.pickle_protocol,
                )
                if temp_hash_file.is_file() and not hashfile.is_file():
                    # On Unix, if target exists and is a file, it will be replaced silently if the user has permission.
                    # for more details: https://docs.python.org/3/library/shutil.html#shutil.move.
                    try:
                        shutil.move(str(temp_hash_file), hashfile)
                    except FileExistsError:
                        pass
        except PermissionError:  # project-monai/monai issue #3613
            pass
        return _item_transformed

    def _transform(self, index: int):
        pre_random_item = self._cachecheck(self.data[index])
        return self._post_transform(pre_random_item)


class DataLoader(monai.data.DataLoader):
    def __init__(
        self,
        datalist: List[dict],
        cache_dir: str,
        batchsize: int,
        shuffle: bool = True,
        num_workers: int = 0,
        crop_mode: str = "center",
        organ_coordinates_path: str = None,
        organ: str = "lungs",
        crop_anchor: str = "apex",
        superior_buffer: int = 5,
        pixdim=None,
        spatial_size=None,
    ):
        self.datalist = datalist
        self.batchsize = batchsize

        # Resolve geometry. ``None`` -> Merlin's historical defaults, so existing
        # default-geometry callers (and their warmed caches) are byte-for-byte
        # untouched; only a non-default pixdim/spatial_size opts into namespacing.
        pixdim = DEFAULT_PIXDIM if pixdim is None else tuple(float(x) for x in pixdim)
        spatial_size = (
            list(DEFAULT_SPATIAL_SIZE)
            if spatial_size is None
            else [int(v) for v in spatial_size]
        )
        is_default_geometry = (
            tuple(pixdim) == DEFAULT_PIXDIM
            and tuple(spatial_size) == DEFAULT_SPATIAL_SIZE
        )

        if is_default_geometry and crop_mode == "center":
            transform = ImageTransforms
        elif is_default_geometry and crop_mode == "organ_centered":
            transform = build_image_transform(
                crop_mode,
                organ_coordinates_path=organ_coordinates_path,
                organ=organ,
                crop_anchor=crop_anchor,
                superior_buffer=superior_buffer,
            )
            # OQ17: the CTPersistentDataset cache key hashes only the image path,
            # so distinct organ/anchor/CSVs would collide in one cache_dir. Give
            # organ_centered its own cache namespace (this also fixes the vista-ct
            # pass-through, which delegates CT preprocessing to this DataLoader).
            # Default-geometry organ_centered keeps its pre-resolution suffix so
            # the warmed apex caches keep hitting.
            cache_dir = str(
                Path(cache_dir)
                / (
                    f"organ_centered__organ_{organ}__anchor_{crop_anchor}"
                    f"__buf_{superior_buffer}__csv_{_localization_csv_hash(organ_coordinates_path)}"
                )
            )
        elif is_default_geometry:
            # default-geometry top_centered (or any other non-center mode): keep
            # using the base cache_dir, preserving the large warmed default cache.
            transform = build_image_transform(crop_mode)
        else:
            # Non-default geometry: build with the resolved pixdim/spatial_size and
            # fold the full geometry into the cache identity so distinct resolutions
            # never collide -- and so this run shares the c3d-warmed cache dir for
            # the same geometry (e.g. pix_1_1_1p5__size_336_336_320__crop_top_centered).
            transform = build_image_transform(
                crop_mode,
                organ_coordinates_path=organ_coordinates_path,
                organ=organ,
                crop_anchor=crop_anchor,
                superior_buffer=superior_buffer,
                pixdim=pixdim,
                spatial_size=spatial_size,
            )
            cache_dir = str(
                Path(cache_dir)
                / _geometry_cache_suffix(
                    pixdim, spatial_size, crop_mode, organ, crop_anchor,
                    superior_buffer, organ_coordinates_path,
                )
            )
        self.cache_dir = cache_dir
        self.dataset = CTPersistentDataset(
            data=datalist,
            transform=transform,
            cache_dir=cache_dir,
        )
        super().__init__(
            self.dataset,
            batch_size=batchsize,
            shuffle=shuffle,
            num_workers=num_workers,
        )
