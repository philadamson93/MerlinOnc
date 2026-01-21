# Dataloading

## Overview

Merlin uses MONAI-based data loading with persistent caching for efficient CT image processing.

**Key Files:**
- `merlin/data/dataloaders.py` - Dataset and DataLoader classes
- `merlin/data/monai_transforms.py` - Image preprocessing pipeline
- `merlin/data/download_data.py` - Sample data download utility

## Classes

### CTPersistentDataset

Extends `monai.data.PersistentDataset` with CT-specific optimizations.

**Location:** `merlin/data/dataloaders.py:14`

```python
class CTPersistentDataset(monai.data.PersistentDataset):
    def __init__(self, data, transform, cache_dir=None)
```

**Features:**
- MD5-based caching of preprocessed images
- Atomic file writing (prevents corruption from interrupted processes)
- Caches only image data, not metadata

**Cache Behavior:**
- Hashes image data to generate unique cache filename
- Stores preprocessed tensors as `.pt` files
- Uses temporary directory + atomic move for safe writes

### DataLoader

Convenience wrapper around MONAI DataLoader.

**Location:** `merlin/data/dataloaders.py:72`

```python
from merlin.data import DataLoader

loader = DataLoader(
    datalist=[{"image": "path/to/scan.nii.gz", ...}],
    cache_dir="./cache",
    batchsize=4,
    shuffle=True,
    num_workers=0
)
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `datalist` | `List[dict]` | List of dicts with "image" key pointing to NIfTI path |
| `cache_dir` | `str` | Directory for caching preprocessed images |
| `batchsize` | `int` | Batch size for loading |
| `shuffle` | `bool` | Whether to shuffle data (default: True) |
| `num_workers` | `int` | Number of worker processes (default: 0) |

## Image Transforms Pipeline

**Location:** `merlin/data/monai_transforms.py`

The `ImageTransforms` compose applies these transforms in order:

| Step | Transform | Parameters | Purpose |
|------|-----------|------------|---------|
| 1 | `LoadImaged` | keys=["image"] | Load NIfTI file |
| 2 | `EnsureChannelFirstd` | keys=["image"] | Standardize to (C, H, W, D) |
| 3 | `Orientationd` | axcodes="RAS" | Convert to RAS orientation |
| 4 | `Spacingd` | pixdim=(1.5, 1.5, 3) | Resample to uniform voxel spacing |
| 5 | `ScaleIntensityRanged` | a_min=-1000, a_max=1000 → 0-1 | HU normalization |
| 6 | `SpatialPadd` | spatial_size=[224, 224, 160] | Pad to minimum size |
| 7 | `CenterSpatialCropd` | roi_size=[224, 224, 160] | Crop to exact size |
| 8 | `ToTensord` | keys=["image"] | Convert to PyTorch tensor |

**Output Shape:** `(1, 224, 224, 160)` - single channel, 224x224x160 voxels

## Data Format

### Input Format

CT scans must be in NIfTI format (`.nii` or `.nii.gz`).

**Datalist structure:**
```python
datalist = [
    {"image": "/path/to/scan1.nii.gz", "label": 0, "patient_id": "001"},
    {"image": "/path/to/scan2.nii.gz", "label": 1, "patient_id": "002"},
]
```

The `"image"` key is required. Additional keys (label, patient_id, etc.) are passed through.

### Sample Data

Download a sample CT scan:

```python
from merlin.data import download_sample_data

download_sample_data(save_dir="./data")
# Downloads: image1.nii.gz
```

## Usage Example

```python
from merlin.data import DataLoader
from merlin.models import Merlin

# Prepare datalist
datalist = [
    {"image": "./data/image1.nii.gz"},
    {"image": "./data/image2.nii.gz"},
]

# Create loader with caching
loader = DataLoader(
    datalist=datalist,
    cache_dir="./cache",
    batchsize=2,
    shuffle=False,
    num_workers=4
)

# Load model
model = Merlin()
model.eval()

# Process batches
for batch in loader:
    images = batch["image"]  # Shape: (B, 1, 224, 224, 160)
    embeddings = model(images)
```

## Cache Management

**Cache Location:** Specified by `cache_dir` parameter

**Clearing Cache:**
```bash
rm -rf ./cache  # Or your cache_dir path
```

Clear cache when:
- Modifying `ImageTransforms` pipeline
- Updating MONAI version
- Encountering corrupted cache files

**Cache Files:** Named by MD5 hash of image data, e.g., `a1b2c3d4e5f6.pt`
