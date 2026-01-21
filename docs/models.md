# Models

## Overview

Merlin's model architecture combines a 3D image encoder (inflated ResNet-152) with a text encoder (Clinical-Longformer) for vision-language tasks.

**Key Files:**
- `merlin/models/load.py` - Main entry point (`Merlin` class)
- `merlin/models/build.py` - `MerlinArchitecture` definition
- `merlin/models/i3res.py` - I3ResNet 3D model
- `merlin/models/inflate.py` - 2D→3D convolution inflation
- `merlin/models/radiology_report_generation.py` - Report generation model

## Model Registry

**Location:** `merlin/models/load.py:12`

```python
DEFAULT_REPO_ID = "stanfordmimi/Merlin"

MODEL_CONFIGS = {
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
    "merlin_onc": {
        "builder": MerlinArchitecture,
        "checkpoint": "i3_resnet_clinical_longformer_best_clip_10-08-2025_03-41-48_epoch_99.pt",
        "repo_id": "philadamson93/MerlinOnc",  # custom HuggingFace repo
    },
}
```

Checkpoints are downloaded from HuggingFace on first use and cached in `merlin/models/checkpoints/`. Each config can specify a custom `repo_id`; if omitted, `DEFAULT_REPO_ID` is used.

## Merlin Class

**Location:** `merlin/models/load.py:33`

Main user-facing wrapper class.

```python
class Merlin(nn.Module):
    def __init__(
        self,
        ImageEmbedding: bool = False,
        PhenotypeCls: bool = False,
        RadiologyReport: bool = False,
        FiveYearPred: bool = False,
        MerlinOnc: bool = False,
        local_checkpoint_path: str = None,
    )
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `ImageEmbedding` | bool | Return 2048-dim image features |
| `PhenotypeCls` | bool | Return 1692 phenotype probabilities |
| `RadiologyReport` | bool | Enable report generation mode |
| `FiveYearPred` | bool | Return 6 disease probabilities |
| `MerlinOnc` | bool | Use MerlinOnc weights (oncology-specific) |
| `local_checkpoint_path` | str | Path to local weights file (skips HuggingFace download) |

### Instantiation Modes

| Mode | Flag | Output | Use Case |
|------|------|--------|----------|
| Default | (none) | (image_features, ehr_features, text_features) | Contrastive learning |
| Image Embedding | `ImageEmbedding=True` | 2048-dim features | Transfer learning |
| Phenotype | `PhenotypeCls=True` | 1692 probabilities | Clinical phenotype prediction |
| Disease Prediction | `FiveYearPred=True` | 6 probabilities | 5-year disease outcomes |
| Report Generation | `RadiologyReport=True` | Generated text | Automated reports |
| MerlinOnc | `MerlinOnc=True` | Same as default | Oncology-specific model |

Only one of `ImageEmbedding`, `PhenotypeCls`, `FiveYearPred` can be True at a time.

### Using Local Weights

Skip HuggingFace download by providing a local path:

```python
# Use local weights file
model = Merlin(MerlinOnc=True, local_checkpoint_path="/path/to/weights.pt")

# Works with any mode
model = Merlin(
    MerlinOnc=True,
    ImageEmbedding=True,
    local_checkpoint_path="/path/to/weights.pt"
)
```

### Methods

- `forward(*args, **kwargs)` - Delegates to underlying model
- `generate(*args, **kwargs)` - Only available when `RadiologyReport=True`

## MerlinArchitecture

**Location:** `merlin/models/build.py:69`

Core architecture combining image and text encoders.

```python
class MerlinArchitecture(nn.Module):
    def __init__(
        self,
        init_logit_scale: float = 1.0,
        ImageEmbedding: bool = False,
        PhenotypeCls: bool = False,
        FiveYearPred: bool = False,
    )
```

### Components

- `encode_image` - `ImageEncoder` (wraps I3ResNet)
- `encode_text` - `TextEncoder` (Clinical-Longformer)
- `logit_scale` - Learnable temperature for contrastive loss

### Forward Pass

```python
def forward(self, image, text=None):
    # ImageEmbedding mode: returns 2048-dim features
    # PhenotypeCls mode: returns 1692 probabilities
    # FiveYearPred mode: returns 6 probabilities
    # Default mode (requires text): returns (image_features, ehr_features, text_features)
```

## ImageEncoder

**Location:** `merlin/models/build.py:12`

Wraps I3ResNet for image encoding.

**Output Dimensions:**
- Default: `(512, 1692)` - contrastive + phenotype features
- `ImageEmbedding=True`: `2048` - raw features
- `PhenotypeCls=True`: `1692` - phenotype probabilities
- `FiveYearPred=True`: `6` - disease probabilities

## TextEncoder

**Location:** `merlin/models/build.py:46`

Uses Clinical-Longformer for text encoding.

- **Model:** `yikuan8/Clinical-Longformer`
- **Max Length:** 1024 tokens
- **Output:** 512-dim embeddings (768→512 via linear layer)
- **Preprocessing:** Lowercase + word tokenization via `sanitize_report()`

## I3ResNet

**Location:** `merlin/models/i3res.py:11`

3D inflated ResNet-152 for CT volume processing.

```python
class I3ResNet(nn.Module):
    def __init__(
        self,
        resnet2d,           # Pretrained 2D ResNet-152
        frame_nb=16,        # Number of temporal frames
        class_nb=1000,      # Number of output classes
        conv_class=False,   # Use conv classifier
        return_skips=False, # Return skip connections
        ImageEmbedding=False,
        PhenotypeCls=False,
        FiveYearPred=False,
    )
```

### Architecture

1. Inflate 2D ResNet-152 to 3D convolutions
2. Use gradient checkpointing for memory efficiency
3. Output heads:
   - `classifier` - Conv3d(2048→class_nb) for phenotypes
   - `contrastive_head` - Conv3d(2048→512) for embeddings

### Input Processing

```python
# Input: (B, 1, 224, 224, 160)
x = x.permute(0, 1, 4, 2, 3)  # → (B, 1, 160, 224, 224)
x = torch.cat((x, x, x), dim=1)  # → (B, 3, 160, 224, 224) - RGB channels
```

## Report Generation Model

**Location:** `merlin/models/radiology_report_generation.py:165`

```python
class Clip3DForTextGeneration(nn.Module):
    def __init__(self):
        self.encode_image = ModifiedImageEncoder()  # I3ResNet → sequence
        self.decode_text = TextDecoder()            # GPT-2 with LoRA
        self.adapter = Adapter(2048, 4096)          # Feature projection
```

### Components

- **ModifiedImageEncoder** - Extracts spatial features as token sequence (N×2048)
- **Adapter** - Linear(2048→4096) to match LLM dimensions
- **TextDecoder** - RadLLaMA-7b with LoRA (r=512, alpha=16)

### Generate Method

```python
model.generate(
    image,          # CT scan tensor
    text_labels,    # Prompt template
    max_new_tokens=512,
    do_sample=True,
    top_p=0.9,
)
```

**Requirements:** 48GB VRAM (A6000 or better)

## Adding New Models

1. Add config to `MODEL_CONFIGS` in `merlin/models/load.py`
2. Create builder class with `__init__(**kwargs)` and `forward()`
3. Upload checkpoint to HuggingFace Hub
4. Update `Merlin.__init__()` to handle new flags
5. Update `_load_model()` for checkpoint loading logic
