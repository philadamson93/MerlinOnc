# Inference

## Quick Start

```python
import torch
from merlin import Merlin
from merlin.data import DataLoader, download_sample_data

device = "cuda" if torch.cuda.is_available() else "cpu"

# Download sample data
data_dir = "./data"
image_path = download_sample_data(data_dir)

# Create datalist
datalist = [{"image": image_path}]

# Load data
loader = DataLoader(
    datalist=datalist,
    cache_dir="./cache",
    batchsize=1,
    shuffle=False,
    num_workers=0,
)

# Load model
model = Merlin()
model.eval()
model.to(device)
```

## Inference Modes

### 1. Contrastive Embeddings (Default)

Returns image embeddings, phenotype predictions, and text embeddings for contrastive learning.

```python
model = Merlin()
model.eval()
model.cuda()

datalist = [{
    "image": "/path/to/scan.nii.gz",
    "text": "Radiology report text here..."
}]

loader = DataLoader(datalist=datalist, cache_dir="./cache", batchsize=1)

for batch in loader:
    outputs = model(batch["image"].cuda(), batch["text"])

    image_embeddings = outputs[0]    # Shape: (B, 512)
    phenotype_preds = outputs[1]     # Shape: (B, 1692)
    text_embeddings = outputs[2]     # Shape: (B, 512)
```

### 2. Image Embeddings Only

Extract 2048-dim features for downstream transfer learning.

```python
model = Merlin(ImageEmbedding=True)
model.eval()
model.cuda()

for batch in loader:
    embeddings = model(batch["image"].cuda())
    # Shape: (1, B, 2048)
```

### 3. Phenotype Classification

Predict 1,692 clinical phenotypes with probabilities.

```python
import pandas as pd
import numpy as np

model = Merlin(PhenotypeCls=True)
model.eval()
model.cuda()

# Load phenotype lookup table
phenotypes = pd.read_csv("documentation/phenotypes.csv")

for batch in loader:
    probs = model(batch["image"].cuda())
    probs = probs.squeeze(0).cpu().numpy()  # Shape: (1692,)

    # Get top predictions
    top_indices = np.argsort(probs)[::-1][:5]
    for idx in top_indices:
        phecode = phenotypes.iloc[idx, 0]
        description = phenotypes.iloc[idx, 1]
        print(f"{phecode}: {description} ({probs[idx]:.4f})")
```

**Phenotype Lookup:** `documentation/phenotypes.csv` contains 1,692 phecodes with descriptions.

### 4. Five-Year Disease Prediction

Predict probability of 6 diseases within 5 years.

```python
model = Merlin(FiveYearPred=True)
model.eval()
model.cuda()

disease_names = [
    "Cardiovascular Disease (CVD)",
    "Ischemic Heart Disease (IHD)",
    "Hypertension (HTN)",
    "Diabetes Mellitus (DM)",
    "Chronic Kidney Disease (CKD)",
    "Chronic Liver Disease (CLD)",
]

for batch in loader:
    probs = model(batch["image"].cuda())
    probs = probs.squeeze(0)  # Shape: (6,)

    for disease, prob in zip(disease_names, probs):
        print(f"{disease}: {prob:.4f}")
```

### 5. Radiology Report Generation

Generate automated radiology reports per organ system.

**Requirements:** 48GB VRAM (A6000 or better)

**Run with accelerate:**
```bash
accelerate launch --mixed_precision fp16 your_script.py
```

```python
from transformers import StoppingCriteria

class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence=[48134]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids, scores, **kwargs):
        last_ids = input_ids[:, -len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids

model = Merlin(RadiologyReport=True)
model.eval()
model.cuda()

# Organ systems for report generation
organ_systems = [
    "lower thorax", "liver", "gallbladder", "spleen",
    "pancreas", "adrenal glands", "kidneys", "bowel",
    "peritoneum", "pelvic", "circulatory", "lymph nodes",
    "musculoskeletal"
]

for batch in loader:
    images = batch["image"].cuda()

    for organ in organ_systems:
        prefix = f"Generate a radiology report for {organ}###\n"
        prefix_batch = [prefix] * len(images)

        reports = model.generate(
            images,
            prefix_batch,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.2,
            max_new_tokens=128,
            stopping_criteria=[EosListStoppingCriteria()],
        )

        # Extract report text
        report_text = reports[0].split("###")[0]
        print(f"{organ}: {report_text}")
```

### 6. MerlinOnc (Oncology Model)

Use the oncology-specific MerlinOnc weights.

```python
# Download from HuggingFace (philadamson93/MerlinOnc)
model = Merlin(MerlinOnc=True)
model.eval()
model.cuda()

for batch in loader:
    outputs = model(batch["image"].cuda(), batch["text"])
```

MerlinOnc can be combined with output modes:

```python
# MerlinOnc with image embeddings
model = Merlin(MerlinOnc=True, ImageEmbedding=True)

# MerlinOnc with phenotype classification
model = Merlin(MerlinOnc=True, PhenotypeCls=True)
```

### Using Local Weights

Skip HuggingFace download by specifying a local checkpoint path:

```python
# Use local weights file
model = Merlin(
    MerlinOnc=True,
    local_checkpoint_path="/path/to/local/weights.pt"
)
model.eval()
model.cuda()

# Works with any model and output mode
model = Merlin(
    local_checkpoint_path="/path/to/custom_weights.pt",
    ImageEmbedding=True
)
```

This is useful when:
- You have pre-downloaded weights
- You're using weights not hosted on HuggingFace
- You want to avoid network requests

## Input Requirements

### Image Format

- **Format:** NIfTI (`.nii` or `.nii.gz`)
- **Preprocessing:** Handled automatically by `DataLoader`
- **Final Shape:** `(B, 1, 224, 224, 160)`

### Text Format

- Plain text radiology reports
- Automatically sanitized (lowercased, word tokenized)
- Max length: 1024 tokens

## GPU Memory Requirements

| Mode | Approximate VRAM |
|------|------------------|
| Image Embedding | ~8 GB |
| Phenotype Classification | ~8 GB |
| Disease Prediction | ~8 GB |
| Contrastive (with text) | ~12 GB |
| Report Generation | ~48 GB |

## Batch Processing

```python
datalist = [
    {"image": "/path/to/scan1.nii.gz"},
    {"image": "/path/to/scan2.nii.gz"},
    {"image": "/path/to/scan3.nii.gz"},
]

loader = DataLoader(
    datalist=datalist,
    cache_dir="./cache",
    batchsize=4,
    shuffle=False,
    num_workers=4,
)

model = Merlin(PhenotypeCls=True)
model.eval()
model.cuda()

all_predictions = []
for batch in loader:
    preds = model(batch["image"].cuda())
    all_predictions.append(preds.cpu())

all_predictions = torch.cat(all_predictions, dim=0)
```

## CPU Inference

CPU inference is supported but significantly slower:

```python
device = "cpu"
model = Merlin()
model.eval()
model.to(device)

for batch in loader:
    outputs = model(batch["image"].to(device), batch["text"])
```

## Demo Scripts

Full working examples are in the `documentation/` folder:

- `documentation/demo.py` - All inference modes except report generation
- `documentation/report_generation_demo.py` - Report generation example
