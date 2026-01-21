# Development

## Setup

### Prerequisites

- Python 3.9+ (3.10 recommended)
- CUDA-capable GPU (recommended)
- Git

### Installation

**From PyPI (user installation):**
```bash
pip install merlin-vlm
```

**Development installation:**
```bash
# Clone repository
git clone https://github.com/StanfordMIMI/Merlin.git
cd Merlin

# Create conda environment
conda create --name merlin python==3.10
conda activate merlin

# Install in editable mode
pip install -e .

# Install dev dependencies
pip install -e ".[dev]"
```

**Exact reproducible environment (with uv):**
```bash
uv sync
```

## Code Quality

### Linting & Formatting

The project uses **Ruff** for linting and formatting.

**Run manually:**
```bash
# Lint with auto-fix
ruff check --fix .

# Format code
ruff format .
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`.

**Setup:**
```bash
pre-commit install
```

**Run manually:**
```bash
pre-commit run --all-files
```

**Configured hooks:**
| Hook | Version | Purpose |
|------|---------|---------|
| ruff | v0.11.2 | Python linting with auto-fix |
| ruff-format | v0.11.2 | Python formatting |
| mdformat | 0.7.13 | Markdown formatting |

### CI/CD

GitHub Actions runs ruff linting on all pushes and PRs.

**Workflow:** `.github/workflows/ruff.yml`

## Project Structure

```
merlin/
├── __init__.py          # Package exports
├── models/
│   ├── __init__.py      # Model exports
│   ├── load.py          # Merlin class (entry point)
│   ├── build.py         # MerlinArchitecture
│   ├── i3res.py         # I3ResNet
│   ├── inflate.py       # 2D→3D inflation
│   └── radiology_report_generation.py
├── data/
│   ├── __init__.py      # Data exports
│   ├── dataloaders.py   # CTPersistentDataset, DataLoader
│   ├── monai_transforms.py  # Image preprocessing
│   └── download_data.py # Sample data download
└── utils/
    ├── __init__.py      # Utils exports
    └── huggingface_download.py
```

## Dependencies

### Core Dependencies

| Package | Min Version | Tested Version |
|---------|-------------|----------------|
| torch | >=2.1.2 | 2.1.2 |
| monai | >=1.3.0 | 1.3.0 |
| transformers | >=4.38.2 | 4.38.2 |
| torchvision | >=0.16.2 | 0.16.2 |
| huggingface_hub | - | 0.30.1 |
| numpy | >=1.26.4 | 1.26.4 |
| nibabel | - | 5.3.2 |
| einops | - | 0.8.1 |
| peft | - | 0.10.0 |
| accelerate | - | 0.34.2 |
| nltk | - | 3.9.1 |
| pandas | - | 2.3.2 |
| rich | - | 14.1.0 |
| matplotlib | - | 3.9.4 |
| sentencepiece | - | 0.2.1 |
| protobuf | - | 6.32.1 |

### Dev Dependencies

```bash
pip install ruff pre-commit mdformat
```

## Testing

Currently no automated unit tests. Validation is done through demo scripts:

```bash
# Run inference demo
python documentation/demo.py

# Run report generation demo (requires 48GB GPU)
accelerate launch --mixed_precision fp16 documentation/report_generation_demo.py
```

## Adding New Features

### Adding a New Model Mode

1. **Update MODEL_CONFIGS** in `merlin/models/load.py`:
   ```python
   MODEL_CONFIGS["new_task"] = {
       "builder": NewModelClass,
       "checkpoint": "checkpoint_filename.pt",
   }
   ```

2. **Add flag to Merlin.__init__**:
   ```python
   def __init__(self, ..., NewTask: bool = False):
   ```

3. **Update task selection logic** in `__init__`

4. **Update _load_model()** if custom loading needed

5. **Upload checkpoint** to HuggingFace Hub (`stanfordmimi/Merlin`)

### Adding New Transforms

1. Edit `merlin/data/monai_transforms.py`
2. Add transform to `ImageTransforms` Compose
3. Clear cache after changes: `rm -rf abct_data_cache/`

### Modifying I3ResNet

Key locations in `merlin/models/i3res.py`:
- Output heads: lines 48-58
- Forward pass: lines 66-118
- Checkpoint memory optimization: uses `torch.utils.checkpoint`

## Checkpoints

Checkpoints are stored on HuggingFace Hub: `stanfordmimi/Merlin`

**Local cache:** `merlin/models/checkpoints/`

**Available checkpoints:**
| File | Size | Task |
|------|------|------|
| `i3_resnet_clinical_longformer_best_clip_*.pt` | ~5 GB | Default/Contrastive |
| `resnet_gpt2_best_stanford_report_generation_average.pt` | ~25 GB | Report Generation |
| `resnet_clinical_longformer_five_year_disease_prediction.pt` | ~5 GB | Disease Prediction |

## Releasing

1. Update version in `pyproject.toml`
2. Run pre-commit hooks: `pre-commit run --all-files`
3. Create git tag: `git tag v0.0.X`
4. Build: `python -m build`
5. Upload to PyPI: `twine upload dist/*`

## Troubleshooting

### CUDA Out of Memory

- Reduce batch size
- Use gradient checkpointing (already enabled)
- For report generation, ensure 48GB VRAM

### Cached Data Issues

Clear the cache directory:
```bash
rm -rf abct_data_cache/
```

### HuggingFace Download Errors

Check network connection and HuggingFace Hub status. Checkpoints will retry on next model load.

### Pre-commit Hook Failures

```bash
# Update hooks
pre-commit autoupdate

# Run with verbose output
pre-commit run --all-files --verbose
```
