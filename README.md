# gpu-and-vision-benchmarks

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![uv](https://img.shields.io/badge/uv-dependency%20manager-purple)
![License](https://img.shields.io/badge/license-MIT-green)

Two notebooks exploring GPU acceleration and pretrained vision models: one that benchmarks ML framework performance across available hardware backends, and one that runs a full vision pipeline (classification, captioning, and semantic search) locally using PyTorch.

---

## What's inside

**`gpu_stress_test.ipynb`** — Measures matrix-multiplication throughput (GFLOPS) for PyTorch, TensorFlow, and CuPy. Automatically detects available backends (CUDA, MPS on Apple Silicon, CPU fallback) and runs each benchmark with warmup and timing. TensorFlow and CuPy degrade gracefully if not installed or not supported on the current platform.

**`vision_models_pipeline.ipynb`** — A local inference pipeline over images using three pretrained models from HuggingFace:

| Model | Task |
|-------|------|
| `google/vit-base-patch16-224` | Image classification (1000 ImageNet classes) |
| `Salesforce/blip-image-captioning-base` | Free-form image captioning |
| `openai/clip-vit-base-patch32` | Semantic image search via cosine similarity |

Models run on CUDA, MPS, or CPU depending on what's available. Sample images are included in `images/` so the pipeline is runnable immediately after installation.

---

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — dependency manager
- GPU acceleration is optional. Both notebooks detect available hardware at runtime and fall back to CPU. On CPU, the stress test benchmarks are slower and the matrix sizes are intentionally large, so expect several minutes per PyTorch run.

---

## Installation

TensorFlow is only needed for `gpu_stress_test.ipynb` and is declared as an optional dependency group. Install accordingly:

**Vision pipeline only:**
```bash
uv pip install -e .
```

**Vision pipeline + stress test (includes TensorFlow):**
```bash
uv pip install -e ".[benchmark]"
```

**Register the kernel with Jupyter** so the notebooks use the project's environment rather than the system Python:
```bash
python -m ipykernel install --user --name gpu-and-vision-benchmarks --display-name "Python (gpu-and-vision-benchmarks)"
```

When opening a notebook, select the `Python (gpu-and-vision-benchmarks)` kernel from the kernel menu.

---

## Usage

```bash
uv run jupyter notebook
```

### gpu_stress_test.ipynb

Run the cells top to bottom. The notebook first checks installed frameworks and available devices, then runs each benchmark sequentially. Sample output on a CPU-only machine:

```
PYTORCH - matrix multiplication benchmark
No CUDA GPU detected — running on CPU
Creating 20000x20000 matrices on cpu...
Total time:   131.5 s
Throughput:   608.25 GFLOPS
```

On a CUDA machine, the same run completes in under a second per multiply and reports GPU memory usage alongside throughput.

### vision_models_pipeline.ipynb

Add images to the `images/` folder (`.jpg`, `.png`, `.jpeg`), then run top to bottom. The final cell runs a full analysis combining all three models on a single image:

```
1. CLASSIFICATION (Top 3):
   1. beach wagon, station wagon... (42.5%)
   2. grille, radiator grille (20.1%)
   3. car wheel (9.7%)

2. AUTOMATIC CAPTION:
   a car parked on the side of the road

3. CONCEPT SIMILARITY:
   animal     ████                 0.244
   vehicle    ████████             0.381
   nature     ███                  0.202
```

Earlier cells also run a semantic search over all images in the folder, ranking each one against text queries like "an animal" or "nature and landscapes".

Models are downloaded from HuggingFace on first run (~346 MB for ViT, ~990 MB for BLIP, ~350 MB for CLIP) and cached in `~/.cache/huggingface/hub`. Subsequent runs reuse the cached weights.

---

## Project structure

```
gpu-and-vision-benchmarks/
├── gpu_stress_test.ipynb        # PyTorch / TensorFlow / CuPy benchmarks
├── vision_models_pipeline.ipynb # ViT + BLIP + CLIP inference pipeline
├── pyproject.toml               # Dependencies (uv); TF in optional [benchmark] group
├── uv.lock                      # Locked dependency tree
└── images/                      # Sample images for the vision pipeline
    ├── auto.jpg
    ├── cascada.jpg
    ├── perro.png
    ├── pollo.jpg
    └── tornillo.jpeg
```

---

## Engineering notes

The standard `tensorflow` PyPI package has a known library path issue on Apple Silicon (ARM64): it ships a Metal plugin (`libmetal_plugin.dylib`) that references `_pywrap_tensorflow_internal.so` via an rpath that does not resolve in pip-installed environments. The effect is that `tf.__version__` and all top-level attributes are missing at import time. The correct approach on Apple Silicon is to use `tensorflow-macos` and `tensorflow-metal` instead of the standard package. `gpu_stress_test.ipynb` handles this gracefully — TensorFlow errors are caught per-function so the PyTorch and CuPy benchmarks still run normally.

TensorFlow was removed from the default dependency set and moved to the `[benchmark]` optional group because `vision_models_pipeline.ipynb` never used it directly. It had been declared as a top-level dependency, but `transformers` does not require TensorFlow (it lists it only as an optional extra via `transformers[tf]`), and none of the three vision models invoke it. Keeping it in the default install would force a large download of a broken package on Apple Silicon for no benefit.

Migrating from `transformers` 4.x to 5.x required one fix in `SemanticImageSearcher`: `CLIPModel.get_image_features()` and `get_text_features()` changed their return type from a plain tensor to `BaseModelOutputWithPooling`. The embedding extraction logic was updated to go through `model.vision_model` / `model.text_model` and apply `visual_projection` / `text_projection` explicitly, which is what the 4.x single-call API did internally. ViT and BLIP had no API changes across this version boundary.

---

## License

MIT
