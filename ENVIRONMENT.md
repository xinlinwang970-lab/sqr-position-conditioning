# Reproduction Environment

## Python / CUDA

- **Python**: 3.11.x
- **PyTorch**: 2.3.1+cu121
- **CUDA**: 12.1
- **GPU**: NVIDIA V100S 32GB (experiments use 10-24 GB depending on architecture)

## Key dependencies

```
torch>=2.0
numpy
soundfile
tqdm
pyyaml
```

## Conda environment

```bash
conda create -n snse python=3.11
conda activate snse
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install soundfile tqdm pyyaml numpy
```

## Verify installation

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.3.1+cu121 True
```
