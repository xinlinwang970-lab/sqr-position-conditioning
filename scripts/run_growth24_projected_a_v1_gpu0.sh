#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${REPO_ROOT}/configs/growth24_projected_a_v1_train.yaml"
TRAIN_ENTRY="${REPO_ROOT}/scripts/train_growth24_projected_a_v1.py"

# CUDA_VISIBLE_DEVICES indexes physical GPUs before masking. Inside Python,
# physical GPU 0 is therefore exposed as logical cuda:0.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is not available on this host." >&2
  exit 2
fi
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -n 1)"
if [[ "${GPU_NAME^^}" != *V100* ]]; then
  echo "ERROR: physical GPU 0 is not a V100 (found: ${GPU_NAME})." >&2
  exit 2
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  CANDIDATE_PYTHON="${PYTHON_BIN}"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  CANDIDATE_PYTHON="${REPO_ROOT}/.venv/bin/python"
else
  CANDIDATE_PYTHON=""
  for name in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
    candidate="$(command -v "${name}" || true)"
    if [[ -n "${candidate}" ]] && "${candidate}" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' >/dev/null 2>&1; then
      CANDIDATE_PYTHON="${candidate}"
      break
    fi
  done
  if [[ -z "${CANDIDATE_PYTHON}" ]]; then
    uv_candidate="${HOME}/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12"
    if [[ -x "${uv_candidate}" ]]; then
      CANDIDATE_PYTHON="${uv_candidate}"
    fi
  fi
fi
if [[ -z "${CANDIDATE_PYTHON}" ]]; then
  echo "ERROR: Python >=3.10 was not found." >&2
  echo "Create .venv with Python 3.10+, install the project and CUDA-enabled PyTorch, or set PYTHON_BIN." >&2
  exit 2
fi
PYTHON_BIN="${CANDIDATE_PYTHON}"
if ! "${PYTHON_BIN}" -c 'import torch' >/dev/null 2>&1; then
  echo "ERROR: PyTorch is not installed for ${PYTHON_BIN}." >&2
  echo "Create .venv with the repository-supported Python/CUDA/PyTorch versions, then rerun." >&2
  exit 2
fi

"${PYTHON_BIN}" - <<'PY'
import sys
import torch

if sys.version_info < (3, 10):
    raise SystemExit("Python >=3.10 is required")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this Python environment")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"Expected one visible GPU, found {torch.cuda.device_count()}")
name = torch.cuda.get_device_name(0)
if "V100" not in name.upper():
    raise SystemExit(f"Expected V100 on logical cuda:0, found {name}")
print(f"Preflight passed: Python {sys.version.split()[0]}, {name}")
PY

if [[ ! -f "${CONFIG}" || ! -f "${TRAIN_ENTRY}" ]]; then
  echo "ERROR: scheme A config or training entrypoint is missing." >&2
  exit 2
fi

LOG_DIR="${REPO_ROOT}/logs/growth24_projected_a_v1_gpu0"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
echo "Starting candidate A on physical GPU 0; log: ${LOG_FILE}"
"${PYTHON_BIN}" -u "${TRAIN_ENTRY}" 2>&1 | tee "${LOG_FILE}"
