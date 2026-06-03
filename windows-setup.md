# Windows 11 CUDA Setup

Personal setup notes for running this project on Windows 11 with Git Bash / bash and an NVIDIA RTX 5080 Laptop GPU.

## Baseline

- Python: 3.14
- Package manager: uv
- GPU target: NVIDIA RTX 50-series / Blackwell, tested with RTX 5080 Laptop GPU
- PyTorch: `torch==2.11.0+cu130`
- Torchvision: `torchvision==0.26.0+cu130`

The PyTorch wheels in `pyproject.toml` are pinned to the official CUDA 13.0 wheel index. This matters for RTX 5080 support because the GPU reports compute capability `sm_120`; older CUDA/PyTorch wheels can install successfully but fail or warn at runtime.

You do not need to install the full CUDA Toolkit just to run these PyTorch wheels. You do need a current NVIDIA driver. This machine validated with NVIDIA driver `596.36`, where `nvidia-smi` reports CUDA runtime compatibility `13.2`.

## Setup

```bash
uv sync --frozen
source .venv/Scripts/activate
```

Run the CUDA smoke test:

```bash
python scripts/check_cuda.py
```

Expected result:

- `torch` reports `2.11.0+cu130`
- CUDA is available
- the GPU name includes RTX 5080
- compute capability is `(12, 0)`
- the test CUDA operation completes

Use notebooks from PyCharm by selecting the project virtual environment as the notebook interpreter/kernel:

- Interpreter: `.venv/Scripts/python.exe`
- Activate manually in bash only when running terminal commands.

## Notes

- Keep `.python-version` aligned with `pyproject.toml`; both currently target Python 3.14.
- Keep the `pytorch-cu130` uv index when updating dependencies.
- If CUDA stops working after pulling another branch, first check for accidental downgrades to CPU-only, CUDA 12.1, or CUDA 12.4 PyTorch wheels.
- The notebook currently uses direct `.cuda()` calls. That is fine for this Windows 11/RTX 5080 setup, but CPU fallback would require changing those calls to a shared `device = torch.device(...)` pattern.
