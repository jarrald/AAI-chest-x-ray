import sys

import torch
import torchvision


def main() -> int:
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"torch cuda build: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("CUDA is not available to PyTorch.")
        return 1

    device_index = 0
    props = torch.cuda.get_device_properties(device_index)
    print(f"gpu: {torch.cuda.get_device_name(device_index)}")
    print(f"compute capability: {torch.cuda.get_device_capability(device_index)}")
    print(f"vram: {props.total_memory / (1024 ** 3):.1f} GiB")

    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"cuda matmul ok: {float(y[0, 0].detach().cpu()):.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
