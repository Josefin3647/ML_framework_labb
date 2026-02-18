# The following code is chatgpt generated/Josefin

import sys


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    raise SystemExit(1)


def ok(msg: str) -> None:
    print(f"[ OK ] {msg}")


def main() -> None:
    # Python version
    ok(f"Python: {sys.version.split()[0]}")

    # Import packages
    try:
        import torch
    except Exception as e:
        fail(f"Could not import torch: {e}")

    try:
        import sklearn
    except Exception as e:
        fail(f"Could not import scikit-learn: {e}")

    try:
        import pandas as pd
    except Exception as e:
        fail(f"Could not import pandas: {e}")

    try:
        import jupyter_core
    except Exception as e:
        fail(f"Could not import jupyter_core: {e}")

    # Versions
    ok(f"torch: {torch.__version__}")
    ok(f"scikit-learn: {sklearn.__version__}")
    ok(f"pandas: {pd.__version__}")
    ok(f"jupyter: {jupyter_core.__version__}")

    # GPU / accelerator check
    cuda_ok = torch.cuda.is_available()
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    ok(f"CUDA available: {cuda_ok}")
    ok(f"MPS available: {mps_ok}")

    # Select device
    if cuda_ok:
        device = torch.device("cuda")
    elif mps_ok:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ok(f"Using device: {device}")

    # Tensor computation
    try:
        a = torch.randn(1024, 1024, device=device)
        b = torch.randn(1024, 1024, device=device)

        c = a @ b

        if not torch.isfinite(c).all():
            fail("Tensor computation produced NaN/Inf")

    except Exception as e:
        fail(f"Tensor computation failed: {e}")

    ok("Tensor matrix multiplication OK")

    print("\nEnvironment verification PASSED ✅")


if __name__ == "__main__":
    main()
