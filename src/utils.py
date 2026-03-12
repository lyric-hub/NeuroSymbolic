import torch

def get_device() -> torch.device:
    """Detects and returns the optimal hardware device (specifically for DGX Spark)."""
    if torch.cuda.is_available():
        # DGX Spark uses the GB10 GPU
        return torch.device("cuda:0")
    # Fallback for local testing
    return torch.device("cpu")