import torch

def get_device():
    """
    Returns the optimal device available on the system:
    'cuda' for NVIDIA GPUs, 'mps' for Apple Silicon, or 'cpu'.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_vram_stats():
    """
    Returns a tuple (allocated_gb, reserved_gb) for the active GPU.
    Returns (0.0, 0.0) if no GPU is available.
    """
    device = get_device()
    if device == "cuda":
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        return allocated, reserved
    elif device == "mps":
        allocated = torch.mps.current_allocated_memory() / (1024**3)
        # MPS doesn't have a direct equivalent to 'reserved' memory in the same way,
        # but driver_allocated_memory is the closest approximation to total footprint.
        reserved = torch.mps.driver_allocated_memory() / (1024**3)
        return allocated, reserved
    return 0.0, 0.0

def get_peak_vram_allocated_mb():
    """
    Returns the peak memory allocated in MB.
    """
    device = get_device()
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / (1024**2)
    elif device == "mps":
        # MPS doesn't expose max memory allocated directly like CUDA.
        # We fallback to current allocated.
        return torch.mps.current_allocated_memory() / (1024**2)
    return 0.0

def reset_vram_stats():
    """
    Resets peak memory statistics for the active GPU.
    """
    device = get_device()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # MPS does not support resetting peak memory stats as of current PyTorch versions.

def empty_cache():
    """
    Empties the cache for the active GPU to free up unreferenced memory.
    """
    device = get_device()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

def synchronize():
    """
    Synchronizes device operations (waits for all kernels to finish).
    """
    device = get_device()
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
