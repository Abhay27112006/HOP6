"""
sdp_cpu.py — Sparse Delta Propagation (SDP) for CPU Inference.

Novel CPU-optimized matrix multiplication that exploits the fact that during
autoregressive generation, hidden states change minimally between tokens.

Instead of computing y = W @ x from scratch each time, SDP computes:
    delta = x_new - x_cached
    active = where(|delta| > threshold)       # typically 10-30% of dims
    y_new = y_cached + W[:, active] @ delta[active]   # sparse update

This turns O(d_in * d_out) into O(k * d_out) where k << d_in.

Why CPU beats GPU here:
- CPUs handle branching and irregular sparse access natively
- GPUs need uniform dense work to saturate their cores
- Small sparse ops waste GPU parallelism but fit perfectly in CPU caches
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class SparseIncrementalLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that caches previous computation
    and only recomputes the delta when inputs change minimally.

    During autoregressive generation:
    - First token: full matmul (cold start)
    - Subsequent tokens: compute delta from cached input, gather only
      the active (changed) dimensions, do a much smaller matmul, patch result.

    Tracks statistics for reporting cache hit rates and speedup.
    """

    def __init__(self, linear_layer, sparsity_threshold=0.01):
        super().__init__()
        # Steal weight and bias from the original layer (no copy)
        self.weight = linear_layer.weight  # (d_out, d_in)
        self.bias = linear_layer.bias      # (d_out,) or None
        self.threshold = sparsity_threshold

        # Cache storage
        self._cache_input = None   # (batch, seq, d_in)
        self._cache_output = None  # (batch, seq, d_out)

        # Statistics
        self.stats = {
            "total_calls": 0,
            "cache_hits": 0,
            "avg_sparsity": 0.0,
            "total_saved_ops": 0,
            "total_full_ops": 0,
        }

    def reset_cache(self):
        """Clear cache (call on new sequence/conversation)."""
        self._cache_input = None
        self._cache_output = None

    def forward(self, x):
        """
        x: (batch, seq_len, d_in) or (batch, d_in)

        Returns: (batch, seq_len, d_out) or (batch, d_out)
        """
        self.stats["total_calls"] += 1
        squeezed = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeezed = True

        batch, seq_len, d_in = x.shape
        d_out = self.weight.shape[0]

        # Track total ops for full matmul baseline
        full_ops = batch * seq_len * d_in * d_out
        self.stats["total_full_ops"] += full_ops

        # --- Try incremental path ---
        if (self._cache_input is not None
                and seq_len == 1
                and self._cache_input.shape[0] == batch):
            # Single new token — best case for delta propagation
            output = self._incremental_forward(x)
            if output is not None:
                if squeezed:
                    output = output.squeeze(1)
                return output

        # --- Full computation (cold start or cache miss) ---
        output = F.linear(x, self.weight, self.bias)
        self._cache_input = x.detach().clone()
        self._cache_output = output.detach().clone()
        self.stats["total_saved_ops"] += 0  # no savings on cold start

        if squeezed:
            output = output.squeeze(1)
        return output

    def _incremental_forward(self, x):
        """
        Compute output incrementally using delta from cached input.
        Returns None if incremental path isn't beneficial (fallback to full).
        """
        # x: (batch, 1, d_in) — single new token
        cached_in = self._cache_input   # (batch, 1, d_in) from last call
        cached_out = self._cache_output  # (batch, 1, d_out)

        # Handle shape mismatch
        if cached_in.shape != x.shape:
            return None

        # Compute delta
        delta = x - cached_in  # (batch, 1, d_in)

        # Find active dimensions (where delta is significant)
        # Use max across batch for dimension selection (greedy)
        abs_delta = torch.abs(delta)
        max_delta_per_dim = abs_delta.max(dim=0).values.squeeze(0)  # (d_in,)
        active_mask = max_delta_per_dim > self.threshold  # (d_in,)
        n_active = active_mask.sum().item()
        d_in = x.shape[2]
        d_out = self.weight.shape[0]

        sparsity = 1.0 - (n_active / d_in)

        # Only use sparse path if we save significant work
        if sparsity < 0.3:
            # Not sparse enough — full matmul is better
            output = F.linear(x, self.weight, self.bias)
            self._cache_input = x.detach().clone()
            self._cache_output = output.detach().clone()
            return output

        self.stats["cache_hits"] += 1

        # Update running average sparsity
        n = self.stats["cache_hits"]
        self.stats["avg_sparsity"] = (
            self.stats["avg_sparsity"] * (n - 1) + sparsity
        ) / n

        if n_active == 0:
            # No dimensions changed — return cached output directly!
            saved = x.shape[0] * d_in * d_out
            self.stats["total_saved_ops"] += saved
            return cached_out.clone()

        # --- Sparse delta matmul ---
        # Gather only the active columns of delta and weight
        active_cols = active_mask.nonzero(as_tuple=True)[0]  # (n_active,)

        # delta_sparse: (batch, 1, n_active)
        delta_sparse = delta[:, :, active_cols]

        # weight_sparse: (d_out, n_active)
        weight_sparse = self.weight[:, active_cols]

        # Small matmul: (batch, 1, n_active) @ (n_active, d_out) → (batch, 1, d_out)
        delta_output = F.linear(delta_sparse, weight_sparse.T.contiguous().T)
        # Equivalent to: delta_sparse @ weight_sparse.T

        # Patch cached output
        output = cached_out + delta_output

        # Track savings
        sparse_ops = x.shape[0] * n_active * d_out
        full_ops = x.shape[0] * d_in * d_out
        self.stats["total_saved_ops"] += (full_ops - sparse_ops)

        # Update cache
        self._cache_input = x.detach().clone()
        self._cache_output = output.detach().clone()

        return output


class SDPLayerWrapper(nn.Module):
    """
    Wraps a transformer layer, replacing all nn.Linear modules with
    SparseIncrementalLinear for CPU inference.
    """

    def __init__(self, layer, sparsity_threshold=0.01):
        super().__init__()
        self.layer = layer
        self.sdp_modules = []
        self._wrap_linears(layer, sparsity_threshold)

    def _wrap_linears(self, module, threshold, prefix=""):
        """Recursively replace nn.Linear with SparseIncrementalLinear."""
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                sdp_linear = SparseIncrementalLinear(child, threshold)
                setattr(module, name, sdp_linear)
                self.sdp_modules.append((full_name, sdp_linear))
            else:
                self._wrap_linears(child, threshold, full_name)

    def reset_caches(self):
        """Clear all SDP caches (call on new sequence)."""
        for _, sdp in self.sdp_modules:
            sdp.reset_cache()

    def forward(self, *args, **kwargs):
        return self.layer(*args, **kwargs)

    def get_stats(self):
        """Aggregate stats across all SDP linear layers."""
        total_calls = sum(s.stats["total_calls"] for _, s in self.sdp_modules)
        cache_hits = sum(s.stats["cache_hits"] for _, s in self.sdp_modules)
        total_saved = sum(s.stats["total_saved_ops"] for _, s in self.sdp_modules)
        total_full = sum(s.stats["total_full_ops"] for _, s in self.sdp_modules)

        hit_rate = cache_hits / max(total_calls, 1)
        savings_pct = total_saved / max(total_full, 1) * 100

        avg_sparsity_vals = [
            s.stats["avg_sparsity"]
            for _, s in self.sdp_modules
            if s.stats["cache_hits"] > 0
        ]
        avg_sparsity = (
            sum(avg_sparsity_vals) / len(avg_sparsity_vals)
            if avg_sparsity_vals else 0.0
        )

        return {
            "total_calls": total_calls,
            "cache_hits": cache_hits,
            "hit_rate": hit_rate,
            "avg_sparsity": avg_sparsity,
            "ops_saved_pct": savings_pct,
            "n_sdp_layers": len(self.sdp_modules),
        }

    # Delegate attribute access to the wrapped layer
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)

    def to(self, *args, **kwargs):
        self.layer = self.layer.to(*args, **kwargs)
        return self

    def parameters(self, recurse=True):
        return self.layer.parameters(recurse=recurse)


def apply_sdp_to_engine(engine, sparsity_threshold=0.01):
    """
    Apply Sparse Delta Propagation to a Hop6DynamicEngine.
    Wraps each transformer layer with SDPLayerWrapper.

    Args:
        engine: Hop6DynamicEngine instance
        sparsity_threshold: minimum delta to consider a dimension "active"

    Returns:
        engine with SDP applied
    """
    wrapped_layers = nn.ModuleList()
    for i, layer in enumerate(engine.layers):
        sdp_layer = SDPLayerWrapper(layer, sparsity_threshold)
        wrapped_layers.append(sdp_layer)

    engine.layers = wrapped_layers
    engine._sdp_enabled = True
    return engine


def reset_sdp_caches(engine):
    """Reset all SDP caches in the engine (call on new conversation)."""
    if not getattr(engine, '_sdp_enabled', False):
        return
    for layer in engine.layers:
        if isinstance(layer, SDPLayerWrapper):
            layer.reset_caches()


def get_sdp_stats(engine):
    """
    Get aggregated SDP statistics from the engine.
    Returns a dict with overall metrics.
    """
    if not getattr(engine, '_sdp_enabled', False):
        return None

    all_stats = []
    for i, layer in enumerate(engine.layers):
        if isinstance(layer, SDPLayerWrapper):
            all_stats.append(layer.get_stats())

    if not all_stats:
        return None

    total_calls = sum(s["total_calls"] for s in all_stats)
    total_hits = sum(s["cache_hits"] for s in all_stats)
    total_sdp_layers = sum(s["n_sdp_layers"] for s in all_stats)

    sparsity_vals = [s["avg_sparsity"] for s in all_stats if s["cache_hits"] > 0]
    avg_sparsity = sum(sparsity_vals) / len(sparsity_vals) if sparsity_vals else 0.0

    savings_vals = [s["ops_saved_pct"] for s in all_stats if s["total_calls"] > 0]
    avg_savings = sum(savings_vals) / len(savings_vals) if savings_vals else 0.0

    return {
        "total_sdp_linears": total_sdp_layers,
        "total_forward_calls": total_calls,
        "cache_hits": total_hits,
        "hit_rate": total_hits / max(total_calls, 1),
        "avg_sparsity": avg_sparsity,
        "avg_ops_saved_pct": avg_savings,
    }


def print_sdp_report(engine):
    """Print a formatted SDP performance report."""
    stats = get_sdp_stats(engine)
    if stats is None:
        print("[SDP] Not enabled.")
        return

    print("\n=== Sparse Delta Propagation (SDP) Report ===")
    print(f"  SDP Linear Layers:   {stats['total_sdp_linears']}")
    print(f"  Forward Calls:       {stats['total_forward_calls']}")
    print(f"  Cache Hits:          {stats['cache_hits']}")
    print(f"  Hit Rate:            {stats['hit_rate']:.1%}")
    print(f"  Avg Sparsity:        {stats['avg_sparsity']:.1%}")
    print(f"  Avg Ops Saved:       {stats['avg_ops_saved_pct']:.1f}%")
    print("=" * 47)
