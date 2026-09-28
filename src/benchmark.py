"""
benchmark.py — Benchmark Hop6 VRAM savings and SDP performance.

Compares:
  1. Full model on GPU (baseline — may OOM)
  2. Hop6 with GPU paging (standard)
  3. Hop6 + SDP on CPU (novel sparse delta propagation)

Usage:
    python benchmark.py --model ./hop6_data/models/qwen3b
    python benchmark.py --model Qwen/Qwen2.5-3B-Instruct --quick
"""

import argparse
import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engine import Hop6DynamicEngine
import device_utils

PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a Python function that sorts a list using merge sort.",
    "What are the main differences between TCP and UDP protocols?",
]


def measure_vram():
    """Return current VRAM allocated in GB."""
    if device_utils.get_device() == "cuda":
        return torch.cuda.memory_allocated() / (1024 ** 3)
    return 0.0


def measure_peak_vram():
    """Return peak VRAM allocated in GB."""
    if device_utils.get_device() == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0


def generate_tokens(model, tokenizer, prompt, device, max_new_tokens=32):
    """Generate tokens and return (output_text, tok_per_sec, n_tokens)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
    t1 = time.perf_counter()

    n_gen = outputs.shape[1] - input_len
    tok_s = n_gen / max(t1 - t0, 0.001)
    text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return text, tok_s, n_gen


def benchmark_full_model(model_id, tokenizer, device, prompt, max_new_tokens):
    """Benchmark 1: Full model on GPU."""
    print("\n[Bench] 1/3: Full Model on GPU...")
    try:
        torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
        )
        model.eval()
        vram_loaded = measure_vram()
        text, tok_s, n_gen = generate_tokens(model, tokenizer, prompt, device, max_new_tokens)
        peak = measure_peak_vram()
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        return {
            "mode": "Full Model (GPU)",
            "vram_gb": peak,
            "tok_s": tok_s,
            "n_tokens": n_gen,
            "status": "OK",
            "text": text[:80],
        }
    except torch.cuda.OutOfMemoryError:
        if device == "cuda":
            torch.cuda.empty_cache()
        return {
            "mode": "Full Model (GPU)",
            "vram_gb": ">6.0",
            "tok_s": "OOM",
            "n_tokens": 0,
            "status": "OOM",
            "text": "Out of memory!",
        }
    except Exception as e:
        return {
            "mode": "Full Model (GPU)",
            "vram_gb": "?",
            "tok_s": "ERR",
            "n_tokens": 0,
            "status": f"Error: {e}",
            "text": str(e)[:80],
        }


def benchmark_hop6_gpu(model_id, tokenizer, device, prompt, max_new_tokens):
    """Benchmark 2: Hop6 with GPU paging."""
    print("[Bench] 2/3: Hop6 + GPU Paging...")
    try:
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        base = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
        )
        engine = Hop6DynamicEngine(base, target_device=device)
        base.forward = engine.forward
        base.eval()

        vram_loaded = measure_vram()
        text, tok_s, n_gen = generate_tokens(base, tokenizer, prompt, device, max_new_tokens)
        peak = measure_peak_vram()

        del engine, base
        if device == "cuda":
            torch.cuda.empty_cache()

        return {
            "mode": "Hop6 (GPU paging)",
            "vram_gb": f"{peak:.2f}",
            "tok_s": tok_s,
            "n_tokens": n_gen,
            "status": "OK",
            "text": text[:80],
        }
    except Exception as e:
        return {
            "mode": "Hop6 (GPU paging)",
            "vram_gb": "?",
            "tok_s": "ERR",
            "n_tokens": 0,
            "status": f"Error: {e}",
            "text": str(e)[:80],
        }


def benchmark_hop6_sdp(model_id, tokenizer, device, prompt, max_new_tokens):
    """Benchmark 3: Hop6 + SDP on CPU."""
    print("[Bench] 3/3: Hop6 + SDP (Sparse Delta Propagation)...")
    try:
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        base = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
        )
        engine = Hop6DynamicEngine(base, target_device=device, use_sdp=True)
        base.forward = engine.forward
        base.eval()

        vram_loaded = measure_vram()
        text, tok_s, n_gen = generate_tokens(base, tokenizer, prompt, device, max_new_tokens)
        peak = measure_peak_vram()

        # Get SDP stats
        from sdp_cpu import get_sdp_stats
        sdp_stats = get_sdp_stats(engine)

        del engine, base
        if device == "cuda":
            torch.cuda.empty_cache()

        return {
            "mode": "Hop6 + SDP (CPU)",
            "vram_gb": f"{peak:.2f}",
            "tok_s": tok_s,
            "n_tokens": n_gen,
            "status": "OK",
            "text": text[:80],
            "sdp_stats": sdp_stats,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "mode": "Hop6 + SDP (CPU)",
            "vram_gb": "?",
            "tok_s": "ERR",
            "n_tokens": 0,
            "status": f"Error: {e}",
            "text": str(e)[:80],
        }


def print_results(results, model_id):
    """Print benchmark results in a formatted table."""
    print(f"\n{'=' * 70}")
    print(f"  Hop6 Benchmark: {model_id}")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if torch.cuda.is_available() else "")
    print(f"{'=' * 70}")
    print(f"{'Mode':<25} {'VRAM (GB)':<12} {'tok/s':<10} {'Status':<10}")
    print(f"{'-' * 25} {'-' * 12} {'-' * 10} {'-' * 10}")

    for r in results:
        vram = r["vram_gb"] if isinstance(r["vram_gb"], str) else f"{r['vram_gb']:.2f}"
        tok_s = r["tok_s"] if isinstance(r["tok_s"], str) else f"{r['tok_s']:.1f}"
        status = "[OK]" if r["status"] == "OK" else "[FAIL] " + r["status"]
        print(f"{r['mode']:<25} {vram:<12} {tok_s:<10} {status}")

    # Print SDP details if available
    for r in results:
        if "sdp_stats" in r and r["sdp_stats"]:
            s = r["sdp_stats"]
            print(f"\n--- SDP Stats ---")
            print(f"  SDP Linear Layers: {s['total_sdp_linears']}")
            print(f"  Cache Hit Rate:    {s['hit_rate']:.1%}")
            print(f"  Avg Sparsity:      {s['avg_sparsity']:.1%}")
            print(f"  Ops Saved:         {s['avg_ops_saved_pct']:.1f}%")

    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Hop6 Architecture")
    parser.add_argument("--model", type=str, required=True, help="Model path or HF ID")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt")
    parser.add_argument("--max_tokens", type=int, default=32, help="Max new tokens")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer tokens)")
    parser.add_argument("--skip_full", action="store_true",
                        help="Skip full-model benchmark (if you know it will OOM)")
    args = parser.parse_args()

    if args.quick:
        args.max_tokens = min(args.max_tokens, 16)

    device = device_utils.get_device()
    prompt = args.prompt or PROMPTS[0]

    print(f"[Bench] Model: {args.model}")
    print(f"[Bench] Device: {device}")
    print(f"[Bench] Prompt: {prompt[:60]}...")
    print(f"[Bench] Max tokens: {args.max_tokens}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    results = []

    # 1. Full model (may OOM)
    if not args.skip_full:
        r = benchmark_full_model(args.model, tokenizer, device, prompt, args.max_tokens)
        results.append(r)
        print(f"  -> {r['status']}: VRAM={r['vram_gb']}, tok/s={r['tok_s']}")

    # 2. Hop6 GPU paging
    r = benchmark_hop6_gpu(args.model, tokenizer, device, prompt, args.max_tokens)
    results.append(r)
    print(f"  -> {r['status']}: VRAM={r['vram_gb']}, tok/s={r['tok_s']}")

    # 3. Hop6 + SDP
    r = benchmark_hop6_sdp(args.model, tokenizer, device, prompt, args.max_tokens)
    results.append(r)
    print(f"  -> {r['status']}: VRAM={r['vram_gb']}, tok/s={r['tok_s']}")

    # Print summary
    print_results(results, args.model)


if __name__ == "__main__":
    main()
