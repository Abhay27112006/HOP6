import argparse
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from engine import Hop6DijkstraEngine
import device_utils

def main():
    parser = argparse.ArgumentParser(description="Test VRAM for ~14B/20B models")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-14B-Instruct", help="Model ID")
    parser.add_argument("--prompt", type=str, default="This is a test of the VRAM usage for the Hop6 architecture on a large model.", help="Test prompt")
    args = parser.parse_args()

    print(f"[VRAM Test] Target Model: {args.model}")
    print("[VRAM Test] Checking CUDA availability...")
    device = device_utils.get_device()
    if device not in ("cuda", "mps"):
        print("Error: CUDA/MPS is not available. Cannot perform VRAM test.")
        return
    
    device_utils.reset_vram_stats()
    
    print("\n[VRAM Test] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    print("[VRAM Test] Loading Base Model to CPU (System RAM)...")
    t0 = time.time()
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading model to CPU. Do you have enough System RAM? ({e})")
        return
    t1 = time.time()
    print(f"  -> Model loaded in {t1-t0:.1f}s")
    
    print("\n[VRAM Test] Wrapping in Hop6 Engine...")
    engine = Hop6DijkstraEngine(base_model, target_device=device)
    engine.eval()
    alloc, _ = device_utils.get_vram_stats()
    print(f"  -> VRAM after wrapping (Permanent VRAM): {alloc:.2f} GB")
    
    print("\n[VRAM Test] Running Single Forward Pass (Prefill)...")
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    
    # Reset peak stats right before the forward pass to get accurate paged activation metrics
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        t0 = time.time()
        # Ensure use_cache is enabled so we can see any overhead from cache tensors
        outputs = engine(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            use_cache=True
        )
        t1 = time.time()
        
    print(f"  -> Forward pass completed in {t1-t0:.2f}s")
    
    # Get empirical VRAM numbers
    peak_mb = device_utils.get_peak_vram_allocated_mb()
    max_allocated = peak_mb / 1024
    alloc, max_reserved = device_utils.get_vram_stats()
    
    print("\n=== VRAM Results ===")
    print("[Theoretical Estimate for Qwen2.5-14B (fp16)]")
    print("  Permanent (embed + lm_head + routers): ~3.15 GB")
    print("  1 Active Layer (GQA, intermediate=13824): ~0.55 GB")
    print("  CUDA Context & Activations: ~0.40 GB")
    print("  -> Expected Total: ~4.10 GB")
    
    print("\n[Empirical Results]")
    print(f"  Peak VRAM Allocated: {max_allocated:.2f} GB")
    print(f"  Peak VRAM Reserved:  {max_reserved:.2f} GB (what PyTorch actually holds)")
    
    if max_reserved > 5.5:
        print("\n[WARNING] Peak reserved memory is very close to the 6GB limit.")
        print("You may encounter OOM during longer generation runs due to fragmentation.")
    elif max_reserved < 5.0:
        print("\n[SUCCESS] Peak reserved memory is well within the 6GB limit.")
        print("Model fits with comfortable headroom on a 6GB card.")

if __name__ == "__main__":
    main()
