#!/usr/bin/env python
"""
Hop6 CLI — Single entry point for the Hop6 Architecture.

Modes:
  1. Chat with full model (best quality, most VRAM)
  2. Chat with dynamic routing (less VRAM, needs trained router)
  3. Chat with extracted 6-layer model (lowest VRAM)
  4. Extract a model to 6-layer Hop6
  5. Train Bridge Network
  6. Train Token Router
  7. Evaluate perplexity
  8. Download model from HuggingFace
"""
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import time
import device_utils
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from engine import Hop6DynamicEngine, convert_to_hop6, load_hop6

MODELS_DIR = "./hop6_data/models"
HOP6_MODELS_DIR = "./hop6_data/hop6_models"
ROUTERS_DIR = "./hop6_data/routers"


def load_config():
    global MODELS_DIR, HOP6_MODELS_DIR, ROUTERS_DIR
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(HOP6_MODELS_DIR, exist_ok=True)
    os.makedirs(ROUTERS_DIR, exist_ok=True)


def get_device():
    return device_utils.get_device()


def print_stats():
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    d = get_device()
    if d == "cuda":
        a, r = device_utils.get_vram_stats()
        gpu = f"GPU: {a:.1f}GB / {r:.1f}GB"
    else:
        gpu = "GPU: N/A"
    print(f"CPU: {cpu:.1f}%  RAM: {ram.used/1024**3:.1f}/{ram.total/1024**3:.1f}GB  {gpu}")


def load_full_model(model_id, device):
    """Load full model directly (best quality, most VRAM)."""
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    return mdl, tok


def load_dynamic(model_id, device, use_sdp=False):
    """Load with Hop6 dynamic routing (saves VRAM, needs trained router)."""
    save_name = model_id.replace("/", "_")
    save_dir = os.path.join(ROUTERS_DIR, save_name)
    bridge_path = os.path.join(save_dir, "bridge.pt")
    router_path = os.path.join(save_dir, "router.pt")
    meta_path = os.path.join(save_dir, "bridge_meta.json")

    has_bridge = os.path.exists(bridge_path)
    has_router = os.path.exists(router_path)

    # Load saved graph edges so inference uses the same topology as training
    fixed_edges = None
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if "graph_edges" in meta:
            fixed_edges = [tuple(e) for e in meta["graph_edges"]]
            print(f"[CLI] Loaded graph topology ({len(fixed_edges)} edges) from bridge_meta.json")

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    engine = Hop6DynamicEngine(
        mdl, target_device=device, use_bridge=has_bridge, use_router=has_router,
        use_sdp=use_sdp, fixed_edges=fixed_edges
    )
    
    if has_bridge:
        engine.bridge.load_state_dict(torch.load(bridge_path, map_location=device, weights_only=True))
        print(f"[CLI] Loaded trained Bridge Network from {bridge_path}")
    if has_router:
        engine.router.load_state_dict(torch.load(router_path, map_location=device, weights_only=True))
        print(f"[CLI] Loaded trained Token Router from {router_path}")

    mdl.forward = engine.forward
    mdl._hop6_engine = engine  # Keep reference for SDP stats
    return mdl, tok


def load_extracted(model_path, device):
    """Load pre-extracted Hop6 model (6 layers, lowest VRAM)."""
    return load_hop6(model_path, device)


def chat_loop(model, tokenizer):
    print("\n--- Chat (type 'exit' to quit) ---")
    print_stats()
    messages = []
    while True:
        try:
            user = input("\nYou: ")
        except (KeyboardInterrupt, EOFError):
            break
        if not user:
            continue
        if user.lower() in ("exit", "quit"):
            break

        messages.append({"role": "user", "content": user})
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = user

        dev = get_device()
        inputs = tokenizer(prompt, return_tensors="pt").to(dev)
        input_len = inputs.input_ids.shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        t1 = time.perf_counter()

        text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        text = text.encode('ascii', 'ignore').decode('ascii')
        n_gen = outputs.shape[1] - input_len
        print(f"Hop6: {text}")
        print(f"  {n_gen/(t1-t0):.1f} tok/s", end="")
        if get_device() == "cuda":
            a, r = device_utils.get_vram_stats()
            print(f"  VRAM: {a:.1f}GB", end="")
        print()
        messages.append({"role": "assistant", "content": text})

    # Print SDP stats if available
    if hasattr(model, '_hop6_engine'):
        model._hop6_engine.print_sdp_stats()


def do_train_bridge():
    """Train the Bridge Network for a model."""
    from train_bridge import train_bridge_for_model
    model_id = input("Model path or HF ID: ").strip()
    if not model_id:
        print("No model specified.")
        return
    epochs = input("Epochs [5]: ").strip()
    epochs = int(epochs) if epochs else 5

    device = get_device()
    save_name = model_id.replace("/", "_")
    save_dir = os.path.join(ROUTERS_DIR, save_name)

    # Reuse saved graph topology if retraining (ensures consistency)
    fixed_edges = None
    meta_path = os.path.join(save_dir, "bridge_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if "graph_edges" in meta:
            fixed_edges = [tuple(e) for e in meta["graph_edges"]]
            print(f"[CLI] Reusing saved graph topology ({len(fixed_edges)} edges)")

    print(f"\n[CLI] Loading {model_id} for bridge training...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    engine = Hop6DynamicEngine(mdl, target_device=device, use_bridge=True,
                               fixed_edges=fixed_edges)

    train_bridge_for_model(engine, tok, save_dir, model_name=model_id, epochs=epochs)
    print(f"[CLI] Bridge saved to {save_dir}")


def do_train_router():
    """Train the Token Router for a model."""
    from train_router import train_router_for_model
    model_id = input("Model path or HF ID: ").strip()
    if not model_id:
        print("No model specified.")
        return
    epochs = input("Epochs [3]: ").strip()
    epochs = int(epochs) if epochs else 3

    device = get_device()
    print(f"\n[CLI] Loading {model_id} for router training...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    engine = Hop6DynamicEngine(mdl, target_device=device, use_bridge=True, use_router=True)

    # Load bridge weights if available
    save_name = model_id.replace("/", "_")
    save_dir = os.path.join(ROUTERS_DIR, save_name)
    bridge_path = os.path.join(save_dir, "bridge.pt")
    if os.path.exists(bridge_path):
        engine.bridge.load_state_dict(torch.load(bridge_path, map_location=device))
        print(f"[CLI] Loaded existing bridge from {bridge_path}")
    else:
        print("[CLI] No trained bridge found — router will train without bridge corrections.")

    train_router_for_model(engine, tok, save_dir, model_name=model_id, epochs=epochs)
    print(f"[CLI] Router saved to {save_dir}")


def do_eval_perplexity():
    """Evaluate perplexity: baseline vs Hop6."""
    from evaluate_perplexity import evaluate_perplexity, DEFAULT_EVAL_TEXT, has_trained_bridge, load_trained_bridge
    model_id = input("Model path or HF ID: ").strip()
    if not model_id:
        print("No model specified.")
        return
    text_file = input("Text file for eval (Enter for default): ").strip()

    device = get_device()

    if text_file and os.path.exists(text_file):
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print("[Eval] Using default evaluation text.")
        text = DEFAULT_EVAL_TEXT

    print(f"\n[Eval] Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    base_model.eval()

    # 1. Baseline
    print("\n[Eval] 1/3: Baseline (Full Model)...")
    base_model.to(device)
    t0 = time.time()
    ppl_baseline = evaluate_perplexity(base_model, tokenizer, text, device=device)
    t1 = time.time()
    print(f"  -> Baseline PPL: {ppl_baseline:.2f} ({t1-t0:.1f}s)")
    base_model.to("cpu")

    # 2. Hop6 no bridge
    engine = Hop6DynamicEngine(base_model, target_device=device, use_bridge=True)
    engine.eval()
    with torch.no_grad():
        engine.bridge.gate.copy_(torch.tensor([-6.0]))
    print("\n[Eval] 2/3: Hop6 (NO Bridge)...")
    t0 = time.time()
    ppl_no_bridge = evaluate_perplexity(engine, tokenizer, text, device=device)
    t1 = time.time()
    print(f"  -> Hop6 NO Bridge PPL: {ppl_no_bridge:.2f} ({t1-t0:.1f}s)")

    # 3. Hop6 with bridge
    model_name = os.path.basename(os.path.normpath(model_id))
    if has_trained_bridge(model_name):
        load_trained_bridge(engine, model_name)
        print("\n[Eval] 3/3: Hop6 (WITH Trained Bridge)...")
    else:
        print("\n[Eval] 3/3: Hop6 (Bridge untrained)...")
    t0 = time.time()
    ppl_with_bridge = evaluate_perplexity(engine, tokenizer, text, device=device)
    t1 = time.time()
    print(f"  -> Hop6 Bridge PPL: {ppl_with_bridge:.2f} ({t1-t0:.1f}s)")

    print("\n=== Summary ===")
    print(f"Baseline PPL:       {ppl_baseline:.2f}")
    print(f"Hop6 NO Bridge PPL: {ppl_no_bridge:.2f}")
    print(f"Hop6 Bridge PPL:    {ppl_with_bridge:.2f}")


def do_benchmark():
    """Run VRAM/speed benchmark comparing Full vs Hop6 vs Hop6+SDP."""
    model_id = input("Model path or HF ID: ").strip()
    if not model_id:
        print("No model specified.")
        return
    skip_full = input("Skip full-model test? (y/N): ").strip().lower() == "y"
    max_tokens = input("Max tokens [32]: ").strip()
    max_tokens = int(max_tokens) if max_tokens else 32

    sys.argv = [
        "benchmark.py",
        "--model", model_id,
        "--max_tokens", str(max_tokens),
    ]
    if skip_full:
        sys.argv.append("--skip_full")

    from benchmark import main as bench_main
    bench_main()


def do_isolation_test():
    """Diagnostic: run Hop6 with bridge fully disabled to identify gibberish source.

    If output is gibberish even without bridge → problem is in routing/KV-cache.
    If output is rough but coherent → bridge training is the right fix.
    """
    model_id = input("Model path or HF ID: ").strip()
    if not model_id:
        print("No model specified.")
        return

    device = get_device()
    print(f"\n[Isolation] Loading {model_id} (bridge DISABLED)...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )

    # NO bridge, NO router — raw skip-routing only
    engine = Hop6DynamicEngine(mdl, target_device=device, use_bridge=False, use_router=False)

    # Pick and fix a deterministic path
    from engine import find_hop_path
    path = find_hop_path(engine.graph, 0, engine.num_layers - 1, engine.max_hops)
    engine._fixed_path = path

    wormholes = [(path[i], path[i+1]) for i in range(len(path)-1) if path[i+1]-path[i] > 1]
    skipped = engine.num_layers - len(path)

    mdl.forward = engine.forward
    mdl._hop6_engine = engine

    test_prompts = [
        "The capital of France is",
        "Hello, how are you doing today",
        "Machine learning is a subset of",
        "Once upon a time, in a kingdom far away",
    ]

    print(f"\n{'='*60}")
    print(f"ISOLATION TEST — Bridge DISABLED, raw skip-routing")
    print(f"Path: {path}")
    print(f"Wormhole jumps (uncorrected): {wormholes}")
    print(f"Layers skipped: {skipped}/{engine.num_layers}")
    print(f"{'='*60}")

    for prompt in test_prompts:
        inputs = tok(prompt, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            outputs = mdl.generate(
                **inputs, max_new_tokens=50,
                pad_token_id=tok.eos_token_id,
                do_sample=False,  # greedy for reproducibility
            )
        text = tok.decode(outputs[0][input_len:], skip_special_tokens=True)
        text = text.encode('ascii', 'ignore').decode('ascii')
        print(f"\n  Prompt: \"{prompt}\"")
        print(f"  Output: \"{text[:200]}\"")

    engine._cleanup_vram()

    print(f"\n{'='*60}")
    print(f"INTERPRETATION:")
    print(f"  Coherent → bridge training will help. Run option 6.")
    print(f"  Gibberish → routing/KV-cache bug. Bridge can't fix this.")
    print(f"{'='*60}")


def menu():
    load_config()
    device = get_device()
    print(f"\nHop6 Architecture — Device: {device}")
    print("=" * 50)

    while True:
        print("\n 1. Chat with FULL model (best quality, most VRAM)")
        print(" 2. Chat with DYNAMIC routing (less VRAM)")
        print(" 3. Chat with DYNAMIC + SDP (sparse delta, CPU-optimized)")
        print(" 4. Chat with EXTRACTED 6-layer model (lowest VRAM)")
        print(" 5. Extract a model to 6-layer Hop6")
        print(" 6. Train Bridge Network")
        print(" 7. Train Token Router")
        print(" 8. Evaluate Perplexity")
        print(" 9. Benchmark (VRAM + Speed comparison)")
        print("10. Download model from HuggingFace")
        print("11. Isolation Test (diagnose gibberish)")
        print(" 0. Exit")

        try:
            choice = input("\nChoice: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        try:
            if choice == "1":
                model_id = input("Model path or HF ID: ").strip()
                model, tok = load_full_model(model_id, device)
                chat_loop(model, tok)

            elif choice == "2":
                model_id = input("Model path or HF ID: ").strip()
                model, tok = load_dynamic(model_id, device)
                chat_loop(model, tok)

            elif choice == "3":
                model_id = input("Model path or HF ID: ").strip()
                print("[SDP] Loading with Sparse Delta Propagation...")
                model, tok = load_dynamic(model_id, device, use_sdp=True)
                chat_loop(model, tok)

            elif choice == "4":
                models = [d for d in os.listdir(HOP6_MODELS_DIR)
                          if os.path.isdir(os.path.join(HOP6_MODELS_DIR, d))]
                if not models:
                    print("No extracted models. Use option 5 first.")
                    continue
                for i, m in enumerate(models):
                    print(f"  {i+1}. {m}")
                idx = int(input("Select: ")) - 1
                model, tok = load_extracted(os.path.join(HOP6_MODELS_DIR, models[idx]), device)
                chat_loop(model, tok)

            elif choice == "5":
                model_id = input("Model path or HF ID to extract: ").strip()
                name = input("Save name (e.g. my_hop6): ").strip()
                if not name:
                    name = "default_extracted"
                save_dir = os.path.join(HOP6_MODELS_DIR, name)
                convert_to_hop6(model_id, save_dir, max_hops=6)
                print(f"Saved to {save_dir}")

            elif choice == "6":
                do_train_bridge()

            elif choice == "7":
                do_train_router()

            elif choice == "8":
                do_eval_perplexity()

            elif choice == "9":
                do_benchmark()

            elif choice == "10":
                from huggingface_hub import snapshot_download
                repo = input("HF repo ID (e.g. Qwen/Qwen2.5-0.5B): ").strip()
                name = input("Folder name: ").strip()
                path = os.path.join(MODELS_DIR, name)
                os.makedirs(path, exist_ok=True)
                snapshot_download(repo_id=repo, local_dir=path, local_dir_use_symlinks=False)
                print(f"Downloaded to {path}")

            elif choice == "11":
                do_isolation_test()

            elif choice == "0":
                break

        except KeyboardInterrupt:
            print("\n[Cancelled]")
        except Exception as e:
            print(f"\n[Error] {e}")


if __name__ == "__main__":
    menu()
