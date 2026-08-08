#!/usr/bin/env python
"""
Hop6 CLI — Minimal working version.
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

CONFIG_FILE = "hop6_config.json"
MODELS_DIR = "./hop6_data/models"
HOP6_MODELS_DIR = "./hop6_data/hop6_models"
ROUTERS_DIR = "./hop6_data/routers"


def load_config():
    global MODELS_DIR, HOP6_MODELS_DIR, ROUTERS_DIR
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
        MODELS_DIR = cfg.get("MODELS_DIR", MODELS_DIR)
        HOP6_MODELS_DIR = cfg.get("HOP6_MODELS_DIR", HOP6_MODELS_DIR)
        ROUTERS_DIR = cfg.get("ROUTERS_DIR", ROUTERS_DIR)
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
        model_id, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    return mdl, tok


def load_dynamic(model_id, device, use_bridge=False):
    """Load with Hop6 dynamic routing (saves VRAM, needs trained router)."""
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True
    )
    engine = Hop6DynamicEngine(mdl, target_device=device, use_bridge=use_bridge)
    mdl.forward = engine.forward
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


def menu():
    load_config()
    device = get_device()
    print(f"\nHop6 Architecture — Device: {device}")
    print("=" * 50)

    while True:
        print("\n1. Chat with FULL model (best quality, most VRAM)")
        print("2. Chat with DYNAMIC routing (less VRAM, needs trained router)")
        print("3. Chat with EXTRACTED 6-layer model (lowest VRAM)")
        print("4. Extract a model to 6-layer Hop6")
        print("5. Download model from HuggingFace")
        print("6. Exit")

        choice = input("\nChoice: ").strip()

        if choice == "1":
            model_id = input("Model path or HF ID: ").strip()
            model, tok = load_full_model(model_id, device)
            chat_loop(model, tok)

        elif choice == "2":
            model_id = input("Model path or HF ID: ").strip()
            model, tok = load_dynamic(model_id, device, use_bridge=False)
            chat_loop(model, tok)

        elif choice == "3":
            models = [d for d in os.listdir(HOP6_MODELS_DIR)
                      if os.path.isdir(os.path.join(HOP6_MODELS_DIR, d))]
            if not models:
                print("No extracted models. Use option 4 first.")
                continue
            for i, m in enumerate(models):
                print(f"  {i+1}. {m}")
            idx = int(input("Select: ")) - 1
            model, tok = load_extracted(os.path.join(HOP6_MODELS_DIR, models[idx]), device)
            chat_loop(model, tok)

        elif choice == "4":
            model_id = input("Model path or HF ID to extract: ").strip()
            name = input("Save name (e.g. my_hop6): ").strip()
            save_dir = os.path.join(HOP6_MODELS_DIR, name)
            convert_to_hop6(model_id, save_dir, max_hops=6)
            print(f"Saved to {save_dir}")

        elif choice == "5":
            from huggingface_hub import snapshot_download
            repo = input("HF repo ID (e.g. Qwen/Qwen2.5-0.5B): ").strip()
            name = input("Folder name: ").strip()
            path = os.path.join(MODELS_DIR, name)
            os.makedirs(path, exist_ok=True)
            snapshot_download(repo_id=repo, local_dir=path, local_dir_use_symlinks=False)
            print(f"Downloaded to {path}")

        elif choice == "6":
            break


if __name__ == "__main__":
    menu()