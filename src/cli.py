"""
Hop6 CLI — Unified command-line interface.

Menu:
  1. Download a model from HuggingFace (+ auto-train Token Router)
  2. Load a model with Dijkstra Dynamic Routing (auto-loads trained router if available)
  3. Convert a model to static 6-hop extraction
  4. Load a pre-extracted Hop6 model
  5. Exit
"""

import os
import sys
import json
import torch
import time
import psutil
import questionary
import device_utils
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.align import Align
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from transformers import AutoModelForCausalLM, AutoTokenizer
from engine import (
    convert_to_hop6,
    load_hop6,
    Hop6DijkstraEngine
)
from train_router import train_router_for_model
from train_bridge import train_bridge_for_model

console = Console()
CONFIG_FILE = "hop6_config.json"
MODELS_DIR = ""
HOP6_MODELS_DIR = ""
ROUTERS_DIR = ""


def setup_hf_home():
    """Set HF_HOME so HuggingFace downloads go where the user expects."""
    config = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
        except Exception:
            pass

    if "hf_home" in config:
        os.environ["HF_HOME"] = config["hf_home"]
        return

    console.print("\n[bold yellow]Where would you like to store downloaded AI models?[/bold yellow]")
    console.print("[dim]This folder will hold large files (several GBs).[/dim]")
    default_path = os.path.expanduser("~/.cache/huggingface")
    path = questionary.path(
        "HuggingFace cache path:",
        default=default_path
    ).ask()

    if not path:
        path = default_path

    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    config["hf_home"] = path
    os.environ["HF_HOME"] = path

    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

    console.print(f"[bold green]HuggingFace cache set to: {path}[/bold green]")


def load_directories():
    global MODELS_DIR, HOP6_MODELS_DIR, ROUTERS_DIR
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
            MODELS_DIR = config.get("MODELS_DIR", "./hop6_data/models")
            HOP6_MODELS_DIR = config.get("HOP6_MODELS_DIR", "./hop6_data/hop6_models")
            ROUTERS_DIR = config.get("ROUTERS_DIR", "./hop6_data/routers")
            return
        except Exception:
            pass

    print_header("Hop6 Initial Setup")
    console.print("[bold yellow]Welcome! Before we start, let's configure storage.[/bold yellow]")
    base_dir = questionary.path(
        "Where do you want to store downloaded AI models and routers? (e.g. ./hop6_data):",
        default="./hop6_data"
    ).ask()
    
    if not base_dir:
        sys.exit(0)
        
    base_dir = os.path.abspath(base_dir)
    MODELS_DIR = os.path.join(base_dir, "models")
    HOP6_MODELS_DIR = os.path.join(base_dir, "hop6_models")
    ROUTERS_DIR = os.path.join(base_dir, "routers")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(HOP6_MODELS_DIR, exist_ok=True)
    os.makedirs(ROUTERS_DIR, exist_ok=True)
    
    with open(CONFIG_FILE, "w") as f:
        json.dump({
            "MODELS_DIR": MODELS_DIR,
            "HOP6_MODELS_DIR": HOP6_MODELS_DIR,
            "ROUTERS_DIR": ROUTERS_DIR
        }, f, indent=4)


def clear():
    os.system("cls" if os.name == "nt" else "clear")

def get_system_stats_panel():
    """Return a Panel with CPU, RAM, and GPU stats aligned to the right."""
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    ram_gb = ram.used / (1024**3)
    ram_tot = ram.total / (1024**3)
    device = device_utils.get_device()
    if device in ("cuda", "mps"):
        vram_alloc, vram_res = device_utils.get_vram_stats()
        backend = "CUDA" if device == "cuda" else "MPS"
        gpu_str = f"[cyan]{backend} VRAM:[/cyan] {vram_alloc:.1f}GB / {vram_res:.1f}GB"
    else:
        gpu_str = "[cyan]GPU:[/cyan] N/A"
        
    stats = f"[cyan]CPU:[/cyan] {cpu:5.1f}%   [cyan]RAM:[/cyan] {ram_gb:.1f}/{ram_tot:.1f}GB   {gpu_str}"
    return Panel(stats, title="System Monitor", border_style="cyan")

def print_header(title_text, subtitle_text=""):
    """Print a header with the title on the left and the system monitor on the right."""
    clear()
    table = Table.grid(expand=True)
    table.add_column(justify="left", ratio=1)
    table.add_column(justify="right", ratio=1)
    
    title_panel = Panel.fit(f"[bold green]{title_text}[/bold green]\n{subtitle_text}", border_style="green")
    monitor_panel = get_system_stats_panel()
    
    table.add_row(title_panel, monitor_panel)
    console.print(table)
    console.print("")


def parse_hf_id(text):
    """Accept either a full HuggingFace URL or a repo id."""
    return text.replace("https://huggingface.co/", "").strip("/")


def scan_folder(folder):
    """Return list of sub-directories inside *folder*."""
    if not os.path.isdir(folder):
        return []
    return sorted(d for d in os.listdir(folder)
                  if os.path.isdir(os.path.join(folder, d)))


def router_path_for_model(model_name):
    """Return the directory where the trained router for this model is stored."""
    safe_name = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    return os.path.join(ROUTERS_DIR, safe_name)


def has_trained_router(model_name):
    """Check if a trained router already exists for this model."""
    rdir = router_path_for_model(model_name)
    return os.path.isfile(os.path.join(rdir, "token_router.pt"))


def load_trained_router(engine, model_name):
    """Load saved router weights into the engine."""
    rdir = router_path_for_model(model_name)
    pt_path = os.path.join(rdir, "token_router.pt")
    if os.path.isfile(pt_path):
        engine.router.load_state_dict(torch.load(pt_path, map_location=engine.target_device))
        console.print(f"[bold green]Loaded trained router from {pt_path}[/bold green]")
        return True
    return False


def has_trained_bridge(model_name):
    """Check if a trained bridge already exists for this model."""
    rdir = router_path_for_model(model_name)
    return os.path.isfile(os.path.join(rdir, "bridge.pt"))


def load_trained_bridge(engine, model_name):
    """Load saved bridge weights into the engine."""
    rdir = router_path_for_model(model_name)
    pt_path = os.path.join(rdir, "bridge.pt")
    if os.path.isfile(pt_path):
        engine.bridge.load_state_dict(torch.load(pt_path, map_location=engine.target_device))
        console.print(f"[bold green]Loaded trained bridge from {pt_path}[/bold green]")
        return True
    return False


# ---- generation helper -----------------------------------------------

def generate(model, tokenizer, prompt, max_new_tokens=128):
    device = device_utils.get_device()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
        )
    t1 = time.perf_counter()

    text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    n_gen = outputs.shape[1] - input_len
    return {"text": text, "tok_sec": n_gen / max(t1 - t0, 1e-6)}


# ---- chat loop --------------------------------------------------------

def chat_loop(model, tokenizer):
    print_header("Hop6 Engine Ready", "Type [bold red]exit[/bold red] to quit.")
    messages = []

    while True:
        try:
            user = questionary.text("You:").ask()
        except (KeyboardInterrupt, EOFError):
            break
        if not user:
            continue
        if user.lower() in ("exit", "quit"):
            break

        messages.append({"role": "user", "content": user})
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = user

        device_utils.reset_vram_stats()

        console.print("[bold magenta]Hop6:[/bold magenta] ", end="")
        try:
            res = generate(model, tokenizer, prompt)
            out = res["text"].strip()
            peak = device_utils.get_peak_vram_allocated_mb()
            if peak > 0:
                console.print(
                    f"\n[dim italic]Speed: {res['tok_sec']:.1f} tok/s  |  "
                    f"Peak VRAM: {peak:.1f} MB[/dim italic]"
                )
            else:
                console.print(f"\n[dim italic]Speed: {res['tok_sec']:.1f} tok/s[/dim italic]")
            messages.append({"role": "assistant", "content": out})
        except Exception as e:
            console.print(f"\n[bold red]Error: {e}[/bold red]")
            messages.pop()


# ---- menu actions -----------------------------------------------------

def action_download():
    """Download a model AND automatically train the Token Router for it."""
    url = questionary.text(
        "Paste HuggingFace URL or Repo ID:"
    ).ask()
    if not url:
        return
    repo_id = parse_hf_id(url)
    folder = questionary.text(
        "Save folder name (e.g. qwen_0.5b):"
    ).ask()
    if not folder:
        return

    save_dir = os.path.join(MODELS_DIR, folder)

    # Step 1: Download
    console.print(f"\n[bold yellow]Step 1/2: Downloading {repo_id} → {save_dir}...[/bold yellow]")
    try:
        from huggingface_hub import snapshot_download
        os.makedirs(save_dir, exist_ok=True)
        snapshot_download(repo_id=repo_id, local_dir=save_dir, local_dir_use_symlinks=False)
        console.print(f"[bold green]Download complete![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Download failed: {e}[/bold red]")
        input("\nPress Enter to continue...")
        return

    # Step 2: Auto-train the Token Router
    console.print(f"\n[bold yellow]Step 2/3: Training Token Router for {repo_id}...[/bold yellow]")
    try:
        tokenizer = AutoTokenizer.from_pretrained(save_dir, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            save_dir, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True
        )
        dev = device_utils.get_device()
        
        engine = Hop6DijkstraEngine(base_model, target_device=dev)

        router_dir = router_path_for_model(folder)
        train_router_for_model(
            engine, tokenizer, router_dir, model_name=repo_id,
            epochs=3, lr=1e-3, print_fn=lambda msg: console.print(f"  [dim]{msg}[/dim]")
        )
        console.print(f"[bold green]Router trained and saved![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Router training failed: {e}[/bold red]")
        console.print("[italic]You can still use the model — the router will use random routing.[/italic]")
        
    # Step 3: Auto-train the Bridge Network
    console.print(f"\n[bold yellow]Step 3/3: Training Bridge Network for {repo_id}...[/bold yellow]")
    try:
        train_bridge_for_model(
            engine, tokenizer, router_dir, model_name=repo_id,
            epochs=5, lr=5e-4, print_fn=lambda msg: console.print(f"  [dim]{msg}[/dim]")
        )
        console.print(f"[bold green]Bridge trained and saved! Model is fully ready to use.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Bridge training failed: {e}[/bold red]")
        console.print("[italic]You can still use the model — the bridge will run as an identity function.[/italic]")

    input("\nPress Enter to continue...")


def action_dijkstra():
    """Load a model and wrap it with Dijkstra routing. Auto-loads trained router if available."""
    source = questionary.select(
        "Where is the model?",
        choices=[
            "a. Pick from downloaded models (scans all drives)",
            "b. Type a HuggingFace URL / Repo ID",
        ],
    ).ask()
    if not source:
        return None, None

    model_name = None

    if source.startswith("a"):
        all_models = []
        if os.path.isdir(MODELS_DIR):
            for m in scan_folder(MODELS_DIR):
                tag = " [trained ✓]" if has_trained_router(m) else ""
                all_models.append({
                    "display": f"{m}{tag}",
                    "name": m,
                    "path": os.path.join(MODELS_DIR, m),
                })

        if not all_models:
            console.print("[bold red]No models found in your configured models directory. Download one first![/bold red]")
            input("\nPress Enter...")
            return None, None

        choice = questionary.select(
            "Select model:",
            choices=[m["display"] for m in all_models],
        ).ask()
        if not choice:
            return None, None

        selected = next(m for m in all_models if m["display"] == choice)
        model_name = selected["name"]
        model_id = selected["path"]
    else:
        raw = questionary.text("HuggingFace URL or Repo ID:").ask()
        if not raw:
            return None, None
        model_id = parse_hf_id(raw)
        model_name = model_id.replace("/", "_")

    console.print(f"\n[bold yellow]Loading {model_id} into System RAM...[/bold yellow]")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True
        )
    except Exception as e:
        console.print(f"[bold red]Load failed: {e}[/bold red]")
        input("\nPress Enter...")
        return None, None

    dev = device_utils.get_device()
    console.print("[bold yellow]Applying Dijkstra Dynamic Router + VRAM Paging...[/bold yellow]")
    engine = Hop6DijkstraEngine(base, target_device=dev)

    # Auto-load trained router if available
    if model_name and has_trained_router(model_name):
        load_trained_router(engine, model_name)
    else:
        console.print("[yellow]No trained router found. Using random routing.[/yellow]")
        # Offer to train now
        should_train = questionary.confirm("Train the Token Router now? (~1 min)").ask()
        if should_train:
            router_dir = router_path_for_model(model_name)
            train_router_for_model(
                engine, tokenizer, router_dir, model_name=model_name,
                epochs=3, lr=1e-3, print_fn=lambda msg: console.print(f"  [dim]{msg}[/dim]")
            )

    # Auto-load trained bridge if available
    if model_name and has_trained_bridge(model_name):
        load_trained_bridge(engine, model_name)
    else:
        console.print("[yellow]No trained bridge found. Using identity bridge (no correction).[/yellow]")
        should_train = questionary.confirm("Train the Bridge Network now? (~5-15 min)").ask()
        if should_train:
            router_dir = router_path_for_model(model_name)
            train_bridge_for_model(
                engine, tokenizer, router_dir, model_name=model_name,
                epochs=5, lr=5e-4, print_fn=lambda msg: console.print(f"  [dim]{msg}[/dim]")
            )

    base.forward = engine.forward
    return base, tokenizer


def action_convert():
    raw = questionary.text(
        "HuggingFace URL, Repo ID, or local path to convert:"
    ).ask()
    if not raw:
        return
    model_id = parse_hf_id(raw)
    name = questionary.text("Save name (e.g. hop6_qwen_0.5B):").ask()
    if not name:
        return
    save_dir = os.path.join(HOP6_MODELS_DIR, name)
    console.print(f"\n[bold yellow]Converting {model_id} → Hop6...[/bold yellow]")
    try:
        convert_to_hop6(model_id, save_dir, max_hops=6)
        console.print(f"[bold green]Saved to {save_dir}[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Conversion failed: {e}[/bold red]")
    input("\nPress Enter to continue...")


def action_load_extracted():
    models = scan_folder(HOP6_MODELS_DIR)
    if not models:
        console.print("[bold red]No extracted Hop6 models found. Convert one first![/bold red]")
        input("\nPress Enter...")
        return None, None
    choice = questionary.select("Select a Hop6 model:", choices=models).ask()
    if not choice:
        return None, None
    path = os.path.join(HOP6_MODELS_DIR, choice)
    dev = device_utils.get_device()
    console.print(f"\n[bold yellow]Loading {choice} with Dynamic VRAM Paging...[/bold yellow]")
    try:
        model, tokenizer = load_hop6(path, device=dev)
        console.print("[bold green]Loaded![/bold green]")
        return model, tokenizer
    except Exception as e:
        console.print(f"[bold red]Failed: {e}[/bold red]")
        input("\nPress Enter...")
        return None, None


# ---- main -------------------------------------------------------------

MENU = [
    "1. Download a model + Auto-train Router & Bridge",
    "2. Load model with Dijkstra Dynamic Routing",
    "3. Convert a model to static Hop6 extraction",
    "4. Load a pre-extracted Hop6 model",
    "5. Exit",
]


def main():
    setup_hf_home()
    load_directories()
    print_header(
        "Hop6 Architecture", 
        "[italic]Small-World Graph Routing · Dynamic VRAM Paging · Dijkstra Depth Router[/italic]"
    )

    while True:
        action = questionary.select("What would you like to do?", choices=MENU).ask()
        if not action:
            sys.exit(0)

        if action.startswith("1"):
            action_download()
            clear()
            continue

        if action.startswith("2"):
            model, tokenizer = action_dijkstra()
            if model is not None:
                chat_loop(model, tokenizer)
            clear()
            continue

        if action.startswith("3"):
            action_convert()
            clear()
            continue

        if action.startswith("4"):
            model, tokenizer = action_load_extracted()
            if model is not None:
                chat_loop(model, tokenizer)
            clear()
            continue

        if action.startswith("5"):
            sys.exit(0)


if __name__ == "__main__":
    main()
