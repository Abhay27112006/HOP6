"""
train_bridge.py — Train the Hop6 Bridge Network.

The Bridge fixes the quality degradation caused by skipping layers.
When the router jumps from layer 1 → layer 23, layer 23 expects input
from layer 22.  The Bridge learns to transform layer 1's output into
something that approximates layer 22's output.

Training:
  Phase 1 — Calibration: Run full model forward pass on calibration data,
            capture hidden states at every layer.
  Phase 2 — Train Bridge: For each wormhole jump (src, dst), train the
            bridge to minimize MSE(bridge(h_src, src, dst), h_{dst-1}).

Usage:
    python train_bridge.py --model Qwen/Qwen1.5-0.5B-Chat --epochs 5

Can also be imported and called from the CLI:
    from train_bridge import train_bridge_for_model
    train_bridge_for_model(engine, tokenizer, save_dir)
"""

import os
import json
import argparse
import random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from engine import Hop6DijkstraEngine
import device_utils


# --------------- Calibration Data -------------------------------------
# Diverse sentences for capturing layer behaviors across different inputs.
# More data = better bridge generalization.
CALIBRATION_DATA = [
    # Factual knowledge
    "The capital of France is Paris, which is known for the Eiffel Tower.",
    "Machine learning is a subset of artificial intelligence that focuses on learning from data.",
    "Python is a popular programming language used for web development and data science.",
    "The sun rises in the east and sets in the west every single day.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "Albert Einstein developed the theory of relativity in the early 20th century.",
    "The human body contains 206 bones and approximately 600 muscles.",
    "JavaScript is the most widely used programming language for web browsers.",
    "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
    "DNA stands for deoxyribonucleic acid and carries genetic information.",
    "The Great Wall of China is one of the most impressive architectural feats in history.",
    "Neural networks are computing systems inspired by biological neural networks in the brain.",
    "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
    "The Pacific Ocean is the largest and deepest ocean on Earth.",
    "Shakespeare wrote many famous plays including Hamlet, Macbeth, and Romeo and Juliet.",
    "Gravity is the force that attracts objects toward the center of the Earth.",
    "The Internet was originally developed as a military communication network.",
    "Artificial intelligence can be categorized into narrow AI and general AI.",
    "The moon orbits the Earth approximately once every 27.3 days.",
    "Quantum computing uses quantum bits or qubits instead of classical binary bits.",
    # Conversational
    "Hello, how are you doing today? I hope you are having a wonderful day.",
    "Can you help me write an email to my colleague about the meeting tomorrow?",
    "The weather forecast says it will rain this afternoon, so bring an umbrella.",
    "I need to schedule a doctor's appointment for next Monday morning.",
    "Thank you for your help with the project. I really appreciate your effort.",
    "Could you please explain how transformers work in natural language processing?",
    "The stock market experienced significant volatility during the past trading week.",
    "Regular exercise and a balanced diet are essential for maintaining good health.",
    "The latest software update includes several bug fixes and performance improvements.",
    "Please review the attached document and provide your feedback by Friday.",
    # Reasoning / multi-step
    "If a train travels at 60 miles per hour for 3 hours, it covers 180 miles total.",
    "The Fibonacci sequence starts with 0, 1, 1, 2, 3, 5, 8, 13, 21, 34 and so on.",
    "To convert Celsius to Fahrenheit, multiply by nine fifths and add thirty two.",
    "In chess, the queen can move any number of squares in any direction on the board.",
    "A binary search algorithm divides the search space in half with each comparison.",
    "The area of a circle is calculated using pi multiplied by the radius squared.",
    "Recursion occurs when a function calls itself to solve smaller subproblems.",
    "The traveling salesman problem is a classic example of NP-hard optimization.",
    "Gradient descent is an optimization algorithm that iteratively moves toward the minimum.",
    "Hash tables provide average case O(1) time complexity for insertions and lookups.",
    # Creative / narrative
    "Once upon a time, in a kingdom far away, there lived a wise old wizard.",
    "The spaceship hurtled through the asteroid belt, narrowly dodging massive rocks.",
    "She opened the ancient book and discovered a map leading to hidden treasure.",
    "The detective examined the crime scene carefully, looking for any clue he could find.",
    "In the depths of the ocean, strange creatures glow with bioluminescent light.",
    "The robot looked at the sunset and wondered what it meant to be alive.",
    "A lone wolf howled at the full moon from the top of the mountain peak.",
    "The time traveler arrived in the year 3000 and found cities floating in the sky.",
    # Code-like / technical
    "To create a Python virtual environment, run python minus m venv followed by the name.",
    "The HTTP status code 404 means the requested resource was not found on the server.",
    "Docker containers provide isolated environments for running applications consistently.",
    "Git branches allow developers to work on features independently without conflicts.",
    "REST APIs use HTTP methods like GET, POST, PUT, and DELETE for resource operations.",
    "SQL joins combine rows from two or more tables based on a related column between them.",
    "Kubernetes orchestrates containerized applications across a cluster of machines.",
    "The TCP three way handshake consists of SYN, SYN-ACK, and ACK packets.",
    # Long-form reasoning
    "Climate change is driven by greenhouse gas emissions from burning fossil fuels, "
    "deforestation, and industrial processes, leading to rising global temperatures.",
    "The human immune system consists of innate and adaptive components that work "
    "together to identify and eliminate pathogens like bacteria and viruses.",
    "Deep learning models with millions of parameters can overfit on small datasets, "
    "which is why techniques like dropout, data augmentation, and regularization exist.",
    "The process of photosynthesis involves light dependent reactions in the thylakoid "
    "membranes and the Calvin cycle in the stroma of chloroplasts.",
]


# ---- Phase 1: Calibration (capture layer outputs) --------------------

def calibrate_layer_outputs(engine, tokenizer, texts, max_length=128,
                            print_fn=None):
    """
    Run ALL layers sequentially on calibration texts.
    Returns a list of dicts, one per text:
        [{"input_ids": tensor, "layer_outputs": {layer_idx: tensor}}, ...]

    All outputs are kept on CPU to save VRAM.
    """
    if print_fn is None:
        print_fn = print

    device = engine.target_device
    results = []

    print_fn(f"[Bridge] Calibrating on {len(texts)} samples ({engine.num_layers} layers each)...")

    for text_idx, text in enumerate(texts):
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = tokens.input_ids.to(device)

        with torch.no_grad():
            # Get embeddings
            h = engine.model.model.embed_tokens(input_ids)
            layer_outputs = {}
            
            kwargs = {}
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)
            kwargs["position_ids"] = position_ids
            
            # Precompute rotary embeddings ONCE (matching engine.py approach)
            if hasattr(engine.model.model, "rotary_emb"):
                # rotary_emb expects (hidden_states, position_ids) but only uses position_ids for shape
                # Pass dummy hidden_states with correct batch/seq shape
                kwargs["position_embeddings"] = engine.model.model.rotary_emb(h, position_ids)

            # Run ALL layers sequentially (full model forward pass)
            for layer_idx in range(engine.num_layers):
                layer = engine.layers[layer_idx]
                layer.to(device)

                if device != "cpu":
                    device_utils.synchronize()

                out = layer(h, use_cache=False, **kwargs)
                h = out[0] if isinstance(out, tuple) else out

                if device != "cpu":
                    device_utils.synchronize()

                layer.to("cpu")

                if device != "cpu":
                    device_utils.empty_cache()

                # Save layer output on CPU
                layer_outputs[layer_idx] = h.detach().cpu()

            results.append({
                "input_ids": input_ids.cpu(),
                "layer_outputs": layer_outputs,
            })

        if (text_idx + 1) % 10 == 0:
            print_fn(f"  Calibrated {text_idx + 1}/{len(texts)} samples...")

    print_fn(f"[Bridge] Calibration complete. Captured {len(results)} samples × {engine.num_layers} layers.")
    return results


# ---- Phase 2: Train Bridge ------------------------------------------

def train_bridge_from_calibration(engine, calibration_results, save_dir,
                                  model_name="unknown", epochs=5, lr=5e-4,
                                  print_fn=None):
    """
    Train the BridgeNetwork using captured calibration data.

    For each wormhole edge (src, dst) where dst - src > 1:
        bridge(h_src, src, dst) should approximate h_{dst-1}
    """
    if print_fn is None:
        print_fn = print

    device = engine.target_device

    # Freeze everything except the bridge
    for param in engine.model.parameters():
        param.requires_grad = False
    for param in engine.router.parameters():
        param.requires_grad = False
    for param in engine.bridge.parameters():
        param.requires_grad = True

    # Use parameter groups: lower LR for gate to prevent instability
    gate_params = [engine.bridge.gate]
    other_params = [p for n, p in engine.bridge.named_parameters() if n != 'gate']
    
    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': lr},
        {'params': gate_params, 'lr': lr * 0.01},  # Much lower LR for gate
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_params = sum(p.numel() for p in engine.bridge.parameters())
    print_fn(f"[Bridge] Training Bridge Network ({n_params/1e3:.1f}K params) for {epochs} epochs...")

    # Collect all wormhole edges (gaps > 1)
    wormhole_edges = [(src, dst) for (src, dst) in engine.edges if (dst - src) > 1]
    print_fn(f"[Bridge] {len(wormhole_edges)} wormhole edges to train on.")

    if not wormhole_edges:
        print_fn("[Bridge] No wormhole edges found. Nothing to train.")
        return engine

    for epoch in range(epochs):
        total_loss = 0.0
        count = 0

        # Shuffle calibration data each epoch
        indices = list(range(len(calibration_results)))
        random.shuffle(indices)

        for sample_idx in indices:
            sample = calibration_results[sample_idx]
            layer_outputs = sample["layer_outputs"]

            # Shuffle edges within each sample for better training
            edges_shuffled = wormhole_edges.copy()
            random.shuffle(edges_shuffled)

            # Accumulate gradients across all wormhole edges for this sample
            optimizer.zero_grad()
            sample_loss = 0.0
            sample_count = 0

            for (src, dst) in edges_shuffled:
                # Source: output of layer `src`
                h_src = layer_outputs[src].to(device)

                # Target: output of layer `dst - 1` (what layer `dst` expects)
                h_target = layer_outputs[dst - 1].to(device)

                # Bridge prediction
                h_pred = engine.bridge(h_src, src, dst)

                # MSE loss between predicted and actual (compute in float32 for stability)
                loss = F.mse_loss(h_pred.float(), h_target.float())
                
                # Debug: check for inf/nan
                if loss.isinf() or loss.isnan():
                    print(f"  [DEBUG] Sample {sample_idx} edge ({src},{dst}): loss={loss.item()}, h_pred range=[{h_pred.min().item():.2f},{h_pred.max().item():.2f}], h_target range=[{h_target.min().item():.2f},{h_target.max().item():.2f}]")
                    continue

                loss.backward()
                sample_loss += loss.item()
                sample_count += 1

            # Clip gradients and update once per sample
            torch.nn.utils.clip_grad_norm_(engine.bridge.parameters(), 1.0)
            optimizer.step()
            
            # Clamp gate to prevent numerical instability
            with torch.no_grad():
                engine.bridge.gate.clamp_(-10.0, 10.0)

            total_loss += sample_loss
            count += sample_count
        
        # Debug: check parameters after epoch
        gate_val = torch.sigmoid(engine.bridge.gate).item()
        print(f"  [DEBUG] After epoch {epoch+1}: gate={gate_val:.6f}, gate_raw={engine.bridge.gate.item():.6f}")

        scheduler.step()

        avg_loss = total_loss / max(count, 1)
        gate_val = torch.sigmoid(engine.bridge.gate).item()
        print_fn(f"  === Epoch {epoch+1}/{epochs} | Avg MSE: {avg_loss:.6f} | "
                 f"Gate: {gate_val:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} ===")

    # Save bridge weights
    os.makedirs(save_dir, exist_ok=True)
    bridge_path = os.path.join(save_dir, "bridge.pt")
    torch.save(engine.bridge.state_dict(), bridge_path)

    meta = {
        "architecture": "Hop6_Bridge",
        "original_model": model_name,
        "bridge_params": n_params,
        "epochs": epochs,
        "lr": lr,
        "calibration_samples": len(calibration_results),
        "wormhole_edges_trained": len(wormhole_edges),
        "final_gate": torch.sigmoid(engine.bridge.gate).item(),
    }
    with open(os.path.join(save_dir, "bridge_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print_fn(f"[Bridge] Saved to {bridge_path} ({os.path.getsize(bridge_path)/1024:.1f} KB)")
    print_fn(f"[Bridge] Gate value: {torch.sigmoid(engine.bridge.gate).item():.4f} "
             f"(0=identity, 1=full correction)")
    return engine


# ---- Public API (called from CLI) ------------------------------------

def train_bridge_for_model(engine, tokenizer, save_dir, model_name="unknown",
                           epochs=5, lr=5e-4, print_fn=None):
    """
    Full bridge training pipeline: calibrate + train.
    """
    if print_fn is None:
        print_fn = print

    # Phase 1: Calibrate
    print_fn(f"\n[Bridge] === Phase 1/2: Calibrating (full model forward pass) ===")
    calibration = calibrate_layer_outputs(
        engine, tokenizer, CALIBRATION_DATA, print_fn=print_fn
    )

    # Phase 2: Train
    print_fn(f"\n[Bridge] === Phase 2/2: Training Bridge Network ===")
    engine = train_bridge_from_calibration(
        engine, calibration, save_dir, model_name=model_name,
        epochs=epochs, lr=lr, print_fn=print_fn
    )

    return engine


# ---- Standalone entry point ------------------------------------------

def train_standalone(model_id, epochs=5, lr=5e-4, save_dir=None):
    device = device_utils.get_device()

    print(f"[Bridge] Loading {model_id} into CPU RAM...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True
    )

    print("[Bridge] Wrapping with Hop6 Dijkstra Engine...")
    engine = Hop6DijkstraEngine(base_model, target_device=device)

    if save_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)
        save_dir = os.path.join(root_dir, "routers", model_id.replace("/", "_"))

    train_bridge_for_model(engine, tokenizer, save_dir, model_name=model_id,
                           epochs=epochs, lr=lr)
    print("[Bridge] Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Hop6 Bridge Network")
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-0.5B-Chat",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs (default: 5)")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate (default: 5e-4)")
    parser.add_argument("--save", type=str, default=None,
                        help="Directory to save bridge weights")
    args = parser.parse_args()

    train_standalone(args.model, epochs=args.epochs, lr=args.lr, save_dir=args.save)
