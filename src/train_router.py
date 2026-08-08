"""
train_router.py — Train the Hop6 Token Router using REINFORCE.

Problem:  Dijkstra is a discrete algorithm — you can't backpropagate through it.
Solution: Treat the Token Router as a policy network.
          Use the language-modeling loss as a reward signal (REINFORCE).
          Only the tiny Router (~5 MB) is trained. All transformer layers + bridge stay frozen.

The BridgeNetwork is active during router training so the router learns to
pick paths that work well WITH the bridge corrections.

Usage:
    python train_router.py --model Qwen/Qwen1.5-0.5B-Chat --epochs 3

Can also be imported and called from the CLI:
    from train_router import train_router_for_model
    train_router_for_model(engine, tokenizer, save_dir)
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
import networkx as nx
from transformers import AutoModelForCausalLM, AutoTokenizer
from engine import Hop6DijkstraEngine
import device_utils

# --------------- Training Data ----------------------------------------
TRAINING_DATA = [
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
]


def compute_reinforce_loss(engine, tokenizer, text, device, baseline_ema):
    """
    One REINFORCE training step with stability fixes.
    Returns (loss_tensor, updated_baseline_ema).
    """
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = tokens.input_ids.to(device)

    if input_ids.shape[1] < 2:
        return None, baseline_ema

    # Embeddings (fp16 from the model)
    hidden_states = engine.model.model.embed_tokens(input_ids)

    kwargs = {}
    position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)
    kwargs["position_ids"] = position_ids
    if hasattr(engine.model.model, "rotary_emb"):
        kwargs["position_embeddings"] = engine.model.model.rotary_emb(hidden_states, position_ids)

    # Router prediction — cast to float32 for training stability
    last_tok = hidden_states[:, -1, :].float()
    edge_logits = engine.router(last_tok)  # (1, num_edges) - RAW LOGITS for REINFORCE
    edge_costs = F.softplus(edge_logits)    # Positive costs for Dijkstra
    
    # CLAMP: Prevent extreme costs that cause NaN gradients
    edge_costs = torch.clamp(edge_costs, min=1e-3, max=10.0)

    # Run Dijkstra (non-differentiable) using COSTS
    costs_np = edge_costs[0].detach().cpu().numpy()
    for idx, (u, v) in enumerate(engine.edges):
        engine.graph[u][v]["weight"] = costs_np[idx].item()

    try:
        path = nx.shortest_path(engine.graph, source=0,
                                target=engine.num_layers - 1, weight="weight")
        if len(path) > engine.max_hops:
            path = path[:engine.max_hops - 1] + [engine.num_layers - 1]
    except nx.NetworkXNoPath:
        path = [0, engine.num_layers - 1]

    # Execute the chosen path (layers are frozen, stay fp16)
    # Track prev_layer_idx so the BridgeNetwork activates on wormhole jumps
    h = hidden_states
    prev_idx = None
    for layer_idx in path:
        out = engine._page_execute(layer_idx, h, prev_layer_idx=prev_idx, **kwargs)
        h = out[0] if isinstance(out, tuple) else out
        prev_idx = layer_idx

    if hasattr(engine.model.model, "norm"):
        h = engine.model.model.norm(h)

    logits = engine.model.lm_head(h).float()  # cast to float32 for loss

    # Next-token prediction loss
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    lm_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                              shift_labels.view(-1))

    # REINFORCE reward (negative loss = higher reward for lower perplexity)
    reward = -lm_loss.detach()
    if baseline_ema is None:
        baseline_ema = reward.item()
    else:
        baseline_ema = 0.99 * baseline_ema + 0.01 * reward.item()  # Slower EMA

    advantage = reward - baseline_ema
    
    # NORMALIZE: Advantage normalization for stability
    advantage = advantage / (torch.abs(advantage) + 1e-8)

    # Policy gradient via logits along chosen path
    # Use log_softmax for numerically stable log-probabilities
    path_edges = list(zip(path[:-1], path[1:]))
    log_probs = F.log_softmax(edge_logits[0], dim=0)  # (num_edges,)
    log_prob = torch.tensor(0.0, device=device)
    for (u, v) in path_edges:
        if (u, v) in engine.edges:
            edge_idx = engine.edges.index((u, v))
            log_prob = log_prob + log_probs[edge_idx]

    reinforce_loss = -log_prob * advantage
    
    # ENTROPY: Add entropy bonus to prevent premature convergence
    # Entropy over all edges (not just path) for better exploration
    all_probs = F.softmax(edge_logits[0], dim=0)
    entropy = -(all_probs * torch.log(all_probs + 1e-8)).sum()
    entropy_bonus = 0.01 * entropy

    total_loss = reinforce_loss + 0.1 * lm_loss - entropy_bonus

    return total_loss, baseline_ema


# ---- Public API (called from CLI) ------------------------------------

def train_router_for_model(engine, tokenizer, save_dir, model_name="unknown",
                           epochs=3, lr=5e-4, print_fn=None):
    """
    Train the Token Router on the given engine.  Saves router weights
    to *save_dir*/token_router.pt.  Returns the trained engine.

    The BridgeNetwork is active (but frozen) during router training so the
    router learns to pick paths that work well with bridge corrections.
    """
    if print_fn is None:
        print_fn = print

    device = engine.target_device

    # Freeze transformer + bridge, unfreeze only router
    for param in engine.model.parameters():
        param.requires_grad = False
    for param in engine.bridge.parameters():
        param.requires_grad = False
    for param in engine.router.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(engine.router.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    baseline_ema = None

    n_router = sum(p.numel() for p in engine.router.parameters())
    n_bridge = sum(p.numel() for p in engine.bridge.parameters())
    print_fn(f"[Train] Training Token Router ({n_router/1e3:.1f}K params) for {epochs} epochs...")
    print_fn(f"[Train] Bridge ({n_bridge/1e3:.1f}K params) is active but frozen.")

    for epoch in range(epochs):
        total_loss = 0.0
        count = 0

        for i, text in enumerate(TRAINING_DATA):
            optimizer.zero_grad()
            loss, baseline_ema = compute_reinforce_loss(engine, tokenizer, text, device, baseline_ema)

            if loss is None:
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(engine.router.parameters(), 0.5)  # Stricter clipping
            optimizer.step()

            total_loss += loss.item()
            count += 1

            if (i + 1) % 10 == 0:
                avg = total_loss / max(count, 1)
                print_fn(f"  Epoch {epoch+1}/{epochs} | Step {i+1}/{len(TRAINING_DATA)} | Loss: {avg:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()
        avg = total_loss / max(count, 1)
        print_fn(f"  === Epoch {epoch+1} Complete | Avg Loss: {avg:.4f} ===")

    # Save router
    os.makedirs(save_dir, exist_ok=True)
    router_path = os.path.join(save_dir, "token_router.pt")
    torch.save(engine.router.state_dict(), router_path)

    meta = {
        "architecture": "Hop6_Dijkstra_v2",
        "original_model": model_name,
        "router_params": n_router,
        "bridge_params": n_bridge,
        "epochs": epochs,
        "lr": lr,
        "training_samples": len(TRAINING_DATA),
        "bridge_active_during_training": True,
    }
    with open(os.path.join(save_dir, "router_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print_fn(f"[Train] Router saved to {router_path} ({os.path.getsize(router_path)/1024:.1f} KB)")
    return engine


# ---- Standalone entry point ------------------------------------------

def train_standalone(model_id, epochs=3, lr=5e-4, save_dir=None):
    device = device_utils.get_device()

    print(f"[Train] Loading {model_id} into CPU RAM...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True
    )

    print("[Train] Wrapping with Hop6 Dijkstra Engine...")
    engine = Hop6DijkstraEngine(base_model, target_device=device)

    if save_dir is None:
        # Use a path relative to the project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)
        save_dir = os.path.join(root_dir, "routers", model_id.replace("/", "_"))

    train_router_for_model(engine, tokenizer, save_dir, model_name=model_id,
                           epochs=epochs, lr=lr)
    print("[Train] Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Hop6 Token Router")
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-0.5B-Chat",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    train_standalone(args.model, epochs=args.epochs, lr=args.lr, save_dir=args.save)
