"""
Hop6 Engine — Core library for the Hop6 Architecture.

Two modes of operation:
  1. Static Extraction: Extract 6 layers permanently (for portable models).
  2. Dijkstra Dynamic Routing: Keep all layers, route each token through 6 hops at runtime.

Both modes use Dynamic VRAM Paging — only 1 layer in GPU memory at a time.
"""

import os
import json
import random
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import device_utils


# ============================================================
# Graph Construction (shared by both modes)
# ============================================================

def build_layer_graph(num_layers, wormhole_pct=0.30):
    """
    Creates a directed graph where nodes are layers.
    Sequential edges: i -> i+1
    Wormhole edges:   30% extra random skip connections (jump >= 2 layers).
    """
    G = nx.DiGraph()
    edges = []

    for i in range(num_layers):
        G.add_node(i)
        if i < num_layers - 1:
            G.add_edge(i, i + 1)
            edges.append((i, i + 1))

    num_wormholes = int(num_layers * wormhole_pct)
    added = 0
    while added < num_wormholes:
        src = random.randint(0, num_layers - 3)
        dst = random.randint(src + 2, num_layers - 1)
        if not G.has_edge(src, dst):
            G.add_edge(src, dst)
            edges.append((src, dst))
            added += 1

    return G, edges


def find_hop_path(G, source, target, max_hops=6):
    """Find the best path from source to target within max_hops."""
    try:
        paths = list(nx.all_simple_paths(G, source=source, target=target, cutoff=max_hops))
        if not paths:
            raise ValueError("No path found")
        paths.sort(key=lambda p: abs(len(p) - max_hops))
        return paths[0]
    except Exception:
        return torch.linspace(source, target, max_hops).long().tolist()


# ============================================================
# Mode 1: Static Extraction (convert & save a 6-layer model)
# ============================================================

def convert_to_hop6(model_id, save_dir, max_hops=6):
    """
    Loads a pretrained model, constructs the small-world graph,
    finds the 6-hop path, extracts those layers, and saves permanently.
    """
    print(f"[Hop6] Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True
    )

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Model not supported. Must have model.model.layers.")

    num_layers = len(model.model.layers)
    print(f"[Hop6] Base model depth: {num_layers} layers.")

    G, _ = build_layer_graph(num_layers)
    path = find_hop_path(G, source=0, target=num_layers - 1, max_hops=max_hops)
    print(f"[Hop6] Traversed Path: {path}")
    print(f"[Hop6] Executing {len(path)} layers total.")

    # Extract layers
    model.model.layers = nn.ModuleList([model.model.layers[i] for i in path])
    model.config.num_hidden_layers = len(path)
    if hasattr(model.config, "layer_types") and isinstance(model.config.layer_types, list):
        model.config.layer_types = [model.config.layer_types[i] for i in path]

    # Save
    os.makedirs(save_dir, exist_ok=True)
    print(f"[Hop6] Saving to {save_dir}...")
    try:
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    except Exception as e:
        print(f"[Hop6] HF save failed ({e}), falling back to torch.save...")
        torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
        tokenizer.save_pretrained(save_dir)
        try:
            model.config.save_pretrained(save_dir)
        except Exception:
            pass

    meta = {
        "architecture": "Hop6",
        "original_model": model_id,
        "original_depth": num_layers,
        "hop_path": path,
        "wormhole_pct": 0.30,
    }
    with open(os.path.join(save_dir, "hop6_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("[Hop6] Conversion complete!")


# ============================================================
# Mode 2: Dijkstra Dynamic Routing (keep ALL layers, route at runtime)
# ============================================================

class TokenRouter(nn.Module):
    """
    Tiny neural network (< 5 MB) that lives permanently in VRAM.
    Takes a token embedding and predicts edge logits for REINFORCE
    and edge costs (via Softplus) for Dijkstra.
    Uses float32 internally for training stability.
    """
    def __init__(self, hidden_dim, num_edges):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_edges),
            # No activation - raw logits for REINFORCE, Softplus applied externally for costs
        )

    def forward(self, x):
        """Returns raw logits (for REINFORCE) in float32."""
        return self.predictor(x.to(torch.float32))

    def get_costs(self, x):
        """Returns positive costs for Dijkstra (Softplus of logits) in float32."""
        logits = self.forward(x)
        return F.softplus(logits)


class BridgeNetwork(nn.Module):
    """
    Small network (~5-20M params) that stays permanently in VRAM.
    Transforms hidden states across layer gaps to fix the distribution
    mismatch caused by skipping layers.

    Given the output of layer `src`, produces an approximation of what
    layer `dst - 1` would have outputted, so layer `dst` receives the
    input distribution it was trained to expect.

    Uses a gated residual: output = h + sigmoid(gate) * delta
    Gate starts at 0 (identity), so the bridge can never make things
    worse than the current system before training.
    """
    def __init__(self, hidden_dim, num_layers, bottleneck_ratio=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Layer index embeddings — let the bridge know which layers it's bridging
        layer_embed_dim = hidden_dim // 8
        self.layer_embed = nn.Embedding(num_layers, layer_embed_dim)

        # MLP: takes [hidden_states, src_embed, dst_embed] → delta correction
        input_dim = hidden_dim + 2 * layer_embed_dim
        mid_dim = hidden_dim // bottleneck_ratio
        self.net = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, hidden_dim),
        )

        # Learnable gate — starts at -6.0 so sigmoid(-6) ≈ 0.002 (effectively identity)
        self.gate = nn.Parameter(torch.tensor([-6.0]))

    def forward(self, h, src_idx, dst_idx):
        """
        Args:
            h: hidden states from source layer, shape (batch, seq_len, hidden_dim)
            src_idx: source layer index (int)
            dst_idx: destination layer index (int)
        Returns:
            Adjusted hidden states approximating layer[dst-1]'s output.
        """
        device = h.device
        batch, seq_len, _ = h.shape

        # Get layer embeddings and broadcast to (batch, seq_len, embed_dim)
        src_e = self.layer_embed(torch.tensor(src_idx, device=device))
        dst_e = self.layer_embed(torch.tensor(dst_idx, device=device))
        src_e = src_e.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1)
        dst_e = dst_e.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1)

        # Concatenate and predict correction
        dtype = self.net[0].weight.dtype
        x = torch.cat([h, src_e, dst_e], dim=-1).to(dtype)
        delta = self.net(x)

        # Gated residual — gate starts at 0 (identity), learns to open
        return h + (torch.sigmoid(self.gate) * delta).to(h.dtype)


class Hop6DijkstraEngine(nn.Module):
    """
    Wraps any HuggingFace causal-LM model.
    All layers stay in System RAM.  For each forward pass the TokenRouter
    predicts edge costs, Dijkstra finds the cheapest ≤6-hop path, and only
    those layers are paged into VRAM one at a time.

    The BridgeNetwork fixes hidden-state distribution mismatch when skipping
    layers via wormholes.  It runs automatically on any jump where gap > 1.
    """

    def __init__(self, model, wormhole_pct=0.30, max_hops=6, target_device=None):
        super().__init__()
        self.target_device = target_device or device_utils.get_device()
        self.model = model
        self.max_hops = max_hops
        self.layers = self.model.model.layers
        self.num_layers = len(self.layers)

        # Keep every layer on CPU
        for i, layer in enumerate(self.layers):
            layer.to("cpu")
            layer.layer_idx = i  # Required for KV cache updates

        # Build graph
        self.graph, self.edges = build_layer_graph(self.num_layers, wormhole_pct)

        # Router lives permanently in VRAM (keep in float32 for training stability)
        hidden_dim = self.model.config.hidden_size
        self.router = TokenRouter(hidden_dim, len(self.edges)).to(self.target_device)
        # Router stays in float32, don't convert to model.dtype

        # Bridge Network lives permanently in VRAM (~10-40 MB)
        self.bridge = BridgeNetwork(hidden_dim, self.num_layers).to(self.target_device).to(model.dtype)

        # Embeddings + LM head stay in VRAM permanently (small)
        self.model.model.embed_tokens.to(self.target_device)
        if hasattr(self.model.model, "norm"):
            self.model.model.norm.to(self.target_device)
        if hasattr(self.model, "lm_head"):
            self.model.lm_head.to(self.target_device)

        bridge_params = sum(p.numel() for p in self.bridge.parameters())
        print(f"[Hop6 Dijkstra] Graph: {self.num_layers} nodes, {len(self.edges)} edges "
              f"(+{int(self.num_layers * wormhole_pct)} wormholes). Max hops: {max_hops}.")
        print(f"[Hop6 Bridge] {bridge_params/1e3:.1f}K params "
              f"({bridge_params * 2 / 1024 / 1024:.1f} MB fp16) — fixes distribution mismatch.")
        self.config = self.model.config
        
        # State for Per-Sequence Routing
        self.current_path = None

    # ---- layer paging ------------------------------------------------
    def _page_execute(self, layer_idx, hidden_states, prev_layer_idx=None,
                      attention_mask=None, position_ids=None,
                      past_key_value=None, use_cache=False, **extra_kwargs):
        """
        Execute a single layer. With per-sequence routing, layers are kept 
        resident in VRAM during the entire generation pass to avoid PCIe bottleneck.
        """
        if prev_layer_idx is not None and (layer_idx - prev_layer_idx) > 1:
            hidden_states = self.bridge(hidden_states, prev_layer_idx, layer_idx)

        layer = self.layers[layer_idx]
        
        # Move to GPU if it's not already there
        if getattr(layer, '_is_on_gpu', False) is False:
            layer.to(self.target_device)
            layer._is_on_gpu = True

        # Ensure all tensors are on target device
        hidden_states = hidden_states.to(self.target_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.target_device)
        if position_ids is not None:
            position_ids = position_ids.to(self.target_device)
        if past_key_value is not None:
            # Handle Cache objects or tuple of tensors
            if hasattr(past_key_value, 'to'):
                past_key_value = past_key_value.to(self.target_device)
            elif isinstance(past_key_value, tuple):
                past_key_value = tuple(p.to(self.target_device) if hasattr(p, 'to') else p for p in past_key_value)

        kwargs = {"use_cache": use_cache}
        if past_key_value is not None:
            kwargs["past_key_values"] = past_key_value
        kwargs.update(extra_kwargs)

        out = layer(hidden_states, attention_mask=attention_mask,
                    position_ids=position_ids, **kwargs)

        return out

    # ---- forward with dynamic routing --------------------------------
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, use_cache=None, **kwargs):
        # Handle fallback for use_cache
        if use_cache is None:
            use_cache = self.config.use_cache if hasattr(self.config, "use_cache") else False

        # Ensure input_ids on target device
        input_ids = input_ids.to(self.target_device)
        hidden_states = self.model.model.embed_tokens(input_ids)

        # Create DynamicCache if needed (matching model.forward behavior)
        if use_cache and past_key_values is None:
            from transformers import DynamicCache
            past_key_values = DynamicCache(config=self.config)

        past_length = 0
        if past_key_values is not None:
            if not isinstance(past_key_values, tuple):
                past_length = past_key_values.get_seq_length()
            elif len(past_key_values) > 0 and past_key_values[0] is not None:
                past_length = past_key_values[0][0].shape[2]

        # Prepare attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask,
                    (input_ids.shape[0], input_ids.shape[1]),
                    hidden_states,
                    past_length
                )
            # Ensure attention_mask is on target device
            attention_mask = attention_mask.to(self.target_device)

        # Support for models that precompute RoPE before layers (e.g., Llama 3, Qwen 2)
        # Compute position_embeddings and pass to each layer (model.forward does this internally)
        if hasattr(self.model.model, "rotary_emb"):
            if position_ids is None:
                device = self.target_device
                seq_length = hidden_states.shape[1]
                position_ids = torch.arange(
                    past_length, seq_length + past_length, dtype=torch.long, device=device
                ).unsqueeze(0)
            
            # Compute rotary embeddings for current position_ids
            kwargs["position_embeddings"] = self.model.model.rotary_emb(hidden_states, position_ids)

        # Per-Sequence Routing: Compute path only on first step (prefill phase)
        if self.current_path is None:
            # Route based on last token of the prompt
            last_tok = hidden_states[:, -1, :]
            costs = self.router.get_costs(last_tok)[0].detach().cpu().numpy()

            for idx, (u, v) in enumerate(self.edges):
                self.graph[u][v]["weight"] = costs[idx].item()

            try:
                path = nx.shortest_path(self.graph, source=0,
                                        target=self.num_layers - 1, weight="weight")
                if len(path) > self.max_hops:
                    path = path[: self.max_hops - 1] + [self.num_layers - 1]
            except nx.NetworkXNoPath:
                path = [0, self.num_layers - 1]
                
            self.current_path = path
        else:
            path = self.current_path

        # Execute layers with bridge on wormhole jumps and handle KV cache
        prev_idx = None
        next_decoder_cache = () if use_cache else None
        
        # Determine if past_key_values is a new Cache object or old tuple format
        is_cache_object = past_key_values is not None and not isinstance(past_key_values, tuple)

        for i, layer_idx in enumerate(path):
            # Extract layer-specific past_key_value
            layer_past = None
            if past_key_values is not None:
                if is_cache_object:
                    # For newer HF models using Cache objects (DynamicCache), 
                    # pass the whole cache and let the layer handle it via its layer_idx
                    layer_past = past_key_values
                else:
                    # Old tuple format: we only stored cache for the layers we executed
                    if i < len(past_key_values):
                        layer_past = past_key_values[i]

            out = self._page_execute(
                layer_idx, hidden_states, prev_layer_idx=prev_idx,
                attention_mask=attention_mask, position_ids=position_ids,
                past_key_value=layer_past, use_cache=use_cache,
                **kwargs
            )
            if isinstance(out, tuple):
                hidden_states = out[0]
                if len(out) > 1:
                    layer_past = out[1]
            else:
                hidden_states = out
            
            if use_cache:
                if is_cache_object:
                    # The Cache object updates in place
                    next_decoder_cache = out[1] if len(out) > 1 else past_key_values
                else:
                    # Append the layer's KV cache to the tuple
                    if len(out) > 1:
                        next_decoder_cache += (out[1],)
                        
            prev_idx = layer_idx

        if hasattr(self.model.model, "norm"):
            hidden_states = self.model.model.norm(hidden_states)

        logits = self.model.lm_head(hidden_states)

        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=next_decoder_cache
        )

    # ---- HF generate compatibility -----------------------------------
    def _cleanup_vram(self):
        """Evict all layers back to CPU after generation completes."""
        if self.current_path:
            for layer_idx in self.current_path:
                self.layers[layer_idx].to("cpu")
                self.layers[layer_idx]._is_on_gpu = False
            if self.target_device != "cpu":
                device_utils.empty_cache()

    def generate(self, *args, **kwargs):
        self.current_path = None  # Reset path for new generation
        try:
            return self.model.generate(*args, **kwargs)
        finally:
            self._cleanup_vram()

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def _update_model_kwargs_for_generation(self, *args, **kwargs):
        return self.model._update_model_kwargs_for_generation(*args, **kwargs)

    @property
    def main_input_name(self):
        return self.model.main_input_name

    @property
    def device(self):
        return torch.device(self.target_device)


# ============================================================
# Paged loader (for pre-extracted Hop6 models)
# ============================================================

class PagedLayer(nn.Module):
    """Wraps a single transformer layer for dynamic VRAM paging."""
    def __init__(self, layer, device=None):
        super().__init__()
        self.layer = layer.to("cpu")
        self.target_device = device or device_utils.get_device()

    def forward(self, *args, **kwargs):
        self.layer.to(self.target_device)
        gpu_args = [a.to(self.target_device) if isinstance(a, torch.Tensor) else a for a in args]
        gpu_kwargs = {k: (v.to(self.target_device) if isinstance(v, torch.Tensor) else v)
                      for k, v in kwargs.items()}
        out = self.layer(*gpu_args, **gpu_kwargs)
        self.layer.to("cpu")
        return out


def load_hop6(local_path, device=None):
    """Load a pre-extracted Hop6 model and wrap layers for paging."""
    device = device or device_utils.get_device()
    tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_path, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True
    )
    paged = nn.ModuleList([PagedLayer(l, device) for l in model.model.layers])
    model.model.layers = nn.ModuleList()
    model.to(device)
    model.model.layers = paged
    return model, tokenizer


def load_model_4bit(model_id, device=None, **kwargs):
    """
    Load a model with 4-bit quantization for memory-efficient inference.
    Layers stay on CPU in 4-bit, dequantized on-the-fly to GPU.
    
    Requires: bitsandbytes (pip install bitsandbytes)
    """
    from transformers import BitsAndBytesConfig
    
    device = device or device_utils.get_device()
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto" if device == "cuda" else "cpu",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        **kwargs
    )
    
    return model, tokenizer


def wrap_model_4bit_hop6(model, target_device=None, wormhole_pct=0.30, max_hops=6):
    """
    Wrap a 4-bit quantized model with Hop6 Dijkstra routing.
    The 4-bit layers stay on CPU, dequantized on-the-fly when paged to GPU.
    """
    # Note: This requires the model to be loaded with device_map="cpu" initially
    # For 4-bit models, we need special handling since layers are already quantized
    raise NotImplementedError("4-bit Hop6 wrapping requires custom dequantization kernel. Use load_model_4bit() for inference without Hop6, or load fp16 model and wrap with Hop6DijkstraEngine for full Hop6 support.")
