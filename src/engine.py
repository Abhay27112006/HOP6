"""
Hop6 Engine — Minimal core for Hop6 Architecture.

Two modes:
  1. Static Extraction: Extract N layers permanently (portable models).
  2. Dynamic Routing: Keep all layers in CPU RAM, route at runtime with VRAM paging.

Both use Dynamic VRAM Paging — only 1 layer in GPU at a time.
"""

import os
import json
import random
import networkx as nx
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import device_utils


def build_layer_graph(num_layers, wormhole_pct=0.30):
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
    try:
        paths = list(nx.all_simple_paths(G, source=source, target=target, cutoff=max_hops))
        if not paths:
            raise ValueError("No path found")
        paths.sort(key=lambda p: abs(len(p) - max_hops))
        return paths[0]
    except Exception:
        return torch.linspace(source, target, max_hops).long().tolist()


# ============================================================
# Mode 1: Static Extraction
# ============================================================

def convert_to_hop6(model_id, save_dir, max_hops=6, wormhole_pct=0.30):
    print(f"[Hop6] Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True
    )
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Model not supported. Must have model.model.layers.")

    num_layers = len(model.model.layers)
    print(f"[Hop6] Base model depth: {num_layers} layers.")

    G, _ = build_layer_graph(num_layers, wormhole_pct)
    path = find_hop_path(G, source=0, target=num_layers - 1, max_hops=max_hops)
    print(f"[Hop6] Path: {path} ({len(path)} layers)")

    model.model.layers = nn.ModuleList([model.model.layers[i] for i in path])
    model.config.num_hidden_layers = len(path)
    if hasattr(model.config, "layer_types") and isinstance(model.config.layer_types, list):
        model.config.layer_types = [model.config.layer_types[i] for i in path]

    os.makedirs(save_dir, exist_ok=True)
    print(f"[Hop6] Saving to {save_dir}...")
    try:
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    except Exception as e:
        print(f"[Hop6] HF save failed ({e}), fallback to torch.save...")
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
        "wormhole_pct": wormhole_pct,
    }
    with open(os.path.join(save_dir, "hop6_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[Hop6] Done!")


# ============================================================
# Mode 2: Dynamic Routing with VRAM Paging
# ============================================================

class Hop6DynamicEngine(nn.Module):
    """
    Wraps any HF causal-LM. All layers stay on CPU.
    Each forward pass routes through max_hops layers via VRAM paging.
    """
    def __init__(self, model, wormhole_pct=0.30, max_hops=6, target_device=None, use_bridge=False):
        super().__init__()
        self.target_device = target_device or device_utils.get_device()
        self.model = model
        self.max_hops = max_hops
        self.use_bridge = use_bridge
        self.layers = self.model.model.layers
        self.num_layers = len(self.layers)

        for i, layer in enumerate(self.layers):
            layer.to("cpu")
            layer.layer_idx = i

        self.graph, self.edges = build_layer_graph(self.num_layers, wormhole_pct)

        hidden_dim = self.model.config.hidden_size

        # Bridge is optional (disabled by default - causes NaN in training)
        self.bridge = None
        if use_bridge:
            self.bridge = BridgeNetwork(hidden_dim, self.num_layers).to(self.target_device).to(model.dtype)

        self.model.model.embed_tokens.to(self.target_device)
        if hasattr(self.model.model, "norm"):
            self.model.model.norm.to(self.target_device)
        if hasattr(self.model, "lm_head"):
            self.model.lm_head.to(self.target_device)

        self.config = self.model.config
        self.current_path = None

    def _page_execute(self, layer_idx, hidden_states, prev_layer_idx=None,
                      attention_mask=None, position_ids=None,
                      past_key_value=None, use_cache=False, **extra_kwargs):
        if self.bridge and prev_layer_idx is not None and (layer_idx - prev_layer_idx) > 1:
            hidden_states = self.bridge(hidden_states, prev_layer_idx, layer_idx)

        layer = self.layers[layer_idx]
        if getattr(layer, '_is_on_gpu', False) is False:
            layer.to(self.target_device)
            layer._is_on_gpu = True

        hidden_states = hidden_states.to(self.target_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.target_device)
        if position_ids is not None:
            position_ids = position_ids.to(self.target_device)
        if past_key_value is not None:
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

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, use_cache=None, **kwargs):
        if use_cache is None:
            use_cache = self.config.use_cache if hasattr(self.config, "use_cache") else False

        input_ids = input_ids.to(self.target_device)
        hidden_states = self.model.model.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            from transformers import DynamicCache
            past_key_values = DynamicCache(config=self.config)

        past_length = 0
        if past_key_values is not None:
            if not isinstance(past_key_values, tuple):
                past_length = past_key_values.get_seq_length()
            elif len(past_key_values) > 0 and past_key_values[0] is not None:
                past_length = past_key_values[0][0].shape[2]

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask,
                    (input_ids.shape[0], input_ids.shape[1]),
                    hidden_states,
                    past_length
                )
            attention_mask = attention_mask.to(self.target_device)

        if hasattr(self.model.model, "rotary_emb"):
            if position_ids is None:
                device = self.target_device
                seq_length = hidden_states.shape[1]
                position_ids = torch.arange(
                    past_length, seq_length + past_length, dtype=torch.long, device=device
                ).unsqueeze(0)
            kwargs["position_embeddings"] = self.model.model.rotary_emb(hidden_states, position_ids)

        # Pick path: random (no trained router) or fixed sequential
        if self.current_path is None:
            path = self._pick_random_path()
            self.current_path = path
        else:
            path = self.current_path

        prev_idx = None
        next_decoder_cache = () if use_cache else None
        is_cache_object = past_key_values is not None and not isinstance(past_key_values, tuple)

        for i, layer_idx in enumerate(path):
            layer_past = None
            if past_key_values is not None:
                if is_cache_object:
                    layer_past = past_key_values
                else:
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
                    next_decoder_cache = out[1] if len(out) > 1 else past_key_values
                else:
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

    def _pick_random_path(self):
        """Pick a random valid path from layer 0 to last within max_hops."""
        try:
            paths = list(nx.all_simple_paths(self.graph, source=0, target=self.num_layers - 1, cutoff=self.max_hops))
            if paths:
                return random.choice(paths)
        except Exception:
            pass
        # Fallback: sequential path
        return list(range(min(self.max_hops, self.num_layers))) + ([self.num_layers - 1] if self.max_hops < self.num_layers else [])

    def _cleanup_vram(self):
        if self.current_path:
            for layer_idx in self.current_path:
                self.layers[layer_idx].to("cpu")
                self.layers[layer_idx]._is_on_gpu = False
            if self.target_device != "cpu":
                device_utils.empty_cache()

    def generate(self, *args, **kwargs):
        self.current_path = None
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


class BridgeNetwork(nn.Module):
    """Optional bridge for wormhole jumps. Disabled by default."""
    def __init__(self, hidden_dim, num_layers, bottleneck_ratio=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        layer_embed_dim = hidden_dim // 8
        self.layer_embed = nn.Embedding(num_layers, layer_embed_dim)
        input_dim = hidden_dim + 2 * layer_embed_dim
        mid_dim = hidden_dim // bottleneck_ratio
        self.net = nn.Sequential(
            nn.Linear(input_dim, mid_dim), nn.GELU(),
            nn.Linear(mid_dim, mid_dim), nn.GELU(),
            nn.Linear(mid_dim, hidden_dim),
        )
        self.gate = nn.Parameter(torch.tensor([-6.0]))

    def forward(self, h, src_idx, dst_idx):
        device = h.device
        batch, seq_len, _ = h.shape
        src_e = self.layer_embed(torch.tensor(src_idx, device=device))
        dst_e = self.layer_embed(torch.tensor(dst_idx, device=device))
        src_e = src_e.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1)
        dst_e = dst_e.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1)
        dtype = self.net[0].weight.dtype
        x = torch.cat([h, src_e, dst_e], dim=-1).to(dtype)
        delta = self.net(x)
        return h + (torch.sigmoid(self.gate) * delta).to(h.dtype)


# ============================================================
# Paged loader (for extracted Hop6 models)
# ============================================================

class PagedLayer(nn.Module):
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