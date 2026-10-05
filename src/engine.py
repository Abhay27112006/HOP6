"""
Hop6 Engine — Core for Hop6 Dynamic Routing Architecture.

Dynamic Routing: Keep all layers in CPU RAM, route at runtime with GPU VRAM paging.
Only 1 layer in GPU at a time.

Requires: CUDA or MPS GPU.
"""

import random
import networkx as nx
import torch
import torch.nn as nn
import os

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None
import device_utils


def build_layer_graph(num_layers, wormhole_pct=0.30, seed=None, fixed_edges=None):
    """Build layer connectivity graph with optional wormhole shortcuts.

    Args:
        fixed_edges: If provided, use these exact edges (for reproducibility
                     when loading a trained bridge at inference).
        seed: If provided, use for deterministic random wormhole placement.
    """
    G = nx.DiGraph()

    # If we have saved edges from a trained bridge, reconstruct the exact graph
    if fixed_edges is not None:
        for i in range(num_layers):
            G.add_node(i)
        for src, dst in fixed_edges:
            G.add_edge(src, dst)
        return G, list(fixed_edges)

    edges = []
    for i in range(num_layers):
        G.add_node(i)
        if i < num_layers - 1:
            G.add_edge(i, i + 1)
            edges.append((i, i + 1))
            
    # Small-World Topology: 6 Hubs (Worlds)
    num_worlds = 6
    cluster_size = max(1, num_layers // num_worlds)
    
    # Create inter-cluster "Hub" connections
    for w in range(num_worlds - 1):
        src_hub = w * cluster_size
        dst_hub = (w + 1) * cluster_size
        # Ensure we don't jump past the model's actual depth
        if dst_hub < num_layers and not G.has_edge(src_hub, dst_hub):
            G.add_edge(src_hub, dst_hub)
            edges.append((src_hub, dst_hub))
            
    # Connect the final hub directly to the final layer to guarantee path completion
    last_hub = (num_worlds - 1) * cluster_size
    if last_hub < num_layers - 1 and not G.has_edge(last_hub, num_layers - 1):
        G.add_edge(last_hub, num_layers - 1)
        edges.append((last_hub, num_layers - 1))
        
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
# Mode 2: Dynamic Routing with VRAM Paging
# ============================================================

class Hop6DynamicEngine(nn.Module):
    """
    Wraps any HF causal-LM. All layers stay on CPU.
    Each forward pass routes through max_hops layers via VRAM paging.
    """
    def __init__(self, model, wormhole_pct=0.30, max_hops=6, target_device=None,
                 use_bridge=False, use_router=False, use_sdp=False,
                 sdp_threshold=0.01, graph_seed=None, fixed_edges=None, prune_mode=False):
        super().__init__()
        self.target_device = target_device or device_utils.get_device()
        self.model = model
        self.max_hops = max_hops
        self.use_bridge = use_bridge
        self.prune_mode = prune_mode
        self.layers = self.model.model.layers
        self.original_num_layers = len(self.layers)
        self.num_layers = len(self.layers)

        if self.prune_mode:
            # Permanently extract the Small-World Hubs
            cluster_size = max(1, self.num_layers // self.max_hops)
            hub_indices = [w * cluster_size for w in range(self.max_hops)]
            # Ensure the final layer is the true final layer
            hub_indices[-1] = self.num_layers - 1
            
            pruned_layers = torch.nn.ModuleList()
            for i, idx in enumerate(hub_indices):
                layer = self.layers[idx].to(self.target_device)
                layer.original_layer_idx = idx
                layer.layer_idx = i  # Sequential mapping for KV cache
                pruned_layers.append(layer)
            
            # Replace the model's layers with the pruned subset
            self.model.model.layers = pruned_layers
            self.layers = pruned_layers
            self.num_layers = len(self.layers)
            
            # Load LoRA adapters if they exist
            lora_path = "./hop6_data/lora_pruned"
            if os.path.exists(lora_path) and PeftModel is not None:
                print(f"[Hop6] Loading LoRA weights from {lora_path} for Pruned Mode...")
                # We merge and unload to return the base model with merged weights
                # This keeps the class structure identical (Qwen2ForCausalLM)
                peft_model = PeftModel.from_pretrained(self.model, lora_path)
                self.model = peft_model.merge_and_unload()
                print("[Hop6] LoRA active and merged! Gibberish should be significantly reduced.")
            
            # In prune mode, the graph is purely sequential
            self.graph, self.edges = build_layer_graph(self.num_layers, wormhole_pct=0, seed=None, fixed_edges=None)
        else:
            for i, layer in enumerate(self.layers):
                layer.to("cpu")
                layer.layer_idx = i

            self.graph, self.edges = build_layer_graph(
                self.num_layers, wormhole_pct, seed=graph_seed, fixed_edges=fixed_edges
            )
        self._fixed_path = None  # Set externally to override random path selection

        hidden_dim = self.model.config.hidden_size

        # Bridge is optional (disabled by default - causes NaN in training)
        self.bridge = None
        if use_bridge:
            self.bridge = BridgeNetwork(hidden_dim, self.original_num_layers).to(self.target_device).to(model.dtype)

        # Router is optional — enables learned path selection via Dijkstra
        self.router = None
        if use_router:
            self.router = TokenRouter(hidden_dim, len(self.edges)).to(self.target_device).to(model.dtype)

        # SDP (Sparse Delta Propagation) — novel CPU matmul optimization
        self._sdp_enabled = False
        if use_sdp:
            from sdp_cpu import apply_sdp_to_engine
            apply_sdp_to_engine(self, sparsity_threshold=sdp_threshold)

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
        # Resolve original indices for the bridge if in prune mode
        true_layer_idx = getattr(self.layers[layer_idx], "original_layer_idx", layer_idx)
        true_prev_idx = None
        if prev_layer_idx is not None:
            true_prev_idx = getattr(self.layers[prev_layer_idx], "original_layer_idx", prev_layer_idx)

        if self.bridge and true_prev_idx is not None and (true_layer_idx - true_prev_idx) > 1:
            hidden_states = self.bridge(hidden_states, true_prev_idx, true_layer_idx)

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

        # Pick path: reset on new prefill (past_length == 0) or first call
        if past_length == 0 or self.current_path is None:
            path = self._pick_random_path(hidden_states)
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

            # KV Cache fix: temporarily mask absolute layer_idx with sequential
            # index so DynamicCache fills slots 0, 1, 2... instead of absolute
            # indices like 0, 5, 23 (which causes out-of-bounds corruption)
            original_idx = getattr(self.layers[layer_idx], "layer_idx", None)
            if is_cache_object and hasattr(self.layers[layer_idx], "layer_idx"):
                self.layers[layer_idx].layer_idx = i

            out = self._page_execute(
                layer_idx, hidden_states, prev_layer_idx=prev_idx,
                attention_mask=attention_mask, position_ids=position_ids,
                past_key_value=layer_past, use_cache=use_cache,
                **kwargs
            )

            # Restore original layer_idx
            if is_cache_object and original_idx is not None:
                self.layers[layer_idx].layer_idx = original_idx

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

    def _pick_random_path(self, hidden_states=None):
        """Pick a path. Uses fixed_path if set, router if available, otherwise random."""
        if self._fixed_path is not None:
            return list(self._fixed_path)
        if self.router is not None and hidden_states is not None:
            return self._pick_router_path(hidden_states)
        try:
            paths = list(nx.all_simple_paths(self.graph, source=0, target=self.num_layers - 1, cutoff=self.max_hops))
            if paths:
                return random.choice(paths)
        except Exception:
            pass
        # Fallback: sequential path
        return list(range(min(self.max_hops, self.num_layers))) + ([self.num_layers - 1] if self.max_hops < self.num_layers else [])

    def _pick_router_path(self, hidden_states):
        """Use the trained router to set edge costs and run Dijkstra."""
        import torch.nn.functional as F
        # Use the actual last token embedding to predict the best path for this prompt
        last_tok = hidden_states[:, -1, :].to(self.target_device).to(next(self.router.parameters()).dtype)
        with torch.no_grad():
            edge_logits = self.router(last_tok)
            edge_costs = F.softplus(edge_logits).clamp(min=1e-3, max=10.0)
        costs_np = edge_costs[0].cpu().numpy()
        for idx, (u, v) in enumerate(self.edges):
            self.graph[u][v]["weight"] = costs_np[idx].item()
        try:
            path = nx.shortest_path(self.graph, source=0,
                                    target=self.num_layers - 1, weight="weight")
            if len(path) > self.max_hops:
                path = path[:self.max_hops - 1] + [self.num_layers - 1]
            return path
        except nx.NetworkXNoPath:
            return [0, self.num_layers - 1]

    def _cleanup_vram(self):
        if self.current_path:
            for layer_idx in self.current_path:
                self.layers[layer_idx].to("cpu")
                self.layers[layer_idx]._is_on_gpu = False
            device_utils.empty_cache()

    def generate(self, *args, **kwargs):
        self.current_path = None
        # Reset SDP caches for new generation
        if self._sdp_enabled:
            from sdp_cpu import reset_sdp_caches
            reset_sdp_caches(self)
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

    def print_sdp_stats(self):
        """Print SDP performance report if SDP is enabled."""
        if self._sdp_enabled:
            from sdp_cpu import print_sdp_report
            print_sdp_report(self)


class TokenRouter(nn.Module):
    """Lightweight MLP that predicts edge costs for Dijkstra path selection."""
    def __init__(self, hidden_dim, num_edges):
        super().__init__()
        mid = hidden_dim // 4
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.GELU(),
            nn.Linear(mid, num_edges),
        )

    def forward(self, x):
        """x: (batch, hidden_dim) → (batch, num_edges) raw logits."""
        return self.net(x)


class BridgeNetwork(nn.Module):
    """Upscaled transformer-based bridge for wormhole jumps."""
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        layer_embed_dim = hidden_dim // 8
        self.layer_embed = nn.Embedding(num_layers, layer_embed_dim)
        input_dim = hidden_dim + 2 * layer_embed_dim
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Scale up to a full transformer block (2 layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=16,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self, h, src_idx, dst_idx):
        device = h.device
        batch, seq_len, _ = h.shape
        src_e = self.layer_embed(torch.tensor(src_idx, device=device))
        dst_e = self.layer_embed(torch.tensor(dst_idx, device=device))
        src_e = src_e.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1)
        dst_e = dst_e.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1)
        
        dtype = self.input_proj.weight.dtype
        x = torch.cat([h, src_e, dst_e], dim=-1).to(dtype)
        
        x = self.input_proj(x)
        delta = self.transformer(x)
        
        return h + (torch.sigmoid(self.gate) * delta).to(h.dtype)


