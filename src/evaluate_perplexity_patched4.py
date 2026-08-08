
import transformers
from transformers.models.qwen2 import modeling_qwen2
orig_forward = modeling_qwen2.Qwen2DecoderLayer.forward
def monkey_forward(self, hidden_states, *args, **kwargs):
    print(f'---- LAYER {self.self_attn.layer_idx} ----')
    print(f'hidden_states: {hidden_states.shape}')
    return orig_forward(self, hidden_states, *args, **kwargs)
modeling_qwen2.Qwen2DecoderLayer.forward = monkey_forward

import argparse
import torch
import math
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from engine import Hop6DijkstraEngine

def evaluate_perplexity(model, tokenizer, text, max_length=512, stride=256, device="cuda"):
    """
    Evaluates perplexity using a sliding window approach.
    """
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids)
            # HF outputs usually have loss, but engine forward only returns logits right now
            # So we calculate CrossEntropy manually
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            neg_log_likelihood = loss * trg_len

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Perplexity")
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-0.5B", help="Model ID")
    parser.add_argument("--text_file", type=str, default=None, help="Path to text file for eval")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        # Fallback to a hardcoded evaluation chunk if no file provided
        print("[Eval] No text file provided, using default evaluation chunk.")
        text = (
            "The history of artificial intelligence (AI) began in antiquity, with myths, stories and rumors of "
            "artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI "
            "were planted by classical philosophers who attempted to describe the process of human thinking as the "
            "mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer "
            "in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it "
            "inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain. "
            "The field of AI research was founded at a workshop held on the campus of Dartmouth College during the summer of 1956. "
            "Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as "
            "intelligent as a human being would exist in no more than a generation, and they were given millions of dollars to "
            "make this vision come true. Eventually, it became obvious that they had grossly underestimated the difficulty of the project. "
            "In 1973, in response to the criticism of James Lighthill and ongoing pressure from the US Congress to fund more productive "
            "projects, both the US and British governments stopped funding undirected research into artificial intelligence. "
            "Seven years later, a visionary initiative by the Japanese Government inspired governments and industry to provide AI with "
            "billions of dollars, but by the late 80s the investors became disillusioned and withdrew funding again. This cycle of boom "
            "and bust, known as the 'AI winter', would repeat itself several times over the following decades. "
            "In the 21st century, AI techniques have experienced a resurgence following concurrent advances in computer power, large "
            "amounts of data, and theoretical understanding; and AI techniques have become an essential part of the technology industry, "
            "helping to solve many challenging problems in computer science, software engineering and operations research."
        ) * 5  # Multiply to make it longer

    print(f"[Eval] Loading Base Model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True
    )
    base_model.eval()

    # 1. Baseline Full Model
    print("\n[Eval] 1/3: Running Baseline (Full Model)...")
    base_model.to(device)
    t0 = time.time()
    ppl_baseline = evaluate_perplexity(base_model, tokenizer, text, device=device)
    t1 = time.time()
    print(f"  -> Baseline Perplexity: {ppl_baseline:.2f} (Took {t1-t0:.2f}s)")
    base_model.to("cpu")

    # Wrap in Engine
    engine = Hop6DijkstraEngine(base_model, target_device=device)
    engine.eval()
    
    # 2. Hop6 NO Bridge
    # We temporarily set the bridge gate to a large negative number so sigmoid(gate) ≈ 0 (identity)
    print("\n[Eval] 2/3: Running Hop6 (NO Bridge - identity)...")
    with torch.no_grad():
        engine.bridge.gate.copy_(torch.tensor([-10.0]))
    t0 = time.time()
    ppl_no_bridge = evaluate_perplexity(engine, tokenizer, text, device=device)
    t1 = time.time()
    print(f"  -> Hop6 NO Bridge PPL: {ppl_no_bridge:.2f} (Took {t1-t0:.2f}s)")

    # 3. Hop6 WITH Bridge
    # Load trained bridge if available
    import os
    from cli import has_trained_bridge, load_trained_bridge
    # Try to load real bridge weights if they exist in C:\6hops\routers
    model_name = args.model.replace("/", "_").replace("\\", "_").replace(":", "_")
    if has_trained_bridge(model_name):
        load_trained_bridge(engine, model_name)
        print("\n[Eval] 3/3: Running Hop6 (WITH Trained Bridge)...")
    else:
        # Simulate a trained bridge by just opening the gate slightly (or we can just run as is)
        print("\n[Eval] 3/3: Running Hop6 (Bridge initialized but untrained)...")
        with torch.no_grad():
            engine.bridge.gate.copy_(torch.tensor([0.0])) # 0.5 sigmoid

    t0 = time.time()
    ppl_with_bridge = evaluate_perplexity(engine, tokenizer, text, device=device)
    t1 = time.time()
    print(f"  -> Hop6 WITH Bridge PPL: {ppl_with_bridge:.2f} (Took {t1-t0:.2f}s)")

    print("\n=== Summary ===")
    print(f"Baseline PPL:       {ppl_baseline:.2f}")
    print(f"Hop6 NO Bridge PPL: {ppl_no_bridge:.2f}")
    print(f"Hop6 Bridge PPL:    {ppl_with_bridge:.2f}")

if __name__ == "__main__":
    main()
