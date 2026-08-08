import sys
import os
import torch
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from cli import *
import device_utils
from train_router import train_router_for_model
from engine import Hop6DijkstraEngine
from transformers import AutoModelForCausalLM, AutoTokenizer

def test():
    model_name = "qwen0.5b"
    save_dir = os.path.join("hop6_data", "models", model_name)
    print("Loading model from", save_dir)
    tokenizer = AutoTokenizer.from_pretrained(save_dir, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        save_dir, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True
    )
    dev = device_utils.get_device()
    engine = Hop6DijkstraEngine(base, target_device=dev)
    
    print("Training router...")
    try:
        train_router_for_model(
            engine, tokenizer, "temp_router", model_name=model_name,
            epochs=1, lr=1e-3, print_fn=print
        )
    except Exception as e:
        traceback.print_exc()

test()
