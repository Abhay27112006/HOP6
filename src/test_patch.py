import transformers
from transformers.models.qwen2 import modeling_qwen2
import traceback

orig = modeling_qwen2.apply_rotary_pos_emb
def monkey_apply(*args, **kwargs):
    try:
        return orig(*args, **kwargs)
    except Exception as e:
        print('================ CRASH SHAPES ================')
        for i, a in enumerate(args):
            print(f'arg {i}: {getattr(a, "shape", type(a))}')
        for k, v in kwargs.items():
            print(f'kwarg {k}: {getattr(v, "shape", type(v))}')
        print('================ TRACEBACK ================')
        traceback.print_stack()
        print('==============================================')
        raise e
modeling_qwen2.apply_rotary_pos_emb = monkey_apply

with open('evaluate_perplexity.py', 'r') as f:
    code = f.read()

exec(code)
