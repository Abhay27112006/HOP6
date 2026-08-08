import os

with open('cli.py', 'r') as f:
    code = f.read()

setup_code = '''
def setup_hf_home():
    config = load_config()
    if "hf_home" not in config:
        print("\\n[Setup] Where would you like to store the downloaded AI models?")
        print("This folder will hold large files (several GBs).")
        path = input("Enter full path (or press Enter for default ~/.cache/huggingface): ").strip()
        if not path:
            path = os.path.expanduser("~/.cache/huggingface")
        config["hf_home"] = os.path.abspath(path)
        save_config(config)
    os.environ["HF_HOME"] = config["hf_home"]
'''

code = code.replace('def init_bridge(model_name: str, hidden_dim: int, num_layers: int):', setup_code + '\ndef init_bridge(model_name: str, hidden_dim: int, num_layers: int):')

code = code.replace('def main():\n', 'def main():\n    setup_hf_home()\n')

with open('cli.py', 'w') as f:
    f.write(code)
