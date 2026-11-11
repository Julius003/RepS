from torch import torch
from transformer_lens.model_bridge import TransformerBridge

def main():
    models = ["Qwen/Qwen2.5-Coder-7B"]
    bridge = TransformerBridge.boot_transformers(models[0], device="cpu", dtype=torch.bfloat16)

    logits, activations = bridge.run_with_cache("Hi World")

if __name__ == "__main__":
    main()
