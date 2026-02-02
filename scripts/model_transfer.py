import torch
from transformers import AutoModel
import transformers
import sys

print(f"--- Environment Check ---")
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Device: {torch.cuda.get_device_name(0)}")

model_name = "microsoft/deberta-v3-base"
print(f"\n--- Loading Model: {model_name} ---")

# 1. Load to CPU (Simulate __init__)
try:
    model = AutoModel.from_pretrained(model_name, use_safetensors=True)
    print("Model loaded to CPU.")
except Exception as e:
    print(f"FAILED to load model: {e}")
    sys.exit(1)

# 2. Check CPU Weights
print("Checking CPU weights...")
if torch.isnan(model.embeddings.word_embeddings.weight).any():
    print("CRITICAL: Weights are NaN on CPU! (Bad download)")
    sys.exit(1)
else:
    print("CPU Weights: OK")

# 3. Move to GPU (The moment of failure)
print("\n--- Moving to GPU ---")
try:
    model.to("cuda")
    print("Move command executed.")
except Exception as e:
    print(f"CRASH during .to('cuda'): {e}")
    sys.exit(1)

# 4. Check GPU Weights
print("Checking GPU weights...")
gpu_weight = model.embeddings.word_embeddings.weight
if torch.isnan(gpu_weight).any():
    print("FAILURE: Weights corrupted during Transfer! (NaNs found)")
else:
    print("SUCCESS: Model is safe on GPU.")