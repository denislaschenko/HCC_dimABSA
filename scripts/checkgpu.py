import torch
import sys

print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Test simple transfer
try:
    x = torch.ones(1000, 1000)
    print("Created Tensor on CPU.")
    x_gpu = x.to("cuda")
    print("Moved to GPU.")

    if torch.isnan(x_gpu).any():
        print("FAILURE: Tensor corrupted (NaN) on GPU!")
    else:
        print("SUCCESS: Simple tensors work.")

except Exception as e:
    print(f"CRASH: {e}")