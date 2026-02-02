import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}") # This will likely be False
if torch.cuda.is_available():
    print(f"Current Device: {torch.cuda.get_device_name(0)}")