
import logging
import torch

def check_gpu():
    if torch.cuda.is_available():
        print(f"Cuda device available: {torch.cuda.get_device_name(0)}")

    else:
        print("NO CUDA DEVICE FOUND. USING CPU")