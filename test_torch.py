import torch

print("PyTorch Working")

print("Torch Version:", torch.__version__)

# Check GPU
if torch.cuda.is_available():
    print("GPU Available")
else:
    print("GPU Not Available")