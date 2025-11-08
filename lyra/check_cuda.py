import torch
import sys

print("=" * 60)
print("CUDA/GPU Check")
print("=" * 60)
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
else:
    print("\n" + "!" * 60)
    print("WARNING: CUDA is NOT available!")
    print("!" * 60)
    print("\nYour PyTorch installation does not have CUDA support.")
    print("Current PyTorch build: CPU-only")
    print("\nTo fix this, you need to:")
    print("1. Uninstall current PyTorch:")
    print("   pip uninstall torch torchvision torchaudio")
    print("\n2. Install PyTorch with CUDA support:")
    print("   Visit: https://pytorch.org/get-started/locally/")
    print("   Or run one of these commands:")
    print("\n   For CUDA 11.8:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("\n   For CUDA 12.1:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("\n3. Verify installation by running this script again")
    
print("=" * 60)
