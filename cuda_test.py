import torch

# GPUが利用可能かを確認
if torch.cuda.is_available():
    print(f"GPU利用可能: {torch.cuda.get_device_name(0)}")
    print(f"利用可能なGPUの数: {torch.cuda.device_count()}")
    print(f"現在のデバイス: {torch.cuda.current_device()}")
else:
    print("GPU利用不可")
