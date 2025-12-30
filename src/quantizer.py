# src/quantizer.py
import torch
import os
from facenet_pytorch import InceptionResnetV1

def quantize_model():
    print("🚀 Starting Model Quantization Pipeline...")
    
    # 1. Setup paths
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    fp32_path = os.path.join(models_dir, "resnet_fp32.pt")
    int8_path = os.path.join(models_dir, "resnet_int8.pt")

    # 2. Load the Standard Model (FP32)
    print("   Loading InceptionResnetV1 (Pre-trained)...")
    model = InceptionResnetV1(pretrained='vggface2').eval()

    # 3. SAVE the FP32 model locally (so we can compare size)
    print(f"   💾 Saving FP32 model to {fp32_path}...")
    torch.save(model.state_dict(), fp32_path)

    # 4. Apply Dynamic Quantization
    print("   🔨 Quantizing model (FP32 -> INT8)...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear},  # Quantizing Linear layers offers best size/speed balance
        dtype=torch.qint8
    )
    
    # 5. SAVE the INT8 model
    print(f"   💾 Saving INT8 model to {int8_path}...")
    torch.save(quantized_model.state_dict(), int8_path)
    
    # 6. Compare File Sizes
    fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
    int8_size = os.path.getsize(int8_path) / (1024 * 1024)
    reduction = (1 - (int8_size / fp32_size)) * 100
    
    print("\n✅ QUANTIZATION SUCCESS!")
    print(f"   Original Size (FP32):   {fp32_size:.2f} MB")
    print(f"   Optimized Size (INT8):  {int8_size:.2f} MB")
    print(f"   📉 Size Reduction:      {reduction:.1f}%")

if __name__ == "__main__":
    quantize_model()