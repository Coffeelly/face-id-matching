# src/face_engine.py
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.nn.functional import cosine_similarity
import time
import os

class FaceVerifier:
    def __init__(self, use_quantized=False):
        """
        Initializes the Engine. 
        If use_quantized=True, loads the INT8 model optimized for CPU.
        """
        # Quantized models run best on CPU (simulating mobile devices)
        self.device = torch.device('cpu') if use_quantized else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_quantized = use_quantized
        
        print(f"Loading Engine | Device: {self.device} | Quantized: {use_quantized}")

        # 1. Load Detector
        self.mtcnn = MTCNN(keep_all=False, device=self.device)

        # 2. Load Recognizer
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        
        if use_quantized:
            # Apply the structure for quantization first
            self.resnet = torch.quantization.quantize_dynamic(
                self.resnet, {torch.nn.Linear}, dtype=torch.qint8
            )
            # Load the trained INT8 weights
            model_path = os.path.join(os.path.dirname(__file__), '../models/resnet_int8.pt')
            if os.path.exists(model_path):
                self.resnet.load_state_dict(torch.load(model_path))
                print("Loaded INT8 Quantized Weights")
            else:
                print("Warning: Quantized weights not found. Running dynamic quantization on the fly.")

        self.resnet.to(self.device)
        
    def process_image(self, image_file):
        """Detects face and returns cropped tensor"""
        try:
            from PIL import Image
            img = Image.open(image_file).convert('RGB')
            img_cropped = self.mtcnn(img) 
            return img_cropped
        except Exception as e:
            return None

    def verify(self, id_tensor, selfie_tensor):
        """Returns similarity score + latency in ms"""
        start_time = time.time()
        
        # Ensure inputs are on the correct device
        id_tensor = id_tensor.to(self.device)
        selfie_tensor = selfie_tensor.to(self.device)

        with torch.no_grad():
            id_emb = self.resnet(id_tensor.unsqueeze(0))
            selfie_emb = self.resnet(selfie_tensor.unsqueeze(0))
        
        score = cosine_similarity(id_emb, selfie_emb).item()
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000 
        
        return score, latency