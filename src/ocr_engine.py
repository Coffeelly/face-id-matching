import base64
import json
import requests
import os
import re 
from typing import Optional, Dict, Any

class IdentityOCR:
    def __init__(self, model_name: str = "qwen3-vl:8b", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.api_url = f"{host}/api/generate"
        self.headers = {"Content-Type": "application/json"}

    def _encode_image(self, image_path: str) -> str:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _clean_and_parse_json(self, raw_text: str) -> Dict[str, Any]:
        """
        Function to fetch valid json
        """
        try:
            # 1. direct pass if the model is correct
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        # 2. if fail, use regex to find json
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        
        if match:
            clean_str = match.group(0)
            try:
                return json.loads(clean_str)
            except json.JSONDecodeError:
                return {"error": "Found JSON-like text but failed to parse", "raw": clean_str}
        
        return {"error": "No JSON object found in response", "raw": raw_text}

    def extract_data(self, image_path: str) -> Dict[str, Any]:
        print(f"🔄 OCR Running (Text Mode) with {self.model_name}...")
        
        try:
            b64_image = self._encode_image(image_path)
            
            prompt = (
                "Extract all text visible in this ID card image.\n"
                "Return the result STRICTLY as a JSON object.\n"
                "Do not include any explanation.\n\n"
                "JSON SCHEMAS:\n"
                "{\n"
                '  "nik": "string",\n'
                '  "nama": "string",\n'
                '  "tempat_tgl_lahir": "string",\n'
                '  "golongan_darah": "string",\n'
                '  "jenis_kelamin": "string",\n'
                '  "alamat": "string",\n'
                '  "rt_rw": "string",\n'
                '  "kel_desa": "string",\n'
                '  "kecamatan": "string",\n'
                '  "agama": "string",\n'
                '  "status_perkawinan": "string",\n'
                '  "pekerjaan": "string",\n'
                '  "kewarganegaraan": "string"\n'
                "}"
            )
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [b64_image],
                "stream": False,
                "options": {
                    "temperature": 0.1 
                }
            }
            
            response = requests.post(self.api_url, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status()
            
            raw_text = response.json().get("response", "")
            
            result_dict = self._clean_and_parse_json(raw_text)
            
            return result_dict

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return {"error": str(e)}

if __name__ == "__main__":
    ocr = IdentityOCR(model_name="qwen3-vl:8b")
 
    if os.path.exists("./result.png"):
        res = ocr.extract_data("./result.png")
        print("\n✅ Final Result (Dict):")
        print(json.dumps(res, indent=4))
    else:
        print("⚠️ File result.png not found.")