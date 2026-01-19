# AI-Powered Identity Verification System

Face-ID Verification System is a production-prototype e-KYC (Know Your Customer) solution designed to demonstrate modern Deep Learning engineering practices for FinTech applications.

The system implements a hybrid verification pipeline using PyTorch for face matching and a Multimodal LLM for intelligent document extraction. It includes a comprehensive Streamlit dashboard for testing, benchmarking, and visualizing the verification process.

## Key Features

- **Hybrid Inference Engine:** Seamlessly toggle between Standard (FP32) and Dynamically Quantized (INT8) models to evaluate accuracy vs. latency trade-offs.
- **Biometric Verification:** Implements 1:1 Face Matching using InceptionResnetV1 (trained on VGGFace2) and MTCNN for face alignment.
- **Generative OCR (New):** Utilizes a local Vision-Language Model (Qwen-VL via Ollama) to intelligently extract and parse structured data (NIK, Name, Address) from ID cards, robust against noise and formatting issues.
- **Performance Optimization:** Includes a dedicated quantization pipeline that compresses model weights by approximately 15%, optimized for CPU deployment.
- **Engineering Reliability:** Features a built-in Unit Test suite (via `unittest`) to validate tensor shapes, model initialization, and similarity logic before deployment.
- **Interactive Dashboard:** A Streamlit-based UI that provides real-time feedback on similarity scores, inference time, extracted text data, and detection visualization.

## Prerequisites

- Anaconda or Miniconda installed on your system.
- [Ollama](https://ollama.com/) installed and running locally.
- A Groq API Key (available from the Groq Console).

## Installation

Follow these steps to set up the project environment.

### 1. Clone the Repository

```
git clone https://github.com/Coffeelly/face-id-matching.git
cd face-id-matching
```

### 2. Install Dependencies

This project uses an environment.yml file to manage dependencies.

```
conda env create -f environment.yml
conda activate face-id-matching
```

### 3. Setup OCR Engine (Ollama)

This system requires a local Vision Model to perform OCR. Ensure Ollama is running, then pull the model:

```
ollama pull qwen3-vl:8b
```

### 4. Initialize Face Models

Run the quantization script to download the base models and generate the optimized INT8 version.

```
python src/quantizer.py
```

### 5. Run the Application

```
streamlit run app.py
```

### 6. Running Unit Tests

Execute the test suite to ensure the engine is functioning correctly.

```
python tests/test_basic.py
```

## Project Structure

Ensure your project files are organized as follows:

```text
/Face-ID-Matching
├── /app                # Frontend Application
│   └── main.py         # Streamlit Dashboard Entry Point
├── /src                # Core AI Logic
│   ├── face_engine.py  # FaceVerifier Class (Inference Logic)
│   ├── ocr_engine.py   # IdentityOCR Class (Ollama/Qwen-VL Integration)
│   └── quantizer.py    # Model Optimization Pipeline (FP32 -> INT8)
├── /models             # Model Registry
│   ├── resnet_fp32.pt  # Baseline Model
│   └── resnet_int8.pt  # Quantized Model
├── /tests              # CI/CD Test Suite
│   └── test_basic.py   # Unit Tests for Engine Integrity
└── environment.yml     # Dependency Management
```

## Performance Benchmarks

The following benchmarks were observed on a standard CPU environment:

| Model Architecture | Precision | Model Size | Inference Latency (Avg) |
| ------------------ | --------- | ---------- | ----------------------- |
| InceptionResnetV1  | FP32      | ~106 MB    | ~51 ms                  |
| InceptionResnetV1  | INT8      | ~91 MB     | ~51 ms                  |

_Note: The current implementation uses Dynamic Quantization, which optimizes Linear layers. Future updates will target Conv2d layers for greater speedups._

## Project Roadmap (To-Do)

### Optimization & Engineering

- [x] **Static Quantization:** Implement Static Quantization with a calibration dataset to optimize Conv2d layers for significant speed improvements on Edge devices. **(Currently blocked by `facenet-pytorch` library architecture (Residual connections lack `FloatFunctional` support). Requires custom model implementation.)**
- **Dockerization:** Create a Dockerfile to containerize the application for cloud deployment (GCP/AWS).
- **FastAPI Backend:** Decouple the engine into a REST API to support mobile app integration.

### Feature Development

- **Live Webcam Integration:** Add support for real-time video capture in the browser to simulate live user onboarding.
- **Liveness Detection:** Implement an anti-spoofing layer (e.g., detecting screen reflections or depth cues) to prevent fraud.
- [x] **OCR Module:** Integrated a Generative AI OCR engine (Qwen-VL) to extract NIK and Name from ID cards.

## Tech Stack

- **Language:** Python 3.10+
- **Core AI:** PyTorch, Torchvision, Facenet-PyTorch
- **Generative AI:** Ollama (Qwen-VL)
- **UI & Tools:** Streamlit, Pillow, Requests
