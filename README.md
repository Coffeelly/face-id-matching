# AI-Powered Identity Verification System

Face-ID Verification System is a production-prototype e-KYC (Know Your Customer) solution designed to demonstrate modern Deep Learning engineering practices for FinTech applications.

The system implements a hybrid face verification pipeline using PyTorch, capable of switching between high-precision (FP32) and high-efficiency (INT8 Quantized) inference modes in real-time. It includes a comprehensive Streamlit dashboard for testing, benchmarking, and visualizing the verification process.

## Key Features

- **Hybrid Inference Engine:** Seamlessly toggle between Standard (FP32) and Dynamically Quantized (INT8) models to evaluate accuracy vs. latency trade-offs.
- **Biometric Verification:** Implements 1:1 Face Matching using InceptionResnetV1 (trained on VGGFace2) and MTCNN for face alignment.
- **Performance Optimization:** Includes a dedicated quantization pipeline that compresses model weights by approximately 15%, optimized for CPU deployment.
- **Engineering Reliability:** Features a built-in Unit Test suite (via `unittest`) to validate tensor shapes, model initialization, and similarity logic before deployment.
- **Interactive Dashboard:** A Streamlit-based UI that provides real-time feedback on similarity scores, inference time, and detection visualization.

## Prerequisites

- Anaconda or Miniconda installed on your system.
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

### 3. Initialize Models

Run the quantization script to download the base models and generate the optimized INT8 version.

```
python src/quantizer.py
```

### 4. Run the Application

```
streamlit run app.py
```

### 5. Running Unit Tests

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

- **Static Quantization:** Implement Static Quantization with a calibration dataset to optimize Conv2d layers for significant speed improvements on Edge devices.
- **Dockerization:** Create a Dockerfile to containerize the application for cloud deployment (GCP/AWS).
- **FastAPI Backend:** Decouple the engine into a REST API to support mobile app integration.

### Feature Development

- **Live Webcam Integration:** Add support for real-time video capture in the browser to simulate live user onboarding.
- **Liveness Detection:** Implement an anti-spoofing layer (e.g., detecting screen reflections or depth cues) to prevent fraud.
- **OCR Module:** Integrate an OCR engine (such as PaddleOCR or Tesseract) to extract Name and other data from IDs.

## Tech Stack

- **Language:** Python 3.10+
- **Frameworks:** PyTorch, Streamlit
- **Libraries:** Facenet-PyTorch, PIL, Unittest
