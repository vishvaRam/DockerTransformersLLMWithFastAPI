# Dockerized CUDA-Accelerated LLM API

This project provides a Dockerized implementation of a CUDA-accelerated API for generating responses using a large language model (LLM). It leverages NVIDIA GPUs for efficient inference and is built using FastAPI, PyTorch, and Hugging Face Transformers.

## Features

- **CUDA Acceleration**: Utilizes NVIDIA GPUs for faster inference using PyTorch.
- **Pretrained Model**: Automatically downloads and uses the `Qwen/Qwen2.5-3B-Instruct` model from Hugging Face.
- **FastAPI Integration**: Exposes endpoints for generating responses from the LLM.
- **Dockerized Deployment**: Simplifies deployment with Docker and Docker Compose.
- **Custom Timer**: Tracks and logs the time taken for model inference.

## Project Structure

- **`AILLM.py`**: Implements the `QwenModel` class for loading the model and generating responses.
- **`main.py`**: Defines the FastAPI application and endpoints.
- **`timer.py`**: Provides a utility class for measuring execution time.
- **`requirements.txt`**: Lists Python dependencies.
- **`Dockerfile`**: Defines the Docker image for the application.
- **`docker-compose.yml`**: Configures the Docker Compose setup for running the application.

## Endpoints

### Root Endpoint
- **URL**: `/`
- **Method**: `GET`
- **Response**: Returns a message indicating the API is running.

### Generate Responses
- **URL**: `/generate`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "prompts": ["Your input prompt here"]
  }
- **Response:** Returns generated responses in JSON format.

## Setup and Usage

Follow these steps to set up and run the project:

### Prerequisites

- Ensure you have the following installed on your system:
  - **Docker**: [Install Docker](https://docs.docker.com/get-docker/)
  - **Docker Compose**: [Install Docker Compose](https://docs.docker.com/compose/install/)
  - **NVIDIA GPU** with CUDA support
  - **NVIDIA Container Toolkit**: [Install NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

---

### Steps to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd Docker_cuda_LLM
2. **Build the Docker Image**:
   Build the Docker image using Docker Compose:
   ```bash
    sudo docker compose build
3. **Run the Application**: Start the application in detached mode:
  ```bash
      sudo docker compose up
```
4. **Test the /generate Endpoint**: Use a tool like curl or Postman to send a POST request to the /generate endpoint:
  ```bash
    curl -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"prompts": ["Your input prompt here"]}'
```
5. **Stop the Application**: To stop the running containers
   ```bash
   sudo docker compose down
**Notes**
- The model is downloaded and loaded during the Docker build process.
- Ensure your NVIDIA drivers and CUDA are properly configured for GPU acceleration.
