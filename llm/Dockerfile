# Use a PyTorch base image
FROM nvcr.io/nvidia/pytorch:24.06-py3


# Set the working directory
WORKDIR /app

# Install Python dependencies
COPY Code .

# RUN  apt-get update && apt-get install -y \
    # git \
    # curl \
    # python3-dev \
    # build-essential \
    # libssl-dev \
    # libffi-dev \
    # libpython3-dev \
    # && rm -rf /var/lib/apt/lists/*  && \
RUN pip install -r requirements.txt

# Download the model during build
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-3B-Instruct')"

# Expose the port FastAPI will run on
EXPOSE 8000

# Set the default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]