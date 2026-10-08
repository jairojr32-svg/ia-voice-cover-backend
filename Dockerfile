# Use an official NVIDIA CUDA runtime as a parent image
# This image contains the CUDA toolkit and drivers needed for the GPU
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Set environment variables to ensure Python and other tools work correctly
ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# Install essential packages and Python itself
RUN apt-get update --fix-missing && \
    apt-get install -y wget git ffmpeg libsndfile1 python3.9 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
# This uses the specific CUDA-enabled version of PyTorch
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Download the RVC models during the build process
RUN python3 download_popular_models.py

# Expose the port and run the server
EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
