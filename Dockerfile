# Use Python 3.10 as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install consolidated requirements
# This avoids version conflicts and reduces memory usage
COPY consolidated_requirements.txt ./
RUN pip install --no-cache-dir -r consolidated_requirements.txt

# Copy project files
COPY . .

# Initialize git submodules (skip --recursive as requested)
RUN git submodule update --init

# Copy .gitmodules file to OpenFace-3.0 directory to fix submodule issue
RUN cp setup-info/.gitmodules OpenFace-3.0/.gitmodules

# Update submodules in OpenFace-3.0 directory
RUN cd OpenFace-3.0 && git submodule update --init

# Install openface-test package
RUN pip install --no-cache-dir openface-test

# Download OpenFace models and move them to the correct location
RUN openface download --output aux && \
    mkdir -p OpenFace-3.0/weights && \
    cp -n aux/* OpenFace-3.0/weights/ 2>/dev/null || true && \
    rm -rf aux

# Expose port for the API (assuming it runs on port 5000)
EXPOSE 5000

# Set default command (can be overridden)
CMD ["python", "openface-api/app.py"]
