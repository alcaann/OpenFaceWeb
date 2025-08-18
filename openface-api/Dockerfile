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

# Copy shared project files
COPY OpenFace-3.0 ./OpenFace-3.0
COPY setup-info ./setup-info
COPY .git ./.git
COPY .gitmodules ./.gitmodules

# Copy API-specific files
COPY openface-api ./openface-api

# Initialize git submodules (skip --recursive as requested)
RUN git submodule update --init

# Copy .gitmodules file to OpenFace-3.0 directory to fix submodule issue
RUN cp setup-info/.gitmodules OpenFace-3.0/.gitmodules

# Update submodules in OpenFace-3.0 directory
RUN cd OpenFace-3.0 && git submodule update --init

# Create weights directory first
RUN mkdir -p OpenFace-3.0/weights

# Create logs directory for API logging
RUN mkdir -p openface-api/logs && chmod 755 openface-api/logs

# Download OpenFace models with cache mount for persistent caching
RUN --mount=type=cache,target=/cache/openface \
    if [ ! -d "/cache/openface/weights" ]; then \
        openface download --output /cache/openface/weights; \
    fi && \
    cp -r /cache/openface/weights/* OpenFace-3.0/weights/

# Expose port for the API (assuming it runs on port 5000)
EXPOSE 5000

# Set default command (can be overridden)
CMD ["python", "openface-api/app.py"]
