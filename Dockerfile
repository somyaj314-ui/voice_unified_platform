FROM python:3.10-slim

# Install system dependencies required for audio processing (soundfile, librosa, sounddevice, whisper)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    portaudio19-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to keep image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose common ports used by the application (as seen in config/env)
EXPOSE 8767
EXPOSE 8768

# Set environment variable to ensure python output is unbuffered
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["python", "main.py"]
