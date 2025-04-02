FROM python:3.12-slim

# Install pip3 and ffmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /Backend

# Copy application files
COPY . /Backend

# Install Python dependencies from requirements.txt 
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

# Expose port 8001
EXPOSE 8001

# Deploy the server
CMD ["python3", "rag_backend.py"]