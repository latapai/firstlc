# Step 1: Use the official Python 3.12 image as the base image
FROM python:3.12-slim
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the requirements.txt into the container
COPY requirements.txt .

# Step 4: Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of your application code into the container
COPY . .

# Step 6: Expose port 8003
EXPOSE 8003

# Step 7: Run your application on port 8003
CMD ["chainlit", "run", "lic_rag.py", "--port", "8003"]
