# Use a slim Python image for efficiency
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for gRPC
RUN apt-get update && apt-get install -y build-essential

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the generated proto files and the server script
COPY . .

# Expose the gRPC port
EXPOSE 50052

# Run the worker
CMD ["python", "model_server.py"]