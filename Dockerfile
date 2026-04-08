# Use a lightweight Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOME=/home/user
ENV PATH="/home/user/.local/bin:${PATH}"

# Create a non-root user
RUN useradd -m -u 1000 user
USER user
WORKDIR $HOME/app

# Copy the requirements file first for caching
COPY --chown=user requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

# Metadata and configuration
ENV API_BASE_URL="https://api-inference.huggingface.co/v1/"
ENV MODEL_NAME="meta-llama/Llama-3-70b-instruct"
ENV HF_TOKEN=""
ENV TASK_ID="easy"

# Run the inference script
CMD ["python", "inference.py"]
