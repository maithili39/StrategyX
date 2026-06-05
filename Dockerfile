# Use official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY config.yaml .
COPY ott_train.csv .
COPY ott_test.csv .

# Expose ports for both Streamlit and FastAPI
EXPOSE 8501
EXPOSE 8000

# Default command launches the Streamlit Control Center
# To launch the API, run: docker run -p 8000:8000 <image> uvicorn src.api:app --host 0.0.0.0 --port 8000
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
