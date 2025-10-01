# Use Python 3.9 slim base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('vader_lexicon', quiet=True); nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True)"

# Copy project files
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY main.py .

# Create necessary directories
RUN mkdir -p models reports

# Set Python path
ENV PYTHONPATH=/app/src:/app

# Expose port (if running a web service in the future)
EXPOSE 8080

# Set the default command
CMD ["python", "main.py"]

# Alternative commands for development:
# For Jupyter notebook: CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
# For interactive shell: CMD ["/bin/bash"]