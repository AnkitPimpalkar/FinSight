FROM python:3.11.0-slim-buster

WORKDIR /app

# Copy all files first
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install . && \
    pip install gunicorn

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV HOST=0.0.0.0

# Create artifacts directory with proper permissions
RUN mkdir -p /app/artifacts/data_ingestion && \
    chown -R nobody:nogroup /app/artifacts && \
    chmod -R 777 /app/artifacts

USER nobody

# Use Gunicorn with proper production settings
CMD exec gunicorn --bind 0.0.0.0:$PORT app:app --workers 1 --threads 8