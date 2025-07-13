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
ENV TF_CPP_MIN_LOG_LEVEL=2

# Create and set permissions for directories the app needs to write to
RUN mkdir -p /app/artifacts /app/logs /app/mlruns && \
    chown -R nobody:nogroup /app/artifacts /app/logs /app/mlruns && \
    chmod -R 700 /app/artifacts /app/logs /app/mlruns

USER nobody

# Use Gunicorn with proper production settings
CMD exec gunicorn --bind 0.0.0.0:$PORT app:app --workers 1 --threads 8