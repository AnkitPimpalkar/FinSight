FROM python:3.11.0-slim-buster

WORKDIR /app

# Copy all files first
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

CMD ["python", "app.py"]