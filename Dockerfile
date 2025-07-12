FROM python:3.11.0-slim-buster

WORKDIR /app

# Copy all files first
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install .

# Configure for Cloud Run
ENV PORT=8080
EXPOSE ${PORT}

# Set Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1

CMD ["python3", "-u", "app.py"]