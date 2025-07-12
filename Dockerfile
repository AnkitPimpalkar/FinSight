FROM python:3.11.0-slim-buster

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the code
COPY . .

# Configure for Cloud Run
ENV PORT=8080
EXPOSE ${PORT}

# Set Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1

CMD ["python3", "-u", "app.py"]