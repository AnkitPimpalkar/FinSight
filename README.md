# FinSight

A financial prediction application that uses machine learning to predict stock prices.

## Setup Instructions

### Local Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Set your API keys (use `export` for Linux/Mac, `$env:` for Windows PowerShell)
# Linux/Mac:
export OPENAI_API_KEY="your-openai-api-key"
export GROQ_API_KEY="your-groq-api-key"
# Windows PowerShell:
# $env:OPENAI_API_KEY="your-openai-api-key"
# $env:GROQ_API_KEY="your-groq-api-key"

# Create mlruns directory with proper permissions
mkdir -p mlruns

# Run the application
python main.py
```

### Docker Setup

```bash
# Build the Docker image
docker build -t finsight .

# Run the container with necessary environment variables
docker run -p 8080:8080 \
  -e OPENAI_API_KEY=your-openai-api-key \
  -e GROQ_API_KEY=your-groq-api-key \
  --name finsight-app \
  finsight
```

## Troubleshooting

### Permission Denied for MLflow
If you encounter `PermissionError: [Errno 13] Permission denied: '/app/mlruns'` when running in Docker, this should be fixed in the latest Dockerfile.

## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the app.py