# FinSight 
## - Stock Price Prediction with ML Pipeline
FinSight is a financial prediction application that uses machine learning to forecast stock prices. The project demonstrates end-to-end MLOps practices with a focus on pipeline automation rather than complex model building.

<img alt="FinSight" src="static/image.png">

## Features
Stock Price Prediction: Get next-day closing price predictions for any NSE (National Stock Exchange of India) listed stock
Dual Selection Methods:
  1.Manual entry of company ticker
  2.AI-powered selection of bullish stocks using LLM agents
Real-time Model Training: Fresh models trained for each stock selection
Interactive Visualization: Chart displays of historical prices and predictions
BankSight Integration: Latest project- BankNifty prediction


## Project Structure
FinSight/
│
├── artifacts/            # Generated ML artifacts by pipeline stages
├── config/               # Configuration files
├── logs/                 # Application logs
├── research/             # Jupyter notebooks for development
├── src/finance_ml/       # Core ML code
│   ├── components/       # ML pipeline components
│   ├── config/           # Configuration management
│   ├── constants/        # Project constants
│   ├── entity/           # Data classes
│   ├── pipeline/         # ML pipeline orchestration
│   └── utils/            # Utility functions
├── static/               # Static web assets
├── templates/            # HTML templates
├── tests/                # Unit tests
├── app.py                # Flask web application
├── main.py               # CLI entry point
├── Dockerfile            # Container definition
└── requirements.txt      # Project dependencies


## Technical Overview
ML Pipeline
The project implements a complete ML pipeline with the following stages:

1. Ticker Selection (stage_01_LLMticker.py): Either manual input or LLM-powered selection
2. Data Ingestion (stage_02_data_ingestion.py): Fetches stock data from Yahoo Finance
3. Data Validation (stage_03_data_validation.py): Validates data against schema
4. Data Transformation (stage_04_data_transformation.py): Preprocessing for LSTM model
5. Model Training (stage_05_model_training.py): Trains LSTM neural network
6. Model Evaluation (stage_06_model_evaluation.py): Calculates performance metrics
7. Model Prediction (stage_07_model_prediction.py): Generates final prediction


## LLM Agent Integration
The application integrates with LLM models through the Phi library to identify bullish stocks:

* Web agent searches current financial news
* Finance agent analyzes financial data using Yahoo Finance tools
* Team of agents collaborates to select the most promising stock


## Model Architecture
* LSTM (Long Short-Term Memory) neural network for time series forecasting
* Configurable hyperparameters in params.yaml
* MLflow tracking for experiment management


## Setup Instructions
### Prerequisites
* Python 3.11
* OpenAI API key (for LLM functionality)
* BankSight API (for BankNifty prediction)


### Local Environment Setup
#Clone the repository
git clone https://github.com/YourUsername/FinSight.git
cd FinSight

#Create virtual environment (optional but recommended)
python -m venv FinSight
source FinSight/bin/activate  # On Windows: FinSight\Scripts\activate

#Install dependencies
pip install -r requirements.txt
pip install -e .

#Set your API keys (use `export` for Linux/Mac, `$env:` for Windows PowerShell)
#Linux/Mac:
export OPENAI_API_KEY="your-openai-api-key"
export BANKSIGHT_API="BankSIght-api-key"
#Windows PowerShell:
#$env:OPENAI_API_KEY="your-openai-api-key"
#$env:BANKSIGHT_API="BankSIght-api-key"

#Create mlruns directory for MLflow
mkdir -p mlruns

#Run the Flask application
python app.py


### Docker Setup
#Build the Docker image
docker build -t finsight .

#Run the container with necessary environment variables
docker run -p 8080:8080 \
  -e OPENAI_API_KEY=your-openai-api-key \
  -e BANKSIGHT_API=BankSIght-api-key \
  --name finsight-app \
  finsight


## Usage
1. Access the application at https://finsight-phuysyk4na-el.a.run.app/
2. Choose between manual stock selection or AI-powered selection:
  * Manual: Enter a company ticker symbol (e.g., "Reliance" == "RELIANCE.NS")
  * LLM-Powered: Let the AI find the most bullish stock
3. Wait for the pipeline to execute (data retrieval, model training, prediction)
4. View the prediction results and performance metrics


## Configuration
The application can be configured through several YAML files:

* config.yaml: Main configuration for pipeline stages
* params.yaml: Model hyperparameters and settings
* schema.yaml: Data validation schema
Example params.yaml settings:
model_training:
  epochs: 20
  batch_size: 32
  lstm_units_1: 64
  lstm_units_2: 64
  dense_units_1: 128
  dropout_rate: 0.5


## Cloud Deployment
The application is deployed on Google Cloud Run for scalability:

1. Each prediction request triggers a complete pipeline run
2. Models are trained on-demand for any requested stock
3. Resources scale automatically based on demand
Note: Due to cold starts and real-time training, predictions may take 2-4 minutes to complete.


## Troubleshooting
### Common Issues

* Access Denied: To prevent misuse, the application enforces request limits. If you encounter this error, it is likely due to reaching the maximum allowed requests. This restriction is in place because the project is intended as a demonstration for recruiters.
* "Invalid ticker" error: Ensure you're using NSE tickers (ending with .NS).
* Pipeline timeout: The pipeline has an 8-minute timeout. For very large datasets, adjust the timeout in app.py.


### Debug Logs
Logs are stored in running_logs.log. Enable debug mode for more verbose logging:
#In app.py
app.config['DEBUG'] = True


## Project Learning Goals
This project was created as a learning exercise focusing on:

* End-to-end MLOps pipeline implementation
* Automated model training and deployment
* Integration of LLM agents for decision support
* Cloud-native ML application development
* Containerization of ML applications


## Future Improvements
* Add more sophisticated technical indicators
* Implement ensemble methods for improved accuracy
* Add user accounts to track prediction history
* Expand to international stock markets
* Optimize pipeline for faster execution


## License
This project is licensed under the terms included in the LICENSE file.

## Acknowledgments
* Data provided by Yahoo Finance API
* LLM capabilities powered by OpenAI and Groq
* Project structure inspired by MLOps best practices

Note: This application is for educational purposes only and should not be used for actual trading decisions.

