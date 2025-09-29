import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
import subprocess
import sys
import pandas as pd
from pathlib import Path
import gc
import logging
import json
import numpy as np
from datetime import datetime
import requests
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
BANKSIGHT_API= os.getenv('BANKSIGHT_API')
app = Flask(__name__)

MAX_REQUESTS_PER_USER = 30   # total lifetime requests allowed per user

# Store usage count per user
usage_log = defaultdict(int)

def has_exceeded_quota(ip: str) -> bool:
    """Check if an IP has exceeded its lifetime usage quota."""
    if usage_log[ip] >= MAX_REQUESTS_PER_USER:
        return True
    usage_log[ip] += 1
    return False

@app.before_request
def apply_quota_limit():
    """Enforce per-user total usage limit."""
    ip = request.remote_addr or "unknown"
    if has_exceeded_quota(ip):
        return jsonify({
            "status": "error",
            "message": "Usage limit reached. This demo is for limited portfolio showcase only."
        }), 403

def get_performance_metrics():
    """Reads the latest performance metrics from the metrics file."""
    # Only use the metrics file in the model_evaluation directory
    metrics_path = Path('artifacts/model_evaluation/metrics.json')
    
    if metrics_path.exists():
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                
                # Map the metrics to what the template expects
                return {
                    'mae': metrics.get('mae_in_rupees', 0),
                    'rmse': metrics.get('rmse_in_rupees', 0),
                    'accuracy': metrics.get('accuracy_percentage', 0),
                    'directional_accuracy': metrics.get('directional_accuracy_percentage', 0),
                    'mape': metrics.get('mape_percentage', 0)
                }
        except Exception as e:
            logging.error(f"Error reading metrics file: {e}")
    
    return None

def get_historical_highlights():
    """Reads prediction history and returns the top 3 most accurate predictions."""
    history_path = Path('artifacts/model_prediction/prediction_history.csv')
    if not history_path.exists():
        return []
    
    try:
        history_df = pd.read_csv(history_path)
        if history_df.empty or 'Actual' not in history_df.columns or 'Predicted' not in history_df.columns:
            return []
            
        # Calculate absolute percentage error for sorting
        history_df['Error'] = np.abs(history_df['Predicted'] - history_df['Actual'])
        history_df = history_df.sort_values(by='Error', ascending=True)
        
        # Get top 3, format them for display
        highlights = history_df.head(3).to_dict('records')
        
        # Clean up memory
        del history_df
        gc.collect()
        
        return highlights
    except Exception as e:
        print(f"Error reading or processing historical highlights: {e}")
        return []

def run_pipeline(choice, ticker=None):
    """Executes the main ML pipeline as a subprocess."""
    command = [sys.executable, 'main.py', '--choice', choice]
    if choice == '1' and ticker:
        command.extend(['--ticker', ticker])
    
    logging.info(f"Executing command: {' '.join(command)}")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        stdout, stderr = process.communicate(timeout=800) # 8-minute timeout
        if process.returncode != 0:
            logging.error(f"Pipeline execution failed.\nSTDOUT: {stdout}\nSTDERR: {stderr}")
            return False, None
        
        # Extract ticker from output - the prediction component prints "Predicted Close Price for {ticker}: {price}"
        detected_ticker = None
        for line in stdout.splitlines():
            if "Predicted Close Price for" in line:
                parts = line.split("Predicted Close Price for")[1].split(":")
                if parts:
                    detected_ticker = parts[0].strip()
                break
        
        logging.info(f"Pipeline executed successfully. Detected ticker: {detected_ticker}")
        return True, detected_ticker
    except Exception as e:
        logging.error(f"Failed to run pipeline: {e}")
        return False, None

def get_latest_prediction():
    """Reads the latest prediction from the predictions file."""
    prediction_path = Path('artifacts/model_prediction/predictions.csv')
    if not prediction_path.exists():
        return None
    try:
        pred_df = pd.read_csv(prediction_path)
        if not pred_df.empty:
            prediction_data = pred_df.tail(1).to_dict('records')[0]
            # Check if we need to extract ticker from the data
            if 'Ticker' in prediction_data:
                prediction_data['ticker'] = prediction_data['Ticker']
            return prediction_data
    except Exception as e:
        logging.error(f"Error reading prediction file: {e}")
    return None

@app.route('/')
def index():
    # On page load, show historical highlights if available
    highlights = get_historical_highlights()
    # Include today's date
    today_date = datetime.now().strftime('%Y-%m-%d')
    return render_template('index.html', highlights=highlights, today_date=today_date)

# Add a health check endpoint
@app.route('/health')
def health_check():
    return {'status': 'healthy'}, 200

# Add better error handling for the main routes
@app.errorhandler(500)
def server_error(e):
    logging.error(f"Server error: {e}", exc_info=True)
    if "Permission denied: '/app/mlruns'" in str(e):
        return {'error': 'MLflow permission error. Please check container permissions.'}, 500
    return {'error': 'Internal server error occurred'}, 500

@app.errorhandler(404)
def not_found(e):
    return {'error': 'Route not found'}, 404

def get_chart_data_from_pipeline(ticker):
    """
    Reads historical data from the file generated by the data ingestion pipeline.
    """
    try:
        # The data ingestion pipeline saves the data here
        data_path = Path('artifacts/data_ingestion/raw_data.csv')
        if not data_path.exists():
            print(f"Data file not found at {data_path}")
            return None, None

        # Read only necessary columns and use chunks for memory efficiency
        chunks = pd.read_csv(data_path, usecols=['Datetime', 'Ticker', 'Close'], chunksize=1000)
        relevant_data = []
        
        for chunk in chunks:
            # Filter for the specific ticker
            ticker_data = chunk[chunk['Ticker'] == ticker]
            if not ticker_data.empty:
                relevant_data.append(ticker_data)
        
        if not relevant_data:
            print(f"No data found for ticker {ticker}")
            return None, None
            
        # Combine filtered chunks
        hist_df = pd.concat(relevant_data, ignore_index=True)
        hist_df['Datetime'] = pd.to_datetime(hist_df['Datetime'])
        hist_df = hist_df.sort_values('Datetime')

        # Get the last 7 days for the chart
        last_7_days = hist_df.tail(7)
        
        if len(last_7_days) == 0:
            print(f"No recent data found for ticker {ticker}")
            return None, None
            
        prices = last_7_days['Close'].tolist()
        dates = last_7_days['Datetime'].dt.strftime('%Y-%m-%d').tolist()
        
        # Clean up memory
        del hist_df, last_7_days
        gc.collect()
        
        print(f"Chart data prepared: {len(dates)} dates and {len(prices)} prices for {ticker}")
        return prices, dates
    except Exception as e:
        print(f"Error reading historical data for chart: {e}")
        return None, None


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return redirect(url_for('index'))
        
    try:
        logging.info("Prediction request received.")
        choice = request.form['choice']
        ticker = request.form.get('ticker') # Safely get the ticker
        logging.info(f"Choice: {choice}, Ticker: {ticker}")

        if choice == '1' and not ticker:
            logging.warning("Company name or ticker symbol is required for manual prediction.")
            return render_template('index.html', error="Company name or ticker symbol is required for manual prediction.")

        # Run the main pipeline
        pipeline_success, detected_ticker = run_pipeline(choice, ticker)
        if not pipeline_success:
            error_message = "An error occurred during the pipeline execution. This may be due to insufficient data for the selected ticker. Please try a different ticker with more historical data."
            return render_template('index.html', error=error_message)

        # After the pipeline runs, get the latest prediction and metrics
        prediction_data = get_latest_prediction()
        logging.info(f"Prediction data: {prediction_data}")  # Debug log
        
        # Determine which ticker to display
        display_ticker = ticker
        if choice == '2' and detected_ticker:
            display_ticker = detected_ticker
            logging.info(f"AI selected ticker: {display_ticker}")
        
        # If we have prediction data but no ticker (should never happen now), add the ticker
        if prediction_data and not 'ticker' in prediction_data and display_ticker:
            prediction_data['ticker'] = display_ticker
        
        metrics = get_performance_metrics()
        highlights = get_historical_highlights() # Refresh highlights

        # Prepare chart data - use the display_ticker for both manual and AI selection
        chart_prices, chart_dates = get_chart_data_from_pipeline(display_ticker)

        # Get today's date in the correct format
        today_date = datetime.now().strftime('%Y-%m-%d')
        
        return render_template(
            'index.html', 
            prediction=prediction_data, 
            metrics=metrics,
            ticker=display_ticker,  # Use the display_ticker which works for both manual and AI selection
            chart_prices=chart_prices,
            chart_dates=chart_dates,
            highlights=highlights,
            today_date=today_date  # Add today's date
        )

    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        return render_template('index.html', error="An error occurred during the prediction process.")



@app.route('/external_predict', methods=['POST'])
def external_predict():
    """
    Calls another Flask app's /predict API and returns the result.
    Expects 'date' in the POST JSON body.
    """
    try:
        data = request.get_json()
        date = data.get('date')
        if not date:
            return {'error': 'Date is required.'}, 400

        # Call the external API
        response = requests.post(
            BANKSIGHT_API,
            json={"date": date}
        )
        # Try to parse JSON, handle errors gracefully
        try:
            result = response.json()
            logging.info(f"External API response: {result}")  # Add this line

            # Map external API response to frontend format
            return {
                "date": result.get("date"),
                "predicted_price": result.get("predicted_close")
            }, response.status_code

        except Exception:
            logging.error(f"External API did not return JSON. Status: {response.status_code}, Text: {response.text}")
            return {'error': 'External API did not return valid JSON.', 'details': response.text}, 502
    except Exception as e:
        logging.error(f"Error calling external API: {e}")
        return {'error': 'Failed to call external API.'}, 500



if __name__ == "__main__":
    # Running in debug mode is convenient for development.
    # Host '0.0.0.0' makes it accessible from your local network.
    app.run(debug=True, host='0.0.0.0', port=8080)
