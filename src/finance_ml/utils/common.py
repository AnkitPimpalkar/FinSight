import os
from box.exceptions import BoxValueError
import yaml
from finance_ml import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import yfinance as yf
import re


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data



@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


# Dictionary mapping Indian company names to their NSE tickers
INDIAN_COMPANIES = {
    # Major companies
    "reliance": "RELIANCE.NS",
    "tata consultancy": "TCS.NS",
    "infosys": "INFY.NS",
    "hdfc bank": "HDFCBANK.NS",
    "itc": "ITC.NS",
    "tcs": "TCS.NS",
    "hdfc": "HDFC.NS",
    "hindustan unilever": "HINDUNILVR.NS",
    "hul": "HINDUNILVR.NS",
    "icici bank": "ICICIBANK.NS",
    "sbi": "SBIN.NS",
    "state bank of india": "SBIN.NS",
    "airtel": "BHARTIARTL.NS",
    "bharti airtel": "BHARTIARTL.NS",
    "axis bank": "AXISBANK.NS",
    "bajaj finance": "BAJFINANCE.NS",
    "maruti": "MARUTI.NS",
    "maruti suzuki": "MARUTI.NS",
    "tata motors": "TATAMOTORS.NS",
    "kotak mahindra bank": "KOTAKBANK.NS",
    "kotak bank": "KOTAKBANK.NS",
    "larsen & toubro": "LT.NS",
    "larsen and toubro": "LT.NS",
    "lt": "LT.NS",
    "mahindra": "M&M.NS",
    "mahindra and mahindra": "M&M.NS",
    "adani": "ADANIPORTS.NS",
    "adani ports": "ADANIPORTS.NS",
    "wipro": "WIPRO.NS",
    "sun pharma": "SUNPHARMA.NS",
    "nestle": "NESTLEIND.NS",
    "nestle india": "NESTLEIND.NS",
    "tech mahindra": "TECHM.NS",
    "asian paints": "ASIANPAINT.NS",
    "hero motocorp": "HEROMOTOCO.NS",
    "hero": "HEROMOTOCO.NS",
    "yes bank": "YESBANK.NS",
    "power grid": "POWERGRID.NS",
    "tata steel": "TATASTEEL.NS",
    "ntpc": "NTPC.NS",
    "ongc": "ONGC.NS",
    "oil and natural gas": "ONGC.NS",
    "bajaj auto": "BAJAJ-AUTO.NS",
    "ultratech cement": "ULTRACEMCO.NS",
    "ultratech": "ULTRACEMCO.NS",
    "hcl tech": "HCLTECH.NS",
    "hcl": "HCLTECH.NS",
    "hcl technologies": "HCLTECH.NS",
    "grasim": "GRASIM.NS",
    "jindal steel": "JINDALSTEL.NS",
    "jspl": "JINDALSTEL.NS",
    "coal india": "COALINDIA.NS"
    # Add more as needed
}


def normalize_input_to_ticker(input_text: str) -> str:
    """Converts user input to a valid yfinance ticker.
    
    This function handles:
    1. Company names -> ticker
    2. Tickers without .NS suffix
    3. Already valid tickers
    
    Args:
        input_text (str): User input for company name or ticker
        
    Returns:
        str: Valid yfinance ticker or None if no match
    """
    if not input_text:
        return None
        
    # Remove extra spaces and convert to lowercase for comparison
    normalized_input = input_text.strip().lower()
    
    # Case 1: Direct match with company name
    if normalized_input in INDIAN_COMPANIES:
        return INDIAN_COMPANIES[normalized_input]
    
    # Case 2: Already a valid ticker with .NS suffix
    if re.match(r"^[A-Za-z0-9]+\.NS$", input_text):
        return input_text.upper()
    
    # Case 3: Valid ticker without .NS suffix
    ticker_candidate = f"{input_text.upper()}.NS"
    
    # Try all possible tickers (original and with .NS suffix)
    possible_tickers = [
        input_text.upper(),          # As is (could be non-Indian ticker)
        ticker_candidate             # With .NS suffix for NSE
    ]
    
    for ticker in possible_tickers:
        try:
            data = yf.download(ticker, period="1d", interval="1d", progress=False)
            if not data.empty:
                return ticker
        except Exception:
            continue
    
    # If no match found, try partial name matching
    for company, ticker in INDIAN_COMPANIES.items():
        if normalized_input in company or company in normalized_input:
            return ticker
    
    # If all else fails, return the input with .NS suffix as best guess
    return ticker_candidate





