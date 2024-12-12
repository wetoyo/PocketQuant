import yfinance as yf
import numpy as np
import xgboost as xgb  # Import xgboost for using the model
from datetime import datetime, timedelta
from train_xgboost import train_xgboost_model
import json

def fetch_minute_data(ticker, interval="1m"):
    """
    Fetches minute-level stock data for the last 7 days.
    
    Parameters:
        ticker (str): Stock ticker symbol.
        interval (str): Data interval (e.g., '1m', '2m').
        
    Returns:
        tuple: Features (X), Target (y), and DataFrame of original stock data.
    """
    stock = yf.Ticker(ticker)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    data = stock.history(interval=interval, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

    if data.empty:
        raise ValueError(f"No data fetched for ticker {ticker} at interval {interval}.")

    # Prepare features and target
    data['Target'] = data['Close'].shift(-1)  # Predict next minute's close
    data = data.dropna()  # Drop rows with NaN values

    X = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    y = data['Target'].values

    return X, y, data

def predict_with_model(model, X):
    """
    Uses the trained XGBoost model to make predictions.
    
    Parameters:
        model (xgboost.Booster): Trained XGBoost model.
        X (np.ndarray): Feature matrix for prediction.
        
    Returns:
        np.ndarray: Predicted values.
    """
    dtest = xgb.DMatrix(X)
    predictions = model.predict(dtest)
    return predictions

def get_current_time_features(ticker, interval="1m"):
    """
    Grabs the current minute's stock data and prepares features for prediction.
    
    Parameters:
        ticker (str): Stock ticker symbol.
        interval (str): Data interval (e.g., '1m', '2m').
        
    Returns:
        list: Features [Open, High, Low, Close, Volume] for the current minute.
    """
    stock = yf.Ticker(ticker)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)  # Just the most recent day's data
    data = stock.history(interval=interval, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

    if data.empty:
        raise ValueError(f"No data fetched for ticker {ticker} at interval {interval}.")

    # Get the most recent minute of data (current minute)
    latest_data = data.iloc[-1]  # The most recent minute of data

    features = {
        "Open": latest_data['Open'],
        "High": latest_data['High'],
        "Low": latest_data['Low'],
        "Close": latest_data['Close'],
        "Volume": latest_data['Volume']
    }
    
    return features

def convert_np_objects(obj):
    """Recursively converts numpy objects to standard Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Converts numpy arrays to regular lists
    elif isinstance(obj, np.generic):
        return obj.item()  # Converts numpy scalar types (e.g., float32) to native Python types
    elif isinstance(obj, dict):
        # Apply recursively to dictionary
        return {key: convert_np_objects(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        # Apply recursively to list
        return [convert_np_objects(item) for item in obj]
    return obj

def stock_prediction(ticker):
    """
    Main function to fetch stock data, train model, make predictions, and display results.
    
    Parameters:
        ticker (str): The stock ticker symbol.
        
    Returns:
        dict: A dictionary containing the results of the stock prediction.
    """
    try:
        # Step 1: Fetch and prepare minute-level data
        X, y, data = fetch_minute_data(ticker)

        # Step 2: Train the model
        booster, mse = train_xgboost_model(X, y)

        # Step 3: Predict using the trained model
        predictions = predict_with_model(booster, X)

        # Prepare the results
        results = {
            "ticker": ticker,
            "predictions": [],
            "mse": mse,
            "current_prediction": None
        }

        # Add predictions data
        for datetime_, actual, predicted in zip(data.index[:5], y[:5], predictions[:5]):
            results["predictions"].append({
                "datetime": datetime_.strftime('%Y-%m-%d %H:%M:%S'),
                "actual_close": actual,
                "predicted_close": predicted
            })

        # Add the MSE
        results["mse"] = mse

        # Step 4: Predict close price for current time
        current_features = get_current_time_features(ticker)
        current_close_prediction = predict_with_model(booster, [list(current_features.values())])[0]

        results["current_prediction"] = {
            "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "predicted_close": current_close_prediction,
            "features": current_features
        }

        # Convert numpy objects to Python types (for JSON serialization)
        results = convert_np_objects(results)

        # Return results as JSON-friendly dictionary
        return json.dumps(results, indent=4)

    except Exception as e:
        return json.dumps({"error": str(e)}, indent=4)



