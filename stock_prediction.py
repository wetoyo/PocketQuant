import yfinance as yf
import numpy as np
import xgboost as xgb  # Import xgboost for using the model
from datetime import datetime, timedelta
from train_xgboost import train_xgboost_model
from sklearn.metrics import mean_squared_error
import json
import pytz
market_open_time = datetime.strptime('09:30:00', '%H:%M:%S').time()
market_close_time = datetime.strptime('16:00:00', '%H:%M:%S').time()
eastern_timezone = pytz.timezone("Etc/GMT+5")

def fetch_and_align_data(tickers, interval="1m", start_date=None, end_date=None):
    """
    Fetches and aligns minute-level stock data for multiple tickers.
    
    Parameters:
        tickers (list): List of stock ticker symbols.
        interval (str): Data interval (e.g., '1m', '2m').
        start_date (datetime): Start date for fetching data (should be in New York timezone).
        end_date (datetime): End date for fetching data (should be in New York timezone).
        
    Returns:
        DataFrame: Aggregated and aligned stock data with features for all tickers.
    """
    # Ensure start_date and end_date are provided as datetime objects, with timezone information
    if start_date is None:
        if end_date is None:
            start_date = datetime.now(eastern_timezone) - timedelta(days=5)
        else:
            start_date = end_date - timedelta(days=5)
    if end_date is None:
        end_date = datetime.now(eastern_timezone)
    
    # Make sure both start_date and end_date are timezone-aware (in 'America/New_York')
    start_date = start_date.astimezone(eastern_timezone)
    end_date = end_date.astimezone(eastern_timezone)

    temp_start = start_date
    temp_end = end_date
    # Ensure the start_date and end_date are in the correct range (across different days)
    if start_date.date() == end_date.date():
        # Modify the end_date to be the next day (just before midnight)
        temp_end = end_date + timedelta(days=1)
    if temp_end.time() >= market_close_time:
        temp_end = move_to_next_weekday(temp_end) + timedelta(hours=9, minutes=30) - timedelta(hours = 16)
    # Strip time components for Yahoo Finance API
    start_date_str = temp_start.strftime('%Y-%m-%d')
    end_date_str = temp_end.strftime('%Y-%m-%d')

    stock_data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        data = stock.history(interval=interval, start=start_date_str, end=end_date_str)
        if data.empty:
            print(f"No data fetched for ticker {ticker} between {start_date_str} and {end_date_str}.")
        stock_data[ticker] = data[['Open', 'High', 'Low', 'Close', 'Volume']].rename(
            columns=lambda col: f"{col}_{ticker}"
        )

    # Align data by index (datetime)
    aligned_data = stock_data[tickers[0]]
    for ticker in tickers[1:]:
        aligned_data = aligned_data.join(stock_data[ticker], how="inner")

    # Now trim the data to the original range (after modifying it)
    aligned_data = aligned_data[(aligned_data.index >= start_date) & (aligned_data.index <= end_date)]

    #print(aligned_data)
    return aligned_data

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
    return model.predict(dtest)

def move_to_next_weekday(date):
    """
    Moves the provided date to the next available weekday (Monday to Friday).
    """
    if date.weekday() == 4:  # Saturday
        return date + timedelta(days=3)  # Move to Monday
    elif date.weekday() == 5:  # Saturday
        return date + timedelta(days=2)  # Move to Monday
    elif date.weekday() == 6:  # Sunday
        return date + timedelta(days=1)  # Move to Monday
    return date

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

def stock_prediction(tickers, target_ticker, num_future_minutes=5, start_date_param=None, end_date_param=None):
    """
    Main function to fetch stock data, train model, make predictions, and display results.
    
    Parameters:
        tickers (list): List of stock ticker symbols.
        target_ticker (str): The stock ticker symbol to predict.
        num_future_minutes (int): Number of future minutes to predict.
        start_date_param (datetime): The start date for the stock data (default: current time in New York timezone).
        end_date_param (datetime): The end date for the stock data (default: current time in New York timezone).
        
    Returns:
        dict: A dictionary containing the results of the stock prediction.
    """
    try:
        # Set start_date_param to current datetime if it's not provided
        if start_date_param is None:
            if end_date_param is None:
                start_date_param = datetime.now(eastern_timezone) - timedelta(days=5)
            else:
                start_date_param = end_date_param - timedelta(days=5)
        if end_date_param is None:
            end_date_param = datetime.now(eastern_timezone)

        # Adjust dates if they fall on weekends or after market hours (4:00 PM)
        start_date_param = move_to_next_weekday(start_date_param)  # Adjust start date if on weekend
        end_date_param = move_to_next_weekday(end_date_param)  # Adjust end date if on weekend

        if end_date_param.hour > 16:  # After 4:00 PM (market close)
            # Move to the next available market day (9:30 AM)
            end_date_param = end_date_param.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(days=1)

        # Fetch and align data
        aligned_data = fetch_and_align_data(tickers, start_date=start_date_param, end_date=end_date_param)

        # Define target and feature matrix
        aligned_data[f"Target_{target_ticker}"] = aligned_data[f"Close_{target_ticker}"].shift(-1)
        X = aligned_data.drop(columns=[f"Target_{target_ticker}"]).values[:-1]
        y = aligned_data[f"Target_{target_ticker}"].dropna().values

        # Train the model
        booster = train_xgboost_model(X, y)

        # Predict for current data
        dtest = xgb.DMatrix(X)
        predictions = booster.predict(dtest)

        # Prepare the results
        results = {
            "target_ticker": target_ticker,
            "multi_stock_predictions": [],
            "future_predictions": []
        }

        # Add predictions for the recent past
        for datetime_, actual, predicted in zip(aligned_data.index[-5:], y[-5:], predictions[-5:]):
            results["multi_stock_predictions"].append({
                "datetime": datetime_.strftime('%Y-%m-%d %H:%M:%S'),
                "actual_close": actual,
                "predicted_close": predicted
            })

        # Predict future close prices
        last_known_features = X[-1:].copy()  # Start with the most recent row
        # Get the actual datetime corresponding to the last known feature
        last_known_datetime = aligned_data.index[-1]  # Last datetime in the aligned data

        # Initialize the prediction time as the datetime of the last known feature
        prediction_time = last_known_datetime

        for i in range(num_future_minutes):
            # Check if prediction time is past market close and adjust to the next valid market open time
            if  prediction_time.time() >= market_close_time:
                # Move to the next available market open time (9:30 AM on the next business day)
                prediction_time = move_to_next_weekday(prediction_time) + timedelta(hours=9, minutes=30) - timedelta(hours = 16)
            
            next_prediction = predict_with_model(booster, last_known_features)[0]
            
            # Update the features with the new predicted values
            last_known_features[0, 0] = last_known_features[0, -1]
            last_known_features[0, -1] = next_prediction  # Close = Predicted Close
            last_known_features[0, 1] = max(last_known_features[0, 0], next_prediction * 1.01)
            last_known_features[0, 2] = min(last_known_features[0, 0], next_prediction * 0.99)
            last_known_features[0, 4] *= 1 + np.random.uniform(-0.01, 0.01)
            
            # Increment the time for the next prediction by one minute
            prediction_time += timedelta(minutes=1)
            
            # Append the predicted data with the correct prediction time
            results["future_predictions"].append({
                "minute_offset": i + 1,
                "predicted_close": next_prediction,
                "predicted_time": prediction_time.strftime('%Y-%m-%d %H:%M:%S')  # Add predicted time
            })

        # Convert numpy objects to Python types (for JSON serialization)
        return json.dumps(convert_np_objects(results), indent=4)

    except Exception as e:
        return json.dumps({"error": str(e)}, indent=4)
def comparison_function(tickers, target_ticker, comparison_date):
    """
    Compares predictions with actual data for the next 10 minutes after market open.
    
    Parameters:
        tickers (list): List of stock tickers to be used for prediction.
        target_ticker (str): The ticker symbol for the stock to predict.
        comparison_date (datetime): The date for which to perform the comparison.
    """
    # Ensure the comparison date is in the correct timezone
    comparison_date = comparison_date.astimezone(eastern_timezone)
    comparison_date = move_to_next_weekday(comparison_date)

    
    if comparison_date.time() >= market_close_time:
        comparison_date += timedelta(days=1)  # Move to the next day
        comparison_date = comparison_date.replace(hour=9, minute=30, second=0, microsecond=0)
    # Set the start and end datetime for stock prediction (market open)
    market_start = comparison_date.replace(hour=9, minute=30, second=0, microsecond=0)
    market_end = comparison_date.replace(hour=16, minute=0, second=0, microsecond=0)

    # Run the stock prediction using the market start as the end time
    prediction_data = stock_prediction(tickers, target_ticker, num_future_minutes=10, start_date_param= comparison_date - timedelta(days=5), end_date_param=comparison_date)
    
    # Parse the returned JSON result to get predictions (skip parsing if you already have structured results)
    predicted_data = json.loads(prediction_data)  # Assuming it's a JSON string
    if "future_predictions" in predicted_data:
        predicted_times = [entry['predicted_time'] for entry in predicted_data['future_predictions']]
        predicted_closes = [entry['predicted_close'] for entry in predicted_data['future_predictions']]
    else:
        print("No future predictions found in the response.")
        predicted_times = []
        predicted_closes = []
    # Fetch actual data for the next 10 minutes after market open
    actual_data = fetch_and_align_data(tickers, interval="1m", start_date=comparison_date, end_date=comparison_date + timedelta(minutes=9))
    
    # Get the actual closing prices for the target ticker
    actual_closes = actual_data[f"Close_{target_ticker}"].values
    # print(predicted_closes)
    # print(actual_data)
    # Ensure we have exactly 10 predicted and 10 actual values
    if len(predicted_closes) != len(actual_closes) or len(predicted_closes) != 10:
        print("Mismatch in the number of predicted or actual data points. This usually means Yahoo Finance is missing a data point. Skipping.")
        return 0
    
    # Calculate and print the error for each prediction
    total_error = 0
    for i in range(10):
        actual = actual_closes[i]
        predicted = predicted_closes[i]
        error = abs(actual - predicted)  # You can use other error metrics like squared error
        total_error += error
        #print(f"Minute {i + 1}: Actual = {actual}, Predicted = {predicted}, Error = {error}")

    # Calculate mean absolute error (optional)
    mean_absolute_error = total_error / 10
    print(f"Mean Absolute Error (MAE) for next 10 minutes: {mean_absolute_error}. Total Absolute Error for the next 10 minutes: {total_error}. Time: {comparison_date}")
    return mean_absolute_error
# # # Example usage (replace `tickers` and `target_ticker` with real data)
comparison_date = datetime(2024, 11, 27, 14, 30, 0, tzinfo=eastern_timezone)  # The day you want to compare
mean = 0
runs = 200
sucess = runs
for i in range(runs):
    value = comparison_function(["AAPL", "GOOG"], "AAPL", comparison_date)
    if value != 0:
        mean += value
    else:
        sucess -=1
    comparison_date += timedelta(minutes=10)  # Increment by 10 minutes
    if comparison_date.time() >= market_close_time:
        comparison_date += timedelta(days=1)  # Move to the next day
        comparison_date = comparison_date.replace(hour=9, minute=30, second=0, microsecond=0)
print(f"Average MAE across runs: {mean/sucess}. There were {runs - sucess} / {runs} unsucessful queries.")