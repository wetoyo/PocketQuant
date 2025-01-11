import yfinance as yf
import numpy as np
import xgboost as xgb  # Import xgboost for using the model
from datetime import datetime, timedelta
from train_xgboost import train_xgboost_model
from sklearn.metrics import mean_squared_error
import json
import pytz
import matplotlib.pyplot as plt
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
    temp_end = end_date + timedelta(days=1)
    # print(f"temp:{temp_end} // start:{start_date} // end:{end_date}")
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

    # print(aligned_data)
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
                start_date_param = datetime.now(eastern_timezone) - timedelta(days=1)
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
            last_known_features[0, 1] = max(last_known_features[0, 0], next_prediction * 1.001)
            last_known_features[0, 2] = min(last_known_features[0, 0], next_prediction * 0.999)
            # last_known_features[0, 4] *= 1 + np.random.uniform(-0.01, 0.01)
            
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
def comparison_function(tickers, target_ticker, comparison_date, predict_interval=10):
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

    # Run the stock prediction using the market start as the end time
    prediction_data = stock_prediction(tickers, target_ticker, num_future_minutes=predict_interval, 
                                       start_date_param=comparison_date - timedelta(days=5), end_date_param=comparison_date)
    
    # Parse the returned JSON result to get predictions
    predicted_data = json.loads(prediction_data)
    # print(predicted_data, comparison_date)
    if "future_predictions" in predicted_data:
        predicted_times = [entry['predicted_time'] for entry in predicted_data['future_predictions']]
        predicted_closes = [entry['predicted_close'] for entry in predicted_data['future_predictions']]
    else:
        print("No future predictions found in the response.")
        predicted_times = []
        predicted_closes = []

    # Fetch actual data for the next interval
    actual_data = fetch_and_align_data(tickers, interval="1m", start_date=comparison_date, end_date=comparison_date + timedelta(minutes=predict_interval -1))
    
    # Get the actual closing prices for the target ticker
    actual_closes = actual_data[f"Close_{target_ticker}"].values
    # print(actual_closes, predicted_closes)
    if len(predicted_closes) != len(actual_closes) or len(predicted_closes) != predict_interval:
        print("Mismatch in the number of predicted or actual data points. Skipping.")
        return [], [], []  # Return empty lists for mismatch
    
    # Return actual and predicted prices for plotting
    return actual_closes, predicted_closes, actual_data

def batch_test(tickers, target_ticker, date, runs=200, interval=10, scale_method="none", plot_values = False,display_target_only=True):
    """
    Performs batch testing for stock predictions, runs the comparison function 
    multiple times while ensuring the date is within the last 20 days.
    
    Parameters:
        tickers (list): List of stock tickers to be used for prediction.
        target_ticker (str): The ticker symbol for the stock to predict.
        date (datetime): The starting date for the comparison.
        scale_method (str): Method for scaling prices ('normalize', 'standardize', 'percentage').
    """
    
    # Ensure the provided date is within 20 days of the current date
    today = datetime.now(eastern_timezone)
    comparison_date = eastern_timezone.localize(date)
    max_date = today - timedelta(days=20)
    if comparison_date < max_date:
        raise ValueError(f"The provided date must be within the last 20 days. Max allowed date: {max_date.strftime('%Y-%m-%d')}")

    all_actual_prices = {ticker: [] for ticker in tickers}
    all_predicted_prices = []
    all_timestamps = []
    successful_queries = 0
    total_error = 0  # Track total error for successful queries
    
    # Run the comparison multiple times and aggregate the results
    for i in range(runs):
        # Fetch actual and predicted data for the target ticker
        actual_prices, predicted_prices, actual_data = comparison_function(tickers, target_ticker, comparison_date, interval)
        
        if actual_prices != [] and predicted_prices != []:  # Only aggregate if there is data
            all_predicted_prices.extend(predicted_prices)
            all_actual_prices[target_ticker].extend(actual_prices)
            all_timestamps.extend(actual_data.index.to_pydatetime())  # Collect timestamps for X-axis
            successful_queries += 1
            
            # Calculate the total error for this successful query
            total_error += sum(abs(a - p) for a, p in zip(actual_prices, predicted_prices))
        
            # Append the actual prices for all tickers from the returned actual_data
            for ticker in tickers:
                if f"Close_{ticker}" in actual_data and ticker != target_ticker:
                    all_actual_prices[ticker].extend(actual_data[f"Close_{ticker}"].values)

        comparison_date += timedelta(minutes=interval)  # Increment by interval
        
        # If comparison date goes past market close time, move to the next day
        if comparison_date.time() >= market_close_time:
            comparison_date += timedelta(days=1)  # Move to the next day
            comparison_date = comparison_date.replace(hour=9, minute=30, second=0, microsecond=0)
        if comparison_date >= datetime.now(eastern_timezone):
            print("Went past current day; No values to test with. Ending early.")
            break

    # Apply scaling to all prices
    if scale_method == "normalize":
        scaler = normalize_prices
    elif scale_method == "standardize":
        scaler = standardize_prices
    elif scale_method == "percentage":
        scaler = percentage_change
    elif scale_method == "none":
        # No scaling applied
        scaler = lambda x: x  # Identity function (returns the data as is)
    else:
        raise ValueError(f"Invalid scale method: {scale_method}")

    # Scale all actual prices and predicted prices
    
    for ticker in all_actual_prices:
        all_actual_prices[ticker] = scaler(np.array(all_actual_prices[ticker]))
    all_predicted_prices = scaler(np.array(all_predicted_prices))
    
    # Plot the results if there were any successful queries
    if successful_queries > 0 and plot_values:

        plot_results(all_actual_prices, all_predicted_prices, target_ticker, display_target_only)
    
        # Print the results
        print(f"Average MAE across {successful_queries} successful runs: {total_error / successful_queries:.3f}")
        print(f"Total number of successful queries: {successful_queries}/{runs}")
    else:
        print("No successful queries, unable to generate plot or calculate MAE.")

def percentage_change(prices):
    return [(p - prices[0]) / prices[0] * 100 for p in prices] if len(prices) > 1 else prices
def standardize_prices(prices):
    return (prices - np.mean(prices)) / np.std(prices) if len(prices) > 1 else prices
def normalize_prices(prices):
    return (prices - min(prices)) / (max(prices) - min(prices)) if len(prices) > 1 else prices
def plot_results(all_actual_prices, all_predicted_prices, target_ticker, display_target_only=True):
    """
    Plots the results of the stock predictions, allowing customization of what is displayed.
    
    Parameters:
        all_actual_prices (dict): Dictionary containing actual prices for each stock.
        all_predicted_prices (list): List containing predicted prices for the target ticker.
        target_ticker (str): The ticker symbol for the stock to predict.
        display_target_only (bool): Whether to display only the target ticker's prices. If False, plots all stocks.
    """
    plt.figure(figsize=(12, 8))

    # If display_target_only is True, only plot the target ticker's actual prices
    
    if display_target_only:
        plt.plot(all_actual_prices[target_ticker], label=f"Actual {target_ticker}", linestyle='solid')
        plt.plot(all_predicted_prices, label=f"Predicted {target_ticker}", color='red', linestyle='dashed', linewidth=2)
    else:
        # Plot actual prices for all stocks in the basket
        for ticker, prices in all_actual_prices.items():
            if prices != []:
                plt.plot(prices, label=f"Actual {ticker}", linestyle='solid')
        
        # Plot the predicted prices for the target ticker
        plt.plot(all_predicted_prices, label=f"Predicted {target_ticker}", color='red', linestyle='dashed', linewidth=2)

    plt.title(f"Scaled Actual vs Predicted Prices for {target_ticker} and Basket Stocks")
    plt.xlabel("Time (10 minute intervals)")
    plt.ylabel("Scaled Price (or Percentage Change)")
    plt.legend()
    plt.grid(True)
    plt.show()