from stock_prediction import stock_prediction
from datetime import datetime, timedelta

# Example usage:

tickers = ["AAPL", "GOOG", "MSFT"]  # Example tickers
target_ticker = "AAPL"  # The stock to predict
start = datetime(2024, 12, 9, 9, 30, 0)
end = datetime(2024, 12, 9, 9, 40, 0)
json_output = stock_prediction(tickers, target_ticker, num_future_minutes=4, start_date_param=start, end_date_param=end)
print(json_output)