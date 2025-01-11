from stock_prediction import stock_prediction, batch_test, fetch_and_align_data, comparison_function
from datetime import datetime, timedelta
import pytz 
# Example usage:

tickers = ["AAPL", "GOOG", "MSFT"]  # Example tickers
target_ticker = "AAPL"  # The stock to predict
eastern_timezone = pytz.timezone("Etc/GMT+5")
start = datetime(2025, 1, 6, 9, 49, 0)
end = datetime(2024, 12, 29, 9, 49, 0)
now = datetime.now()
# json_output = stock_prediction(tickers, target_ticker, num_future_minutes=4, end_date_param=now)
# print(json_output) 
#print(comparison_function(["AAPL", "GOOG", "MSFT"], "AAPL", datetime(2024, 12, 9, 9, 30), 10))
batch_test(tickers= tickers, target_ticker= target_ticker, date = start, runs = 50, interval= 2, plot_values= True , display_target_only= True)
#print(fetch_and_align_data(tickers,"1m", start, end ))
