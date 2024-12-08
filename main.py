import pandas as pd
import xgboost as xgb
from train_xgboost import train_xgboost_model  # Assuming this is in the separate script
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import yfinance as yf
def process_data(target, *arrays):
    """
    This function takes a target array and any number of feature arrays.
    It combines them into a DataFrame and returns the combined data.
    
    Parameters:
        target (array-like): The target array.
        *arrays (array-like): Any number of feature arrays.
        
    Returns:
        pd.DataFrame: A DataFrame combining the target and feature arrays.
    """

    data =  list(arrays) + [target]
    
    columns = [f'feature_{i+1}' for i in range(len(arrays))] + ['target']

    df = pd.DataFrame(np.column_stack(data), columns=columns)
    
    return df

# True target values for MSE calculation
target = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

# Larger training data with more variety
train_data = process_data(target, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],[2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

params = {
    "max_depth": 6,         # Deeper trees to capture more complex patterns
    "learning_rate": 0.05,  # Lower learning rate to allow more boosting rounds
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "subsample": 0.8,       # Randomly sample training data to reduce overfitting
    "colsample_bytree": 0.8 # Randomly sample features for each tree
}

booster = train_xgboost_model(train_data, params=params, num_boost_round=100)

# FIND MSE
true_values_train = train_data["target"].values  # True values from the training data

# Convert the training data into a DMatrix
dtrain = xgb.DMatrix(data=train_data.iloc[:, :-1])

# Make predictions using the trained booster on the training set
train_predictions = booster.predict(dtrain)

# Calculate Mean Squared Error (MSE) for the training data
train_mse = mean_squared_error(true_values_train, train_predictions)

# Calculate the standard deviation of predictions for confidence interval
train_std_dev = np.std(train_predictions)

# Assuming a normal distribution for the predictions, calculate 95% confidence interval for training predictions
train_confidence_interval = norm.interval(0.95, loc=np.mean(train_predictions), scale=train_std_dev)

#print("Predictions on training data:", train_predictions)
print("Mean Squared Error (MSE) on training data:", train_mse)
print(f"95% Confidence Interval on training data: {train_confidence_interval}")











# Test data (new dataset)
test_data = pd.DataFrame({
    "feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "feature_2": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
})

# Convert the test data into a DMatrix
dtest = xgb.DMatrix(data=test_data)

# Make predictions using the trained booster
predictions = booster.predict(dtest)

print("Predictions on new data:", predictions)


# import matplotlib.pyplot as plt
# plt.scatter(train_data["feature1"], train_data["target"], label="Feature 1")
# plt.scatter(train_data["feature2"], train_data["target"], label="Feature 2")
# plt.xlabel("Feature Value")
# plt.ylabel("Target")
# plt.legend()
# plt.show()


def prepare_stock_data(ticker, period="1y"):
    """
    Fetches historical stock data using yfinance and prepares it for model training.
    
    Parameters:
        ticker (str): The stock ticker symbol.
        period (str): The period for which to fetch data (e.g., '1y', '6mo').
        
    Returns:
        np.ndarray: Feature matrix (X) and target vector (y).
    """
    # Fetch historical data for the stock
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)

    # Use the 'Open', 'High', 'Low', 'Volume' as features
    X = data[['Open', 'High', 'Low', 'Volume']].values
    
    # Use the 'Close' price as the target
    y = data['Close'].values

    return X, y

# # Example usage: Fetch data for Apple (AAPL) for the past year
# ticker = "AAPL"
# X, y = prepare_stock_data(ticker)

# # Display the first few rows of features and target
# print("Features (X):", X[:5])  # First 5 rows of features
# print("Target (y):", y[:5])  # First 5 rows of target
