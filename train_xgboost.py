import xgboost as xgb
from sklearn.metrics import mean_squared_error

def train_xgboost_model(X, y, num_boost_round=100):
    """
    Trains an XGBoost model using the provided features and target.
    
    Parameters:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        num_boost_round (int): Number of boosting rounds for training.
        
    Returns:
        tuple: Trained XGBoost booster and Mean Squared Error on training data.
    """
    # Convert to DMatrix format for XGBoost
    dtrain = xgb.DMatrix(X, label=y)

    # Define parameters for XGBoost
    params = {
        "max_depth": 4,
        "learning_rate": 0.1,
        "objective": "reg:squarederror"
    }

    # Train the model
    booster = xgb.train(params, dtrain, num_boost_round=num_boost_round)

    # Predict on training data to calculate MSE
    train_preds = booster.predict(dtrain)
    mse = mean_squared_error(y, train_preds)

    return booster, mse
