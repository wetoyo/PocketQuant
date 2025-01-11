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

    params = {
        "booster": "gbtree",           # Type of booster to use (tree model)
        "max_depth": 4,                # Maximum depth of trees
        "learning_rate": 0.05,         # Step size (learning rate)
        "objective": "reg:squarederror", # Task type (regression)
        "eval_metric": "rmse",         # Evaluation metric
        "subsample": 0.8,              # Fraction of samples used for each boosting round
        "colsample_bytree": 0.8,       # Fraction of features used for each tree
        "colsample_bylevel": 0.8,      # Fraction of features used for each level of tree
        "min_child_weight": 5,         # Minimum sum of instance weight needed in a child
        "lambda": 1,                   # L2 regularization
        "alpha": 0,                    # L1 regularization
    }
    booster = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    y_pred = booster.predict(dtrain)

    # Calculate Mean Squared Error (MSE) on training data
    mse = mean_squared_error(y, y_pred)
    
    # print(f"MSE is {mse}")

    return booster
