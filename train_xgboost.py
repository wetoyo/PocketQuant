import xgboost as xgb
import pandas as pd

def train_xgboost_model(dataframe, params=None, num_boost_round=10):
    """
    Trains an XGBoost model on the given dataset.
    
    Parameters:
    - dataframe (pd.DataFrame): The input table. The last column is assumed to be the target variable.
    - params (dict): Optional dictionary of hyperparameters for XGBoost. Defaults to a basic setup.
    - num_boost_round (int): Number of boosting rounds. Defaults to 10.
    
    Returns:
    - booster (xgb.Booster): The trained XGBoost model.
    """
    if dataframe.shape[0] < 2 or dataframe.shape[0] > 20:
        raise ValueError("The input dataset must have between 2 and 20 rows.")

    # Split into features (X) and target (y)
    X = dataframe.iloc[:, :-1]
    y = dataframe.iloc[:, -1]

    # Convert to DMatrix format required by XGBoost
    dtrain = xgb.DMatrix(data=X, label=y)

    # Set default parameters if none are provided
    if params is None:
        params = {
            "objective": "reg:squarederror",  # Default for regression, change as needed
            "max_depth": 6,
            "learning_rate": 0.1,
            "eval_metric": "rmse"  # Default evaluation metric
        }

    # Train the XGBoost model
    booster = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round)

    return booster
