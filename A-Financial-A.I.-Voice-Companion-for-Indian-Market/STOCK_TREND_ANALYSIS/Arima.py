import os
import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import joblib
from sklearn.metrics import mean_squared_error
import joblib

def train_arima_model(ticker_symbol, epochs):
    # Fetch data from Yahoo Finance
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=10)
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    # Take only the 'Close' prices for analysis
    data = data['Close'].dropna()

    # Fit ARIMA model
    p, d, q = 1, 1, 1  # Example: you can choose your own parameters
    model = ARIMA(data, order=(p, d, q))
    for _ in range(epochs):
        model_fit = model.fit()
    
    # Create folder to save models if it doesn't exist
    folder_name = "models"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Save the trained model
    model_file = os.path.join(folder_name, f"{ticker_symbol}_arima_model.pkl")
    joblib.dump(model_fit, model_file)
    print(f"Model trained and saved as {model_file}")

# Example usage
ticker_symbol = "AAPL"  # Example: Apple Inc. (AAPL)
epochs = 10  # Number of training epochs
train_arima_model(ticker_symbol, epochs)

def load_arima_model(ticker_symbol):
    folder_name = "models"
    model_file = os.path.join(folder_name, f"{ticker_symbol}_arima_model.pkl")
    if os.path.exists(model_file):
        return joblib.load(model_file)
    else:
        print("Model file not found.")
        return None

def test_arima_model(ticker_symbol):
    # Load the saved model
    model = load_arima_model(ticker_symbol)
    if model is None:
        return

    # Fetch data from Yahoo Finance for testing
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(months=1)
    test_data = yf.download(ticker_symbol, start=start_date, end=end_date)['Close'].dropna()

    # Make predictions
    predictions = model.forecast(steps=len(test_data))

    # Evaluate accuracy
    mse = mean_squared_error(test_data, predictions)
    rmse = mse ** 0.5

    # Print accuracy
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Example usage
ticker_symbol = "AAPL"  # Example: Apple Inc. (AAPL)
test_arima_model(ticker_symbol)