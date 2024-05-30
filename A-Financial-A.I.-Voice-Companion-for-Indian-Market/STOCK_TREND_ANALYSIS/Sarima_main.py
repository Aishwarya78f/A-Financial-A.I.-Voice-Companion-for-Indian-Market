import os
import yfinance as yf
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
from sklearn.metrics import mean_squared_error

def train_sarima_model(ticker_symbol, epochs):
    # Fetch data from Yahoo Finance
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=10)
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    # Take only the 'Close' prices for analysis
    data = data['Close'].dropna()

    # Fit SARIMA model
    p, d, q, P, D, Q, s = 1, 1, 1, 1, 1, 1, 12  # Example: you can choose your own parameters
    model = SARIMAX(data, order=(p, d, q), seasonal_order=(P, D, Q, s))
    for _ in range(epochs):
        model_fit = model.fit()

    # Create folder to save models if it doesn't exist
    folder_name = "models"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Save the trained model
    model_file = os.path.join(folder_name, f"{ticker_symbol}_sarima_model.pkl")
    joblib.dump(model_fit, model_file)
    print(f"Model trained and saved as {model_file}")

def load_sarima_model(ticker_symbol):
    folder_name = "models"
    model_file = os.path.join(folder_name, f"{ticker_symbol}_sarima_model.pkl")
    if os.path.exists(model_file):
        return joblib.load(model_file)
    else:
        print("Model file not found.")
        return None

def test_sarima_model(ticker_symbol):
    # Load the saved model
    model = load_sarima_model(ticker_symbol)
    if model is None:
        return

    # Fetch data from Yahoo Finance for testing
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(months=1)
    test_data = yf.download(ticker_symbol, start=start_date, end=end_date)['Close'].dropna()

    # Make predictions
    pred_start_date = test_data.index[0]  # Use the first date in the test data as the prediction start date
    pred_end_date = test_data.index[-1]  # Use the last date in the test data as the prediction end date
    predictions = model.predict(start=pred_start_date, end=pred_end_date, dynamic=True)

    # Evaluate accuracy
    mse = mean_squared_error(test_data, predictions)
    rmse = mse ** 0.5

    # Print accuracy
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Example usage for training
ticker_symbol = "AAPL"  # Example: Apple Inc. (AAPL)
epochs = 10  # Number of training epochs
train_sarima_model(ticker_symbol, epochs)

# Example usage for testing
ticker_symbol = "AAPL"  # Example: Apple Inc. (AAPL)
test_sarima_model(ticker_symbol)
