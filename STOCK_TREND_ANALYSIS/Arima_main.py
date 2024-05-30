import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMAResults
import joblib
import os
# Function to load the ARIMA model
def load_arima_model(ticker_symbol):
    
    model_file = "AAPL_arima_model.pkl"
    if os.path.exists(model_file):
        return joblib.load(model_file)
    else:
        print("Model file not found.")
        return None

# Function to fetch stock data from Yahoo Finance
def fetch_stock_data(ticker_symbol, start_date, end_date):
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    return data['Close'].dropna()

# Function to plot the stock data and predictions
def plot_stock_data(stock_data, predictions):
    fig, ax = plt.subplots()
    ax.plot(stock_data.index, stock_data.values, label='Actual Stock Price')
    ax.plot(stock_data.index, predictions, color='red', label='Predicted Stock Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.set_title('Stock Price Prediction')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Main function for Streamlit app
def main():
    st.title("Stock Price Prediction App")

    # Sidebar for user input
    st.sidebar.title("Input Parameters")
    ticker_symbol = st.sidebar.text_input("Enter Stock Ticker Symbol:", "AAPL")
    start_date = st.sidebar.date_input("Start Date:", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date:", pd.Timestamp.now())

    # Load the ARIMA model
    model = load_arima_model(ticker_symbol)
    if model is None:
        st.error("Model not found. Please train a model first.")
        return

    # Fetch stock data from Yahoo Finance
    st.write(f"Fetching stock data for {ticker_symbol}...")
    stock_data = fetch_stock_data(ticker_symbol, start_date, end_date)

    # Make predictions
    st.write("Making predictions...")
    predictions = model.forecast(steps=len(stock_data))

    # Plot the stock data and predictions
    plot_stock_data(stock_data, predictions)

if __name__ == "__main__":
    main()
