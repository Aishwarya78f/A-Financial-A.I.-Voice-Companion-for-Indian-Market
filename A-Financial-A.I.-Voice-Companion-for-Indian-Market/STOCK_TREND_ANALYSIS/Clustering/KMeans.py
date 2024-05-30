import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def retrieve_stock_data(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date)
        return data
    except Exception as e:
        print("Error occurred while retrieving stock data:", e)
        return None

def preprocess_data(data):
    if data is None:
        return None
    # Handling missing values if any
    data.dropna(inplace=True)
    return data

def cluster_stocks(data, clusters):
    if data is None:
        return None
    open_values = np.array(data['Open'].T)
    close_values = np.array(data['Close'].T)
    daily_movements = close_values - open_values
    scaler = StandardScaler()
    daily_movements_scaled = scaler.fit_transform(daily_movements)
    clustering_model = KMeans(n_clusters=clusters, max_iter=1000)
    clustering_model.fit(daily_movements_scaled)
    labels = clustering_model.labels_
    return labels

def main():
    # Input tickers from the user
    user_tickers = input("Enter tickers separated by spaces (e.g., AAPL NVDA TSLA): ").strip()
    if user_tickers:
        company_tickers = user_tickers.split()
    else:
        company_tickers = ['AAPL', 'NVDA', 'TSLA',  'ABBV', 'MCD', 'CCL', 'MSFT', 'GS', 'JPM']
    
    # Input number of clusters from the user
    user_clusters = input("Enter the number of clusters (default is 5): ").strip() 
    if user_clusters:
        clusters = int(user_clusters)
    else:
        clusters = 5
    
    start_date = dt.datetime.now() - dt.timedelta(days=365*2)
    end_date = dt.datetime.now()
    
    # Retrieve stock data
    stock_data = retrieve_stock_data(company_tickers, start_date, end_date)
    # Preprocess data
    preprocessed_data = preprocess_data(stock_data)
    # Cluster stocks
    labels = cluster_stocks(preprocessed_data, clusters)
    if labels is not None:
        results = pd.DataFrame({'clusters': labels, 'tickers': company_tickers})
        print("Stocks in each cluster (Clustered based on daily movements of stock prices):")
        for cluster_id in range(clusters):
            stocks_in_cluster = results[results['clusters'] == cluster_id]['tickers'].tolist()
            print(f"Cluster {cluster_id + 1}: {', '.join(stocks_in_cluster)}")
        # Visualize clusters
        # reduce dimensionality for visualization (for example using PCA)
        # visualize_clusters(reduced_data, labels)

if __name__ == "__main__":
    main()
