import yfinance as yf
import pandas as pd
from yahoo_fin import stock_info as si
import datetime as dt

# Get the list of Nifty 50 tickers
tickers = si.tickers_nifty50()

# Set the date range for historical data
start = dt.datetime.now() - dt.timedelta(days=365)
end = dt.datetime.now()

# Initialize lists to store returns and create an empty DataFrame for final results
return_list = []
final_df = pd.DataFrame(columns=['Ticker', 'Latest_Price', 'Score', 'PE_Ratio', 'PEG_Ratio', 'SMA_150', 'SMA_200', '52_Week_Low', '52_Week_High'])

# Iterate through Nifty 50 tickers
for ticker in tickers:
    try:
        # Download historical data for each ticker
        df = yf.download(ticker+'.NS', start=start, end=end)
        if df.empty:
            continue

        # Calculate cumulative return for the stock
        df['Pct Change'] = df['Adj Close'].pct_change()
        stock_return = (df['Pct Change'] + 1).cumprod()[-1]

        # Compare stock return with Nifty 50 return
        returns_compared = round((stock_return / 1), 2)  # Assuming Nifty 50 return is 1
        return_list.append(returns_compared)

        # Limit to top 10 tickers for demonstration
        if len(return_list) == 10:
            break
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")

# Create DataFrame with returns information
best_performers = pd.DataFrame(list(zip(tickers, return_list)), columns=['Ticker', 'Returns Compared'])
best_performers['Score'] = best_performers['Returns Compared'].rank(pct=True) * 100
best_performers = best_performers[best_performers['Score'] >= best_performers['Score'].quantile(0.7)]

# Analyze additional metrics for the best-performing stocks
for ticker in best_performers['Ticker']:
    try:
        df = pd.read_csv(f'stock_data/{ticker}.csv', index_col=0)
        if df.empty:
            continue

        # Calculate moving averages
        moving_averages = [150, 200]
        for ma in moving_averages:
            df['SMA_' + str(ma)] = round(df['Adj Close'].rolling(window=ma).mean(), 2)

        # Extract relevant metrics
        latest_price = df['Adj Close'][-1]
        pe_ratio = float(si.get_quote_table(ticker+'.NS')['PE Ratio (TTM)'])
        peg_ratio = float(si.get_stats_valuation(ticker+'.NS')[1][4])
        moving_average_150 = df['SMA_150'][-1]
        moving_average_200 = df['SMA_200'][-1]
        low_52week = round(min(df['Low'][-(52*5):]), 2)
        high_52week = round(max(df['High'][-(52*5):]), 2)
        score = round(best_performers[best_performers['Ticker'] == ticker]['Score'].tolist()[0])

        # Check conditions for stock selection
        condition_1 = latest_price > moving_average_150 > moving_average_200
        condition_2 = latest_price >= (1.3 * low_52week)
        condition_3 = latest_price >= (0.75 * high_52week)
        condition_4 = pe_ratio < 40
        condition_5 = peg_ratio < 2

        if condition_1 and condition_2 and condition_3 and condition_4 and condition_5:
            final_df = final_df.append({'Ticker': ticker,
                                        'Latest_Price': latest_price,
                                        'Score': score,
                                        'PE_Ratio': pe_ratio,
                                        'PEG_Ratio': peg_ratio,
                                        'SMA_150': moving_average_150,
                                        'SMA_200': moving_average_200,
                                        '52_Week_Low': low_52week,
                                        '52_Week_High': high_52week}, ignore_index=True)
    except Exception as e:
        print(f"Error analyzing data for {ticker}: {e}")

# Display the final DataFrame sorted by score
final_df.sort_values(by='Score', ascending=False)
