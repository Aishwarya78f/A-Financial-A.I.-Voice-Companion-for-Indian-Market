import yfinance as yf
import pandas as pd 
import datetime as dt 

def get_weekly_performance(tickers, start_date, end_date):
    weekly_performance = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            df['SMA_150'] = df['Adj Close'].rolling(window=150).mean()
            df['SMA_200'] = df['Adj Close'].rolling(window=200).mean()
            df['52_Week_Low'] = df['Low'].rolling(window=52*5).min()
            df['52_Week_High'] = df['High'].rolling(window=52*5).max()

            latest_price = df['Adj Close'][-1]
            pe_ratio = float(yf.Ticker(ticker).info.get('forwardPE', 'N/A'))
            peg_ratio = float(yf.Ticker(ticker).info.get('pegRatio', 'N/A'))
            moving_average_150 = df['SMA_150'][-1]
            moving_average_200 = df['SMA_200'][-1]
            low_52week = min(df['52_Week_Low'][-(52*5):])
            high_52week = max(df['52_Week_High'][-(52*5):])

            # Calculating weekly performance based on provided parameters
            weekly_performance[ticker] = {
                'Performance': latest_price / moving_average_150 * low_52week / high_52week * (1 / (pe_ratio * peg_ratio)),
                'Latest_Price': latest_price,
                'PE_Ratio': pe_ratio,
                'PEG_Ratio': peg_ratio,
                'SMA_150': moving_average_150,
                'SMA_200': moving_average_200,
                '52_Week_Low': low_52week,
                '52_Week_High': high_52week
            }
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return weekly_performance

if __name__ == "__main__":
    # Get tickers of 50 Indian stocks
    indian_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'HINDUNILVR.NS', 'INFY.NS', 'HDFC.NS', 'ICICIBANK.NS', 
                     'KOTAKBANK.NS', 'ITC.NS', 'LT.NS', 'SBIN.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'MARUTI.NS', 
                     'ASIANPAINT.NS', 'NTPC.NS', 'WIPRO.NS', 'SUNPHARMA.NS', 'AXISBANK.NS', 'POWERGRID.NS', 
                     'ULTRACEMCO.NS', 'TITAN.NS', 'ONGC.NS', 'COALINDIA.NS', 'IOC.NS', 'JSWSTEEL.NS', 
                     'BAJAJFINSV.NS', 'SHREECEM.NS', 'HCLTECH.NS', 'BPCL.NS', 'INDUSINDBK.NS', 'NESTLEIND.NS', 
                     'DRREDDY.NS', 'DIVISLAB.NS', 'HEROMOTOCO.NS', 'GRASIM.NS', 'UBL.NS', 'CIPLA.NS', 'TATASTEEL.NS', 
                     'ADANIPORTS.NS', 'TECHM.NS', 'BAJAJ-AUTO.NS', 'BRITANNIA.NS', 'M&M.NS', 'HINDALCO.NS', 
                     'GAIL.NS', 'EICHERMOT.NS', 'HDFCLIFE.NS', 'IOC.NS']

    # Define start and end dates for fetching data
    end_date = dt.datetime.now().strftime('%Y-%m-%d')
    start_date = (dt.datetime.now() - dt.timedelta(days=365)).strftime('%Y-%m-%d')  # Fetch data for the last year

    # Get weekly performance for Indian stocks
    weekly_performance = get_weekly_performance(indian_stocks, start_date, end_date)

    # Sort stocks based on weekly performance
    sorted_performance = sorted(weekly_performance.items(), key=lambda x: x[1]['Performance'], reverse=True)

    # Print top 10 performers
    print("Top 10 performers based on weekly performance:")
    for i, (ticker, performance) in enumerate(sorted_performance[:10], 1):
        print(f"{i}. {ticker}: {performance['Performance']}")
