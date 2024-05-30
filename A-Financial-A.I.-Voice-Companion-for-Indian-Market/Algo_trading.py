import datetime as dt
import matplotlib.pyplot as plt
import yfinance as yf

ma_1 = 30
ma_2 = 100

# Specify the start and end dates
start = dt.datetime.now() - dt.timedelta(days=365 * 3)
end = dt.datetime.now()

# Fetch data using yfinance
data = yf.download('AAPL', start=start, end=end)

# Calculate moving averages
data[f'SMA_{ma_1}'] = data['Adj Close'].rolling(window=ma_1).mean()
data[f'SMA_{ma_2}'] = data['Adj Close'].rolling(window=ma_2).mean()
data = data.iloc[ma_2:]

# Plot the data
plt.plot(data['Adj Close'], label="Share Price", color="lightgrey")
plt.plot(data[f'SMA_{ma_1}'], label=f"SMA_{ma_1}", color="orange")
plt.plot(data[f'SMA_{ma_2}'], label=f"SMA_{ma_2}", color="purple")
plt.legend(loc="upper left")


buy_signal = []
sell_signal = []
trigger = 0
for x in range(len(data)):
    if data[f'SMA_{ma_1}'].iloc[x]>data[f'SMA_{ma_2}'].iloc[x] and trigger != 1:
        buy_signal.append(data['Adj Close'].iloc[x])
        sell_signal.append(float('nan'))
        trigger = 1
    elif data[f'SMA_{ma_1}'].iloc[x]<data[f'SMA_{ma_2}'].iloc[x] and trigger != -1 :
        buy_signal.append(float('nan'))
        sell_signal.append(data['Adj Close'].iloc[x])
        trigger =-1
    else:
        buy_signal.append(float('nan'))
        sell_signal.append(float('nan'))

data['Buy Signals']= buy_signal
data['Sell Signals']= sell_signal

print(data)
plt.plot(data['Adj Close'], label="Share Price", alpha = 0.5)
plt.plot(data[f'SMA_{ma_1}'], label=f"SMA_{ma_1}", color="orange",linestyle = "--")
plt.plot(data[f'SMA_{ma_2}'], label=f"SMA_{ma_2}", color="purple",linestyle = "--")
plt.scatter(data.index,data['Buy Signals'],label ="Buy Signal",marker = "^",color = "#00ff00",lw=3)
plt.scatter(data.index,data['Sell Signals'],label ="Sell Signal",marker = "v",color = "#ff0000",lw=3)
plt.legend(loc = "upper left")
plt.show()
