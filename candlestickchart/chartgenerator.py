import pandas as pd
import mplfinance as mpf

# Load CSV data
data = pd.read_csv('intraday_5min_IBM.csv')

# Convert 'timestamp' to datetime and set as index
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)

# Prepare data for plotting (renaming columns for mplfinance)
data_for_plot = data[['open', 'high', 'low', 'close', 'volume']]
data_for_plot.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

# Plot and save the candlestick chart as a PNG
mpf.plot(data_for_plot, type='candle', volume=True, style='yahoo', title="IBM Stock Price (5min intervals)", 
         ylabel="Price", savefig='candlestick_chart.png')
