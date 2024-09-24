import pandas as pd
import mplfinance as mpf
import os

# File path for the CSV
csv_file = 'candlestickchart/intraday_5min_IBM.csv'

# Load CSV data
data = pd.read_csv(csv_file)

# Convert 'timestamp' to datetime and set as index
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)

# Prepare data for plotting (renaming columns for mplfinance)
data_for_plot = data[['open', 'high', 'low', 'close', 'volume']]
data_for_plot.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

# Generate the output PNG file name by modifying the CSV file name
output_png = os.path.splitext(csv_file)[0] + '_chartpng.png'

# Plot and save the candlestick chart with the customized filename
mpf.plot(data_for_plot, type='candle', volume=True, style='yahoo', title="IBM Stock Price (5min intervals)", 
         ylabel="Price", savefig=output_png)

print(f"Chart saved as: {output_png}")
