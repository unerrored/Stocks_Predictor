import pandas as pd
import mplfinance as mpf
import os

csv_file = 'candlestickchart/intraday_5min_IBM.csv'

data = pd.read_csv(csv_file)

data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)

data_for_plot = data[['open', 'high', 'low', 'close', 'volume']]
data_for_plot.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

output_png = os.path.splitext(csv_file)[0] + '_chartpng.png'

mpf.plot(data_for_plot, type='candle', volume=True, style='yahoo', title="IBM Stock Price (5min intervals)", 
         ylabel="Price", savefig=output_png)

print(f"Chart saved as: {output_png}")
