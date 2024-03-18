import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import time

api_key = 'B6WE192UVE8M9P0Q'

ts = TimeSeries(key=api_key, output_format='pandas')
data, meta_data = ts.get_daily(symbol='MSFT')
print(data)
data.to_csv('DailyStockPrices.csv')
