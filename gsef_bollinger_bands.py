## Bollinger Bands

import timeit
start_time = timeit.default_timer()

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
from datetime import timedelta
import io

def get_num_lines(fname):
    with open(fname) as f:
        for i, _ in enumerate(f):
            pass
    return i + 1

num_lines = get_num_lines("gsef_output/historical_gsef.csv")

n = 506 #load the data of 2 trading years (number of trading days per year is 253)
df = pd.read_csv("gsef_output/historical_gsef.csv", skiprows=range(1, num_lines-n), parse_dates=['Date'], usecols= ["Date", "Close"], dayfirst=True)

closing_prices = df['Close'] # Use only closing prices

def get_sma(prices, rate):
    return prices.rolling(rate).mean()

def get_bollinger_bands(prices, rate= 20):
    sma = get_sma(prices, rate)
    std = prices.rolling(rate).std()
    bollinger_up = sma + std * 2 # Calculate top band
    bollinger_down = sma - std * 2 # Calculate bottom band
    return bollinger_up, bollinger_down

df.index = np.arange(df.shape[0])

bollinger_up, bollinger_down = get_bollinger_bands(closing_prices)

last_down_bollinger= bollinger_down.iloc[-1]

last_closing_price= df.iloc[-1]['Close']

if last_closing_price > last_down_bollinger:
    to_do= 'Do not buy today.'
else:
    to_do= 'Consider Buying today.'
    
# Set series indexes as the data's date
closing_prices= closing_prices.set_axis(df.Date)
bollinger_up= bollinger_up.set_axis(df.Date)
bollinger_down= bollinger_down.set_axis(df.Date)
    
from matplotlib.pyplot import figure

figure(figsize=(12, 8), dpi=80)

plt.title('Bollinger Bands')
plt.xlabel('Days')
plt.ylabel('Closing Prices')
closing_prices.plot(label='Closing Prices')
bollinger_up.plot(label='Bollinger Up', c='k')
bollinger_down.plot(label='Bollinger Down', c='m')
plt.legend()

bollinger_bands_chart= 'gsef_output/bollinger_bands.png'

import os
if os.path.exists(bollinger_bands_chart):
  os.remove(bollinger_bands_chart)
else:
  print("The Bollinger Bands chart didn't exist and it has been created.")

#adding text inside the plot

today = datetime.date.today()
two_trading_years_ago = today - timedelta(days=506)


plt.text(two_trading_years_ago, 400, to_do, fontsize = 22, c='g')

plt.savefig('gsef_output/bollinger_bands.png', dpi=100)

#plt.show()

# Calculate and print the time elapsed to run ETL process and the date and time of the latest run
time_elapsed = timeit.default_timer() - start_time

time_elapsed_seconds = int(time_elapsed % 60)

now = datetime.datetime.now()

print("\n Success! Your Bollinger Bands chart has been updated in {} seconds on {}".format(time_elapsed_seconds, now.strftime("%Y-%m-%d %H:%M:%S")))