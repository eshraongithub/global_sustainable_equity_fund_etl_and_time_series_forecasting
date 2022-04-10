## Bollinger Bands

import timeit
start_time = timeit.default_timer()

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
import io

def get_num_lines(fname):
    with open(fname) as f:
        for i, _ in enumerate(f):
            pass
    return i + 1

num_lines = get_num_lines("gsef_folder/historical_gsef.csv")

n = 40
df = pd.read_csv("gsef_folder/historical_gsef.csv", skiprows=range(1, num_lines-n), parse_dates=['Date'], usecols= ["Date", "Close"], dayfirst=True)

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

if last_down_bollinger < last_closing_price:
    to_do= 'Do not buy today.'
elif last_up_bollinger < last_closing_price:
    to_do= 'Sell today.'
else:
    to_do= 'Buy today.'
    
from matplotlib.pyplot import figure

figure(figsize=(12, 8), dpi=80)

plt.title(' Bollinger Bands')
plt.xlabel('Days')
plt.ylabel('Closing Prices')
plt.plot(closing_prices, label='Closing Prices')
plt.plot(bollinger_up, label='Bollinger Up', c='k')
plt.plot(bollinger_down, label='Bollinger Down', c='m')
plt.legend()

#adding text inside the plot
plt.text(0, 540, to_do, fontsize = 22, c='g')

bollinger_bands_chart= 'gsef_folder/bollinger_bands.png'

import os
if os.path.exists(bollinger_bands_chart):
  os.remove(bollinger_bands_chart)
else:
  print("The Bollinger Bands chart didn't exist and it has been created.")

plt.savefig('gsef_folder/bollinger_bands.png', dpi=100)

#plt.show()

# Calculate and print the time elapsed to run ETL process and the date and time of the latest run
time_elapsed = timeit.default_timer() - start_time

time_elapsed_minutes = int((time_elapsed % 3600) // 60)

time_elapsed_seconds = int(time_elapsed % 60)

now = datetime.datetime.now()

print("\n Success! Your Bollinger Bands chart has been updated in {} minutes and {} seconds on {}".format(time_elapsed_minutes, time_elapsed_seconds, now.strftime("%Y-%m-%d %H:%M:%S")))