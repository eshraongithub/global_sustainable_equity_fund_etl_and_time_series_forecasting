import timeit
start_time = timeit.default_timer()

## Import libraries
import time
import datetime
import pandas as pd
import io
import matplotlib.pyplot as plt

pd.set_option('display.max_column',None)
tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).isoformat() #get tomorrow in iso format as needed'''
url = pd.read_html("https://markets.ft.com/data/funds/tearsheet/historical?s=LU0119216553:EUR", header=0)
table = url[0]
gsef_latest = table[['Date', 'Close']]
gsef_latest_selected= gsef_latest.copy()

gsef_latest_selected['Date'] = gsef_latest_selected.loc[:,'Date'].str[-12:]
gsef_latest_selected['Date'] = pd.to_datetime(gsef_latest_selected['Date'])

gsef_historical= pd.read_csv('gsef_folder/historical_gsef.csv', parse_dates=['Date'])

gsef = pd.concat([gsef_latest_selected, gsef_historical], ignore_index=True).sort_values(by="Date")

gsef = gsef.reset_index(drop=True)

gsef['pct_change']= (gsef["Close"].pct_change()*100).round(2)

gsef = gsef.drop_duplicates('Date')

gsef.to_csv('gsef_folder/historical_gsef.csv', index=False)

gsef.plot(x= 'Date', y='Close', figsize=(10,10), title= 'GSEF Price by Date', legend=False)

#plt.show()

historical_chart= 'gsef_folder/gsef_historical.png'

import os
if os.path.exists(historical_chart):
  os.remove(historical_chart)
else:
  print("The graph didn't exist and it has been created.")

plt.savefig('gsef_folder/gsef_historical.png')

# GSEF Prediction
## Imports & data loading
import pmdarima as pm
from pmdarima.model_selection import train_test_split
import numpy as np

gsef_selected= gsef[['Date','Close']]

y = gsef_selected['Close'].values

## Data splitting
train_data, test_data = gsef_selected[0:int(len(gsef_selected)*0.8)], gsef_selected[int(len(gsef_selected)*0.8):]

# Load/split your data
y_train, y_test = train_test_split(y, train_size= len(train_data))

## Pre-modeling analysis
from pandas.plotting import lag_plot

fig, axes = plt.subplots(3, 2, figsize=(12, 18))
plt.title('GSEF Autocorrelation plot')

# The axis coordinates for the plots
ax_idcs = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
    (2, 0),
    (2, 1)
]

for lag, ax_coords in enumerate(ax_idcs, 1):
    ax_row, ax_col = ax_coords
    axis = axes[ax_row][ax_col]
    lag_plot(gsef_selected['Close'], lag=lag, ax=axis)
    axis.set_title(f"Lag={lag}")

#plt.show()

## Estimating the differencing term
from pmdarima.arima import ndiffs

kpss_diffs = ndiffs(y_train, alpha=0.05, test='kpss', max_d=6)
adf_diffs = ndiffs(y_train, alpha=0.05, test='adf', max_d=6)
n_diffs = max(adf_diffs, kpss_diffs)

print(f"Estimated differencing term: {n_diffs}")
# Estimated differencing term: 1

## Fitting our model
auto = pm.auto_arima(y_train, d=n_diffs, seasonal=True, stepwise=True,
                     suppress_warnings=True, error_action="ignore", max_p=6,
                     max_order=None, trace=True)

## Updating the model
from sklearn.metrics import mean_squared_error
from pmdarima.metrics import smape

model = auto  # seeded from the model we've already fit

def forecast_one_step():
    fc, conf_int = model.predict(n_periods=1, return_conf_int=True)
    return (
        fc.tolist()[0],
        np.asarray(conf_int).tolist()[0])

forecasts = []
confidence_intervals = []

for new_ob in y_test:
    fc, conf = forecast_one_step()
    forecasts.append(fc)
    confidence_intervals.append(conf)

    # Updates the existing model with a small number of MLE steps
    model.update(new_ob)

print(f"Mean squared error: {mean_squared_error(y_test, forecasts)}")
print(f"SMAPE: {smape(y_test, forecasts)}")
# Mean squared error: 28.692414878594857
# SMAPE: 0.8443963447520769

print(model.summary())

## Forecast the Price for the next 10 business days
# Forecast the Priece for the next 10 business days
prediction= model.predict(start= len(gsef_selected), end= len(gsef_selected)+10, type= 'levels')

following_day= gsef_selected.Date.iloc[-1]+ pd.DateOffset(1)

index_future_dates= pd.date_range(start= following_day, periods=10, freq='B')

prediction_df = pd.DataFrame(prediction, columns = ['Close'], index= pd.DatetimeIndex(index_future_dates)).reset_index(level=0)

prediction_df.rename(columns={'index': 'Date'}, inplace=True)

existing_with_prediction= pd.concat([gsef_selected, prediction_df], ignore_index=True, sort=False)

## Visualise and save the forecasts
plt.figure(figsize=(14, 14))

plt.plot(existing_with_prediction['Date'][-10:], existing_with_prediction['Close'][-10:], color='dodgerblue')

plt.ylabel('Price')

plt.title('GSEF 10 Day Forecast')

prediction_chart= 'gsef_folder/gsef_10_day_prediction.png'

import os
if os.path.exists(prediction_chart):
  os.remove(prediction_chart)
else:
  print("The prediction chart didn't exist and it has been created.")

plt.savefig('gsef_folder/gsef_10_day_prediction.png', dpi=100)

## Time Elapsed
# Calculate and print the time elapsed to run ETL process and the date and time of the latest run
time_elapsed = timeit.default_timer() - start_time

time_elapsed_minutes = int((time_elapsed % 3600) // 60)

time_elapsed_seconds = int(time_elapsed % 60)

now = datetime.datetime.now()

print("\n Success! Your GSEF data and forecast have been updated in {} minutes and {} seconds on {}".format(time_elapsed_minutes, time_elapsed_seconds, now.strftime("%Y-%m-%d %H:%M:%S")))