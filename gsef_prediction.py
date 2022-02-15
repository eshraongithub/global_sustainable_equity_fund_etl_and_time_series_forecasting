import time
start = time.time()

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

gsef = gsef.drop_duplicates()

gsef.to_csv('gsef_folder/historical_gsef.csv', index=False)

gsef.plot(x= 'Date', figsize=(10,10), title= 'GSEF Price by Date', legend=False)

#plt.show()

graph_file= 'gsef_folder/gsef_historical.png'

import os
if os.path.exists(graph_file):
  os.remove(graph_file)
else:
  print("The graph didn't exist and it has been created.")

plt.savefig('gsef_folder/gsef_historical.png')

end = time.time()
print('\n Success! Your GSEF data has been updated in', int(end - start), 'seconds.')