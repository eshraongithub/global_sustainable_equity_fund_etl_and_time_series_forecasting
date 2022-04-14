# global_sustainable_equity_fund_etl_and_time_series_forecasting
 
This project enables scraping and storing the data of the NN (L) Global Sustainable Equity fund as well as time series forecasting for the following 10 business days.

The "pmdarima" statistical library has been used for the time series analysis and forecast.

A python notebook + script have been added to perform each of the following analyses and processes separately:
- ETL from the Financial Times website.
- A 10 buisness days Time Series forecast.
- Bollinger Bands (20 days).

The required version of Python and packages are included in the "gsef_venv" virtual environment that can be found in the "envs" folder.