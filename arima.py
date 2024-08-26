from statsmodels.tsa.stattools import adfuller
import numpy as np
import sys
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
import requests
import datetime
import time
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import json
import itertools


def plot_forecast(df, forecast, forecast_dates, name):
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['Close'], label='Historical Data')
    plt.plot(forecast_dates, [df['Close'].iloc[-1]] + forecast, color='red', marker='o', linestyle='dashed', label='Forecast')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'{name} Stock Price Forecast')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def adf_test(series):
    result = adfuller(series, autolag='AIC')
    return result[1]  # p-value



def auto_arima_order(df, max_p=5, max_d=2, max_q=5):
    p = d = q = range(0, max_p)
    pdq = list(itertools.product(p, d, q))
    
    best_aic = float('inf')
    best_order = None

    for param in pdq:
        try:
            model = ARIMA(df, order=param)
            model_fit = model.fit()
            aic = model_fit.aic
            if aic < best_aic:
                best_aic = aic
                best_order = param
        except:
            continue
    return best_order



def get_stock_symbol(name):

    try:
        with open("/home/toy/pfa/python/stocks.txt","r") as file:
            lines = file.readlines()
    except Exception as e:
        print(f"Failed to write {name}.csv: {e}")
    for line in lines:
        if name.lower() in line.lower():
            parts = line.split(":")
            return parts[1].strip()

def date_to_unix_timestamp(date):
    return int(time.mktime(date.timetuple()))


def get_csv(name):

    # Get today's date and the date one year before
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365)
    
    # Convert to UNIX timestamps
    period1 = date_to_unix_timestamp(start_date)
    period2 = date_to_unix_timestamp(end_date)

    url = f'https://query1.finance.yahoo.com/v7/finance/download/{name}?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true'
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
    response = requests.get(url, headers = headers)
    if response.status_code == 200:
        try:
            with open(f"/home/toy/pfa/python/{name}.csv", 'wb') as file:
                file.write(response.content)
        except Exception as e:
            print(f"Failed to write {name}.csv: {e}")
    else:
        print(f"Failed to download {name}.csv: Status code {response.status_code}")

def arima(name):
    get_csv(name)

    df = pd.read_csv("/home/toy/pfa/python/"+name + ".csv")

    # Perform the ADF test to determine the level of differencing
   # if adf_test(df['Close']) > 0.05:
    #    order = auto_arima_order(df['Close'].diff().dropna())
    #else:
     #   order = auto_arima_order(df['Close'])
    
#    print(f"Best ARIMA order: {order}")

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.asfreq('D')


    df['Close'] = df['Close'].ffill()
    #order = auto_arima_order(df['Close'])

    #print(f"Best ARIMA order: {order}")

    model = ARIMA(df['Close'], order=(4, 1, 0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=10)

    forecast_dates = pd.date_range(start=df.index[-1], periods=11).strftime('%Y-%m-%d').tolist()
    
    result = {
        'forecast': forecast.tolist(),
        'forecast_dates': forecast_dates
    }

    #plot_forecast(df, forecast, forecast_dates, name)

    return json.dumps(result)

def main(stock_name):
    name = get_stock_symbol(stock_name)
    result = arima(name)
    print(result)

if __name__ == "__main__":
    name = sys.argv[1]
    main(name)

