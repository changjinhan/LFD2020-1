import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def get_data_path(symbol):
    # Return CSV file path given symbol.
    commodity_dir = '../data/commodities'
    currency_dir = '../data/currencies'

    if symbol in ['AUD', 'CNY', 'EUR', 'GBP', 'HKD', 'JPY', 'USD']:
        path = os.path.join(currency_dir, symbol + '.csv')
    else:
        path = os.path.join(commodity_dir, symbol + '.csv')

    return path


def merge_data(start_date, end_date, symbols):
    dates = pd.date_range(start_date, end_date, freq='B') # without Saturday, Sunday
    df = pd.DataFrame(index=dates)

    if 'USD' not in symbols:
        symbols.insert(0, 'USD')

    for symbol in symbols:
        df_temp = pd.read_csv(get_data_path(symbol), index_col="Date", parse_dates=True, na_values=['nan'])
        df_temp = df_temp.reindex(dates)
        df_temp.columns = [symbol + '_' + col for col in df_temp.columns]  # rename columns
        df = df.join(df_temp)

    return df


def make_features(start_date, end_date, is_training):
    # TODO: select symbols
    # commodity : BrentOil, Copper, CrudeOil, Gasoline, Gold, NaturalGas, Platinum, Silver
    # currency : AUD, CNY, EUR, GBP, HKD, JPY, USD
    table = merge_data(start_date, end_date, symbols=['USD','HKD'])

    # TODO: cleaning or filling missing value
    table.fillna(method='ffill', inplace=True)
    table.fillna(method='bfill', inplace=True)


    # TODO: select columns to use
    USD_price = table['USD_Price']
    USD_price_square = table['USD_Price']**2
    USD_price_root = table['USD_Price']**(1/2)
    other_features = table[['USD_High', 'HKD_Low', 'HKD_Price', 'USD_Open', 'USD_Low']]

    features = pd.concat([other_features, USD_price, USD_price_square, USD_price_root], axis=1)

    # TODO: make your features
    input_days = 5

    x = windowing_x(features, input_days)
    y = windowing_y(USD_price, input_days)


    # split training and test data
    training_x = x[:-10]
    training_y = y[:-10]
    test_x = x[-10]
    test_y = y[-10]

    # normalize
    standad_scaler_x = StandardScaler()
    standard_scaler_y = StandardScaler()

    standad_scaler_x.fit(training_x)
    scaled_training_x = standad_scaler_x.transform(training_x)
    scaled_test_x = standad_scaler_x.transform(test_x.reshape(1,-1)).reshape(-1)

    standard_scaler_y.fit(training_y)
    scaled_training_y = standard_scaler_y.transform(training_y)

    return (scaled_training_x, scaled_training_y) if is_training else (scaled_test_x, test_y)


def windowing_y(data, input_days):
    windows = [data[i + input_days:i + input_days + 10] for i in range(len(data) - input_days)]
    return windows


def windowing_x(data, input_days):
    windows = [np.reshape(np.array(data[i:i + input_days]), -1) for i in range(len(data) - input_days)]
    return windows


# return inverse scaled y
def return_y_inv_scaled(start_date, end_date, predict_y):
    table = merge_data(start_date, end_date, symbols=['USD'])
    USD_price = table['USD_Price']

    input_days = 5
    y = windowing_y(USD_price, input_days)
    training_y = y[:-10]

    standard_scaler_y = StandardScaler()
    standard_scaler_y.fit(training_y)

    predict_y = standard_scaler_y.inverse_transform(predict_y)

    return predict_y

if __name__ == "__main__":
    start_date = '2012-01-02'
    end_date = '2020-04-06'

    make_features(start_date, end_date, is_training=True)

