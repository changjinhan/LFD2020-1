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
    dates = pd.date_range(start_date, end_date, freq='B')
    df = pd.DataFrame(index=dates)

    if 'Gold' not in symbols:
        symbols.insert(0, 'Gold')

    for symbol in symbols:
        df_temp = pd.read_csv(get_data_path(symbol), index_col="Date", parse_dates=True, na_values=['nan'])
        df_temp.columns = [symbol + '_' + col for col in df_temp.columns]  # rename columns
        df = df.join(df_temp)

    return df


def make_features(start_date, end_date, is_training):
    # TODO: select symbols
    # commodity : BrentOil, Copper, CrudeOil, Gasoline, Gold, NaturalGas, Platinum, Silver
    # currency : AUD, CNY, EUR, GBP, HKD, JPY, USD
    table = merge_data(start_date, end_date, symbols=['Gold'])

    # TODO: cleaning or filling missing value
    table.fillna(method='ffill', inplace=True)
    table.fillna(method='bfill', inplace=True)

    # TODO:  select columns to use
    gold_high = table['Gold_High']
    gold_open = table['Gold_Open']
    gold_price = table['Gold_Price']

    # TODO:  make features
    gold_price = gold_price[1:]
    d_gold_h = np.diff(gold_high)
    d_gold_o = np.diff(gold_open)
    d_gold_p = np.diff(gold_price)

    input_days = 1
    training_sets = list()
    for time in range(len(gold_price)-input_days):
        d_g_h = d_gold_h[time:time + input_days]
        d_g_o = d_gold_o[time:time + input_days]
        d_g_p = d_gold_p[time:time + input_days]

        daily_feature = np.concatenate((d_g_h[::-1],d_g_o[::-1],d_g_p[::-1]))
        training_sets.append(daily_feature)

    training_x = training_sets[:-10]
    test_x = training_sets[-10:]

    scaler = StandardScaler()
    training_x = scaler.fit_transform(training_x[:2674])
    test_x = scaler.transform(test_x)

    past_price = gold_price[-11:-1]
    target_price = gold_price[-10:]
    return training_x if is_training else (test_x, past_price, target_price)


if __name__ == "__main__":
    start_date = '2010-01-01'
    end_date = '2020-04-18'

    make_features(start_date, end_date, is_training=False)