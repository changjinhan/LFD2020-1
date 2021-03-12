from sklearn.metrics import mean_absolute_error
import numpy as np
import pickle
import DataGenerator
import pandas as pd
from DataGenerator import get_data_path


def get_past_price(start_date, end_date):
    """
    Do not fix this function
    """
    df = pd.read_csv(get_data_path('Gold'), index_col="Date", parse_dates=True, na_values=['nan'])
    price = df['Price'].loc[end_date: start_date][1:11][::-1]
    return price


def get_target_price(start_date, end_date):
    """
    Do not fix this function
    """
    df = pd.read_csv(get_data_path('Gold'), index_col="Date", parse_dates=True, na_values=['nan'])
    price = df['Price'].loc[end_date: start_date][:10][::-1]
    return price


def main():

    start_date = '2010-01-01'
    end_date = '2020-04-18'

    test_x, past_price, target_price = DataGenerator.make_features(start_date, end_date, is_training=False)

    ###################################################################################################################
    # inspect data
    assert past_price.tolist() == get_past_price(start_date, end_date).tolist(), 'your past price data is wrong!'
    assert target_price.tolist() == get_target_price(start_date, end_date).tolist(), 'your target price data is wrong!'
    ###################################################################################################################

    # TODO: fix pickle file name
    filename = 'team10_model.pkl'
    model = pickle.load(open(filename, 'rb'))
    print('load complete')
    print(model.get_params())

    hidden_states = model.predict(test_x)
    expected_diff_price = np.dot(model.transmat_, model.means_)
    diff = list(zip(*expected_diff_price))[0]

    predicted_price = list()
    for idx in range(10):  # predict gold price for 10 days
        state = hidden_states[idx]
        current_price = past_price[idx]
        next_day_price = current_price + diff[state]  # predicted gold price of next day

        predicted_price.append(next_day_price)

    predict = np.array(predicted_price)

    # print predicted_prices
    print('past price : {}'.format(np.array(past_price)))
    print('predicted price : {}'.format(predict))
    print('real price : {}'.format(np.array(target_price)))
    print()
    print('mae :', mean_absolute_error(target_price, predict))


if __name__ == '__main__':
    main()