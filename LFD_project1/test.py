import DataGenerator
import pickle
import numpy as np
import pandas as pd
from DataGenerator import get_data_path, return_y_inv_scaled
from sklearn.metrics import mean_absolute_error
import warnings


def get_test_dollar_price(start_date, end_date):
    """
    Do not fix this function
    """
    df = pd.read_csv(get_data_path('USD'), index_col="Date", parse_dates=True, na_values=['nan'])
    price = df['Price'].loc[end_date: start_date][:10][::-1]
    return price


def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    start_date = '2012-01-02'
    end_date = '2020-04-06'

    test_x, test_y = DataGenerator.make_features(start_date, end_date, is_training=False)


    ###################################################################################################################
    # inspect test data
    assert test_y.tolist() == get_test_dollar_price(start_date, end_date).tolist(), 'your test data is wrong!'
    ###################################################################################################################

    # TODO: fix pickle file name
    filename = 'team11_model.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))
    print('load complete')
    print(loaded_model.get_params())

    predict = loaded_model.predict([test_x])
    #  inverse scaled
    predict = return_y_inv_scaled(start_date, end_date, predict)
    print('mae: ', mean_absolute_error(np.reshape(predict, -1), test_y))


if __name__ == '__main__':
    main()
