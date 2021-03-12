from DataGenerator import get_data_path, merge_data, windowing_x, windowing_y, return_y_inv_scaled
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn import ensemble
import pickle
from sklearn import neural_network as NN
from sklearn.metrics import mean_absolute_error
import warnings


random_state = 0
warnings.simplefilter(action='ignore', category=FutureWarning)

# preprocessing test
# changing start_date from 2010 to 2015
start_date = '2012-01-02' # 2010-01-01, 2011-01-03, 2012-01-02, 2013-01-01, 2014-01-01
end_date = '2020-04-06'
table = merge_data(start_date, end_date, symbols=['AUD', 'CNY', 'EUR', 'GBP', 'HKD', 'JPY', 'USD', 'BrentOil', 'Copper', 'CrudeOil', 'Gasoline', 'Gold', 'NaturalGas', 'Platinum', 'Silver'])

def inspect_missing(table):
    vars_with_missing = []

    for feature in table.columns:
        missings = len(table[table[feature].isnull()][feature])
        if missings > 0:
            vars_with_missing.append(feature)
            missings_perc = missings/table.shape[0]

            print('Variable {} has {} records ({:.2%}) with missing values'.format(feature, missings, missings_perc))

    print('In total, there are {} variables with missing values'.format(len(vars_with_missing)))


def make_features_search(table, feat_name, start_date, end_date, is_training):
    # TODO: select columns to use
    USD_price = table['USD_Price']

    # USD price는 제곱과 root 더해줌
    USD_price_square = table['USD_Price'] ** 2
    USD_price_root = table['USD_Price'] ** (1 / 2)
    features = pd.concat([table[feat_name], USD_price, USD_price_square, USD_price_root], axis=1)

    # normalize
    standard_scaler_x = StandardScaler()
    standard_scaler_y = StandardScaler()

    # TODO: make your features
    input_days = 5

    x = windowing_x(features, input_days)
    y = windowing_y(USD_price, input_days)

    # split training and test data
    training_x = x[:-10]
    standard_scaler_x.fit(training_x)
    scaled_training_x = standard_scaler_x.transform(training_x)

    training_y = y[:-10]
    standard_scaler_y.fit(training_y)

    scaled_training_y = standard_scaler_y.transform(training_y)

    test_x = x[-10]
    scaled_test_x = (standard_scaler_x.transform(test_x.reshape(1, -1))).reshape(-1)
    test_y = y[-10]

    return (scaled_training_x, scaled_training_y) if is_training else (scaled_test_x, test_y)

# na 처리 방법
# 1번 방법: drop rows
# table.dropna(inplace=True)

# 2번 방법: fillna with ffill and bfill for ffill nan
table.fillna(method='ffill', inplace=True)
table.fillna(method='bfill', inplace=True)

# 3번 방법: drop columns
# table.dropna(axis=1, inplace=True)

# volume features processing
for feature in table.columns:
    if 'Volume' in feature:
        table[feature].replace({'-' : np.nan}, inplace=True)
        for  i in table.index:
            if 'K' in str(table.loc[i, feature]):
                table.loc[i, feature] = float(str(table.loc[i, feature])[:-1])*1000
            elif 'M' in str(table.loc[i,feature]):
                table.loc[i, feature] = float(str(table.loc[i, feature])[:-1])*1000000

table.fillna(method='ffill', inplace=True)
table.fillna(method='bfill', inplace=True)
table.drop(columns=['Platinum_Volume'], inplace=True)


# Feature selection using ExtraTreeRegressor
# Normalization
train_set = table.iloc[:-10]
test_set = table.iloc[-10:]

standard_scaler = StandardScaler()
standard_scaler.fit(train_set)
scaled_train_set = standard_scaler.transform(train_set)
scaled_test_set = standard_scaler.transform(test_set)

train_set_df = pd.DataFrame(scaled_train_set, index=train_set.index, columns=train_set.columns)
test_set_df = pd.DataFrame(scaled_test_set, index=test_set.index, columns=test_set.columns)

scaled_table = pd.concat([train_set_df, test_set_df], axis=0)

train_y = train_set_df['USD_Price']
train_df  = train_set_df.drop(['USD_Price'], axis=1)
feat_names = train_df.columns.values

model = ensemble.ExtraTreesRegressor(n_estimators=25, max_depth=30, max_features=0.3, n_jobs=-1, random_state=random_state)
model.fit(train_df, train_y)

importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

topk = 5 # test topk for 1, 3, 5, 10, 20
indices = np.argsort(importances)[::-1][:topk]


# Training
training_x, training_y = make_features_search(table, feat_names[indices], start_date, end_date, is_training=True)

# search parameter with gridsearchcv
# from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# for topk in [1,3,5,10,20]:
#     indices = np.argsort(importances)[::-1][:topk]
#     # Training
#     training_x, training_y = make_features(table, feat_names[indices], start_date, end_date, is_training=True)

#     # TODO: set model parameters
#     parameters =  {'hidden_layer_sizes':[(64,16), (32, 8), (64,32)]}
#     model = NN.MLPRegressor(random_state=random_state, shuffle=False, max_iter=2000,
#                             early_stopping=True, activation='tanh', solver='sgd', learning_rate='adaptive')
#     tscv = TimeSeriesSplit(n_splits=5)
#     gs = GridSearchCV(model, parameters, scoring='neg_mean_absolute_error', cv=tscv)
#     gs.fit(training_x, training_y)
#     print('final params', gs.best_params_)   # 최적의 파라미터 값 출력
#     print('best score', gs.best_score_)      # 최고의 점수

#     model = gs.best_estimator_

# TODO: set model parameters
model = NN.MLPRegressor(random_state=random_state, shuffle=False, max_iter=2000,
                        early_stopping=True, activation='tanh', solver='sgd', learning_rate='adaptive',
                       hidden_layer_sizes=(64,32))


model.fit(training_x, training_y)

# TODO: fix pickle file name
# method = 'ETR'
# # filename = 'team11_%02d%02d_%02d%02d_%s.pkl' % (tuple(localtime(time()))[1:5] + tuple([method]))
# filename = 'team11_%s.pkl' % (method)
# pickle.dump(model, open(filename, 'wb'))
#
training_predict = model.predict(training_x)
training_predict = return_y_inv_scaled(start_date, end_date, training_predict)
training_y = return_y_inv_scaled(start_date, end_date, training_y)

print(training_predict.shape, training_y.shape )
print('train mae: ', mean_absolute_error(training_predict, training_y))

# Test
def get_test_dollar_price(start_date, end_date):
    """
    Do not fix this function
    """
    df = pd.read_csv(get_data_path('USD'), index_col="Date", parse_dates=True, na_values=['nan'])
    price = df['Price'].loc[end_date: start_date][:10][::-1]
    return price

test_x, test_y = make_features_search(table, feat_names[indices], start_date, end_date, is_training=False)
###################################################################################################################
# inspect test data
assert test_y.tolist() == get_test_dollar_price(start_date, end_date).tolist(), 'your test data is wrong!'
###################################################################################################################

# TODO: fix pickle file name
method = 'ETR'
# filename = 'team11_%02d%02d_%02d%02d_%s.pkl' % (tuple(localtime(time()))[1:5] + tuple([method]))
# filename = 'team11_%s.pkl' % (method)
# loaded_model = pickle.load(open(filename, 'rb'))
# print(loaded_model.get_params())

print(model.get_params())
predict = model.predict([test_x])
predict = return_y_inv_scaled(start_date, end_date, predict)

print('test mae: ', mean_absolute_error(np.reshape(predict, -1), test_y))
print("start_date ={}, FillNA, topk={}\n".format(start_date, topk))
#     print("start_date ={}, DROPNA_bycolumn, topk={}\n\n".format(start_date, topk))

result = pd.DataFrame({'predict': np.reshape(predict,-1), 'truth': test_y.values}, index=test_y.index)
print(result, '\n\n')