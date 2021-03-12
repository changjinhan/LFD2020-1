import os
import pandas as pd
import numpy as np
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
register_matplotlib_converters()

symbol_dict = {'cell': 'Celltrion',
               'hmotor': 'HyundaiMotor',
               'naver': 'NAVER',
               'kakao': 'Kakao',
               'lgchem': 'LGChemical',
               'lghnh': 'LGH&H',
               'bio': 'SamsungBiologics',
               'samsung1': 'SamsungElectronics',
               'samsung2': 'SamsungElectronics2',
               'sdi': 'SamsungSDI',
               'sk': 'SKhynix',
               'kospi': 'KOSPI'}


def symbol_to_path(symbol, base_dir="../data"):
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def merge_data(start_date, end_date, symbols):
    dates = pd.date_range(start_date, end_date)

    df = pd.DataFrame(index=dates)
    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col="Date", parse_dates=True,
                              usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Open': symbol + '_open', 'High': symbol + '_high', 'Low': symbol + '_low',
                                          'Close': symbol + '_close', 'Volume': symbol + '_volume'})
        if symbol == "KOSPI":
            df_temp[symbol + '_volume'] = df_temp[symbol + '_volume'].str.replace(pat='K', repl='')
            df_temp[symbol + '_volume'] = df_temp[symbol + '_volume'].str.replace(pat=',', repl='')
            df_temp[symbol + '_volume'] = df_temp[symbol + '_volume'].apply(pd.to_numeric)
        # volume 0일때 데이터 삭제  (네이버 액면분할, 삼바 등)
        df_temp[symbol + '_volume'].replace(0, np.nan, inplace=True)

        df = df.join(df_temp)

    # TODO: cleaning or filling missing value
    df.dropna(inplace=True)
    return df


def make_features(trade_company_list, start_date, end_date, is_training):
    # TODO: Choose symbols to make feature
    feature_company_list = trade_company_list
    symbol_list = [symbol_dict[c] for c in feature_company_list]
    table = merge_data(start_date, end_date, symbol_list)

    # DO NOT CHANGE
    test_days = 10
    open_prices = np.asarray(table[[symbol_dict[c] + '_open' for c in trade_company_list]])
    close_prices = np.asarray(table[[symbol_dict[c] + '_close' for c in trade_company_list]])

    ######################
    # Correlation matrix #
    ######################
    #     corr_matrix = pd.DataFrame()
    #     for c in feature_company_list:
    #         corr_matrix[c]  = table[symbol_dict[c] + '_close']
    #     sns.clustermap(corr_matrix.corr(method ='pearson'),
    #                 cmap = 'RdYlBu_r',
    #                 annot = True,   # 실제 값을 표시한다
    #                 linewidths=.5,  # 경계면 실선으로 구분하기
    #                 cbar_kws={"shrink": .5},# 컬러바 크기 절반으로 줄이기
    #                 vmin = -1,vmax = 1   # 컬러바 범위 -1 ~ 1
    #                )
    #     plt.savefig("corr_short.png")
    #     plt.show()

    data = dict()
    for c in feature_company_list:
        data[c, '_close_ch'] = table[symbol_dict[c] + '_close'].pct_change()
        data[c, '_open_ch'] = table[symbol_dict[c] + '_open'].diff() / table[symbol_dict[c] + '_close']
        data[c, '_high_ch'] = table[symbol_dict[c] + '_high'].diff() / table[symbol_dict[c] + '_close']
        data[c, '_low_ch'] = table[symbol_dict[c] + '_low'].diff() / table[symbol_dict[c] + '_close']
        data[c, '_volume_ch'] = table[symbol_dict[c] + '_volume'].pct_change()

    # Scaling
    # scaler 생성은 test days 전까지 한정, fit은 table 전체에 대해 진행

    # TODO: make features
    input_days = 3

    features = list()
    for a in range(data['hmotor', '_close_ch'].shape[0] - input_days):
        # close feature
        tmps = list()
        for c in trade_company_list:
            tmp = data[c, '_close_ch'][a:a + input_days]
            tmps.append(tmp)
        close_feature = np.concatenate(tmps, axis=0)
        # open feature
        tmps = list()
        for c in trade_company_list:
            tmp = data[c, '_open_ch'][a:a + input_days]
            tmps.append(tmp)
        open_feature = np.concatenate(tmps, axis=0)
        # high feature
        tmps = list()
        for c in trade_company_list:
            tmp = data[c, '_high_ch'][a:a + input_days]
            tmps.append(tmp)
        high_feature = np.concatenate(tmps, axis=0)
        # low feature
        tmps = list()
        for c in trade_company_list:
            tmp = data[c, '_low_ch'][a:a + input_days]
            tmps.append(tmp)
        low_feature = np.concatenate(tmps, axis=0)
        # volumn feature
        tmps = list()
        for c in trade_company_list:
            tmp = data[c, '_volume_ch'][a:a + input_days]
            tmps.append(tmp)
        volume_feature = np.concatenate(tmps, axis=0)

        features.append(np.concatenate([
            close_feature,
            open_feature,
            high_feature,
            low_feature,
            volume_feature
        ], axis=0))

    features = np.array(features)

    if not is_training:
        return open_prices[-test_days:], close_prices[-test_days:], features[-test_days:]

    return open_prices[input_days + 1:], close_prices[input_days + 1:], features[1:]


if __name__ == "__main__":
    trade_company_list = ['hmotor', 'naver', 'lgchem', 'kakao', 'lghnh', 'samsung2', 'sdi']
    # trade_company_list =  ['cell', 'hmotor', 'naver', 'lgchem','kakao', 'lghnh', 'samsung1', 'samsung2', 'sdi', 'sk']
    open, close, feature = make_features(trade_company_list, '2018-01-01', '2020-05-19', True)
    #     print(open.shape)
    #     print(open,'\n')
    #     print(close,'\n')
    #     print(feature[0])