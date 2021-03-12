import DataGenerator
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.tsa.api as smt
from statsmodels.tsa.api import VAR

start_date = '2010-01-01'
end_date = '2020-04-06'

df = DataGenerator.merge_data(start_date, end_date, symbols=['AUD', 'CNY', 'EUR','GBP', 'HKD', 'JPY', 'BrentOil', 'Copper', 'CrudeOil', 'Gasoline', 'Gold', 'NaturalGas', 'Platinum', 'Silver'])

dataset = pd.concat([df.USD_Price, df.AUD_Price, df.CNY_Price, df.EUR_Price, df.GBP_Price, df.HKD_Price, df.JPY_Price, df.BrentOil_Price, df.Copper_Price, df.CrudeOil_Price, df.Gasoline_Price, df.Gold_Price, df.NaturalGas_Price, df.Platinum_Price, df.Silver_Price], axis=1)
dataset = dataset.fillna(method = 'ffill')
dataset = dataset.fillna(method = 'bfill')

nobs = 10
X_train, X_test = dataset[0:-nobs], dataset[-nobs:]

transform_data = X_train.diff().dropna()


def adfuller_test(series, signif=0.05, name='', verbose=False):
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic': round(r[0], 4), 'pvalue': round(r[1], 4), 'n_lags': round(r[2], 4), 'n_obs': r[3]}
    p_value = output['pvalue']

    def adjust(val, length=6):
        return str(val).ljust(length)

    print(f' Augmented Dickey-Fuller Test on "{name}"', "\n", '-' * 47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Siginificance Level   = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key, val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")


# ADF Test on each column
for name, column in transform_data.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')

from statsmodels.tsa.stattools import grangercausalitytests

maxlag = 12


def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    X_train = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in X_train.columns:
        for r in X_train.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)]
            if verbose: print(f'Y= {r}, X = {c}, P-Values = {p_values}')
            min_p_value = np.min(p_values)
            X_train.loc[r, c] = min_p_value

    X_train.columns = [var + '_x' for var in variables]
    X_train.index = [var + '_y' for var in variables]
    return X_train


def cointegration_test(transform_data, alpha=0.05):
    out = coint_johansen(transform_data, -1, 5)
    d = {'0.90': 0, '0.95': 1, '0.99': 2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1 - alpha)]]

    def adjust(val, length=6): return str(val).ljust(length)

    print('Name :: Test Stat > C(95%)   =>   Signif \n', '--' * 20)
    for col, trace, cvt in zip(transform_data.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace, 2), 9), ">", adjust(cvt, 8), ' =>   ', trace > cvt)


mod = smt.VAR(X_train)
res = mod.fit(maxlags=maxlag, ic='aic')
print(res.summary())


