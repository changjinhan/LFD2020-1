import DataGenerator as DataGenerator
from decision_ql import QLearningDecisionPolicy
import simulation as simulation
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
tf.compat.v1.reset_default_graph()

if __name__ == '__main__':
    start, end = '2018-01-01', '2020-05-19'
    company_list = ['hmotor', 'naver', 'lgchem', 'kakao', 'lghnh', 'samsung2', 'sdi']
    actions = company_list + ['not_buying']
    #########################################
    open_prices, close_prices, features = DataGenerator.make_features(company_list, '2018-01-01', '2020-05-19', is_training=True)
    scaler = StandardScaler()
    scaler.fit(features)
    ###########################################
    open_prices, close_prices, features = DataGenerator.make_features(company_list, start, end, is_training=False)
    features = scaler.transform(features)

    budget = 10. ** 8
    num_stocks = [0] * len(company_list)
    input_dim = len(features[0])

    # TODO: fix checkpoint directory name
    policy = QLearningDecisionPolicy(epsilon=0, gamma=0, decay=0, lr=0, actions=actions, input_dim=input_dim,
                                     model_dir="LFD_project4_team09")
    final_portfolio = simulation.run(policy, budget, num_stocks, open_prices, close_prices, features)

    print("Final portfolio: %.2f won" % final_portfolio)

