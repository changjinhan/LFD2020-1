from decision_ql import QLearningDecisionPolicy
import DataGenerator as DataGenerator
import simulation as simulation
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
tf.compat.v1.reset_default_graph()

if __name__ == '__main__':
    start_train, end_train = '2018-01-01', '2020-04-01'
    start_valid, end_valid = '2020-04-01', '2020-05-19'
    # TODO: Choose companies for trading
    # company_list = ['cell', 'hmotor', 'naver', 'lgchem','kakao', 'lghnh', 'samsung1', 'samsung2', 'sdi', 'sk']
    company_list = ['hmotor', 'naver', 'lgchem', 'kakao', 'lghnh', 'samsung2',
                    'sdi']

    # TODO: define action
    actions = company_list + ['not_buying']

    # TODO: tuning model hyperparameters
    epsilon_init = 0.8
    decay = 1 - 5 * 10. ** -3  # epslion 은 epoch을 거듭할수록 값이 작아지도록 설정
    gamma = 0  # 어떤 action을 하든 next state는 변함이 없다. transaction cost도 없으므로
    lr = 0.005
    num_epoch = 200
    #########################################
    open_prices, close_prices, features = DataGenerator.make_features(company_list, start_train, end_train, is_training=True)
    scaler = StandardScaler()
    scaler.fit(features)
    features = scaler.transform(features)

    budget = 10. ** 8
    num_stocks = [0] * len(company_list)  # stock 개수 만큼의 0 list
    input_dim = len(features[0])  # + len(company_list), budget 뺏음

    # Define model
    policy = QLearningDecisionPolicy(epsilon=epsilon_init, gamma=gamma, decay=decay, lr=lr, actions=actions,
                                     input_dim=input_dim, model_dir="model")
    # training
    simulation.run_simulations(company_list=company_list, policy=policy, budget=budget, num_stocks=num_stocks,
                                open_prices=open_prices, close_prices=close_prices, features=features,
                                num_epoch=num_epoch, start_valid=start_valid, end_valid=end_valid, scaler=scaler)


    # TODO: fix checkpoint directory name
    # policy.save_model("LFD_project4_team00-e{}".format(num_epoch))
    policy.save_model("LFD_project4_team09")
