from pandas.plotting import register_matplotlib_converters
import numpy as np
import pandas as pd
import DataGenerator as DataGenerator
from matplotlib import pyplot as plt
import random
register_matplotlib_converters()

PRINT_EPOCH = 1000
VALID_EPOCH = 5

class ReplayBuffer(object):
    def __init__(self):
        self.storage = list()

    def add(self, state, action, reward, new_state):
        self.storage.append((state, action, reward, new_state))

    def shuffle(self):
        random.shuffle(self.storage)

    # 전날 샀던 주식을 당일 open에 모두 팔고, 새로운 주식을 산다.


# 분산 투자 기법을 이용 (타겟 주식, 상관도가 낮은 주식들에 투자)
def do_action(action_list, action, budget, num_stocks_list, stock_price_list):
    num_stocks = len(num_stocks_list)
    if action == 'not_buying':
        budget += sum(stock_price_list * num_stocks_list)
        num_stocks_list = [0] * (len(action_list) - 1)
        for idx in range(num_stocks):
            n_buy_rest = min(np.ceil((10. ** 7) / num_stocks / stock_price_list[idx]),
                             (budget / num_stocks) // stock_price_list[idx])
            num_stocks_list[idx] += n_buy_rest
            budget -= stock_price_list[idx] * n_buy_rest
    else:
        for i, a in enumerate(action_list[:-1]):
            if a == action:
                budget += sum(stock_price_list * num_stocks_list)
                num_stocks_list = [0] * (len(action_list) - 1)
                # 자금의 반은 target을 매입
                # n_buy_target = min(np.ceil(50* (10.** 6) / stock_price_list[i]), budget // stock_price_list[i])
                n_buy_target = 0.5 * budget // stock_price_list[i]
                num_stocks_list[i] += n_buy_target
                budget -= stock_price_list[i] * n_buy_target
                temp_budget = budget
                # 나머지 자금은 나머지 종목을 매입
                for idx in range(num_stocks):
                    if not idx == i:
                        # n_buy_rest = min(np.ceil((5* 10. ** 7)/num_stocks / stock_price_list[idx]), (budget/num_stocks) // stock_price_list[idx])
                        n_buy_rest = (temp_budget / num_stocks) // stock_price_list[idx]
                        num_stocks_list[idx] += n_buy_rest
                        budget -= stock_price_list[idx] * n_buy_rest

    return budget, num_stocks_list


def run_simulation(policy, initial_budget, initial_num_stocks, open_prices, close_prices, features):
    action_count = [0] * len(policy.actions)  # policy는 model이라고 생각하면 됨, actions는 stock list
    action_seq = list()

    budget = initial_budget
    num_stocks_list = initial_num_stocks
    replay_buffer = ReplayBuffer()

    for t in range(len(open_prices) - 2):
        ##### TODO: define current state
        current_state = np.asmatrix(features[t])  # np.hstack((features[t], num_stocks_list)))

        # calculate current portfolio value
        current_portfolio = budget + sum(num_stocks_list * open_prices[t])

        ##### select action & update portfolio values
        action = policy.select_action(current_state, True)
        action_seq.append(action)
        action_count[policy.actions.index(action)] += 1

        budget, num_stocks_list = do_action(policy.actions, action, budget, num_stocks_list, open_prices[t])

        ##### TODO: define reward
        # calculate new portofolio after taking action
        new_portfolio = budget + sum(num_stocks_list * open_prices[t + 1])
        # calculate reward from taking an action at a state
        reward = new_portfolio - current_portfolio  # reward를 하루간 portpolio 변화량으로 정의 (open)

        ##### TODO: define next state
        next_state = np.asmatrix(features[t + 1])  # np.asmatrix(np.hstack((features[t+1], num_stocks_list)))
        # Replay buffer
        replay_buffer.add(current_state, action, reward, next_state)
    replay_buffer.shuffle()
    ##### update the policy after experiencing a new action
    for current_state, action, reward, next_state in replay_buffer.storage:
        policy.update_q(current_state, action, reward, next_state)

    # compute final portfolio worth
    portfolio = budget + sum(num_stocks_list * close_prices[-1])
    print('budget: {}, shares: {}, stock price: {} =>  portfolio: {}'.format(budget, num_stocks_list, close_prices[-1],
                                                                             portfolio))

    return portfolio, action_count, np.asarray(action_seq)


## For validation
def run_valid(policy, initial_budget, initial_num_stocks, open_prices, close_prices, features):
    action_count = [0] * len(policy.actions)
    action_seq = list()

    budget = initial_budget
    num_stocks_list = initial_num_stocks
    for t in range(len(open_prices) - 2):
        ##### TODO: define current state
        current_state = np.asmatrix(features[t])  # np.asmatrix(np.hstack((features[t], num_stocks_list)))

        # calculate current portfolio value
        current_portfolio = budget + sum(num_stocks_list * open_prices[t])

        ##### select action & update portfolio values
        action = policy.select_action(current_state, False)
        action_seq.append(action)
        action_count[policy.actions.index(action)] += 1

        budget, num_stocks_list = do_action(policy.actions, action, budget, num_stocks_list, open_prices[t])

        # calculate new portofolio after taking action
        new_portfolio = budget + sum(num_stocks_list * open_prices[t + 1])
        ##### TODO: define next state
        next_state = np.asmatrix(features[t + 1])  # np.asmatrix(np.hstack((features[t+1], num_stocks_list)))
    # compute final portfolio worth
    portfolio = budget + sum(num_stocks_list * close_prices[-1])

    return portfolio


## For training
def run_simulations(company_list, policy, budget, num_stocks, open_prices, close_prices, features, num_epoch,
                    start_valid, end_valid, scaler):
    # Load validation dataset
    open_prices_valid, close_prices_valid, features_valid = DataGenerator.make_features(company_list, start_valid, end_valid,
                                                                                        is_training=True)
    features_valid = scaler.transform(features_valid)
    open_prices_test, close_prices_test, features_test = DataGenerator.make_features(company_list, start_valid, end_valid,
                                                                                    is_training=False)
    features_test = scaler.transform(features_test)
    # Training
    best_portfolio_valid, best_portfolio_test = 0, 0
    final_portfolios = list()
    for epoch in range(num_epoch):
        print("-------- simulation {} --------".format(epoch + 1))
        final_portfolio, action_count, action_seq = \
            run_simulation(policy, budget, num_stocks, open_prices, close_prices, features)
        final_portfolios.append(final_portfolio)

        print('actions : ', *zip(policy.actions, action_count), )

        if (epoch + 1) % PRINT_EPOCH == 0:
            action_seq2 = np.concatenate([['.'], action_seq[:-1]])  # 명일 매도
            for i, a in enumerate(policy.actions[:-1]):
                plt.figure(figsize=(40, 20))
                plt.title('Company {} / Epoch {}'.format(a, epoch + 1))
                plt.plot(open_prices[0: len(action_seq), i], 'grey')
                plt.plot(pd.DataFrame(open_prices[: len(action_seq), i])[action_seq2 == a], 'ro', markersize=1)  # sell
                plt.plot(pd.DataFrame(open_prices[: len(action_seq), i])[action_seq == a], 'bo', markersize=1)  # buy
                plt.show()

        if (epoch + 1) % VALID_EPOCH == 0:
            policy.save_model("LFD_project4_test_{}".format(epoch))
            test_portfolio = run(policy, budget, num_stocks, open_prices_test, close_prices_test, features_test)
            print("================Test portfolio: %.2f won============================" % test_portfolio)
            policy.save_model("LFD_project4_valid_{}".format(epoch))
            valid_portfolio = run_valid(policy, budget, num_stocks, open_prices_valid, close_prices_valid,
                                        features_valid)
            print("================Valid portfolio: %.2f won============================" % valid_portfolio)
            #             ##### save if best portfolio value is updated
            #             if best_portfolio_valid < valid_portfolio:
            #                 best_portfolio_valid = valid_portfolio
            #                 policy.save_model("LFD_project4-{}".format(epoch))
            #             test_portfolio = run(policy, budget, num_stocks, open_prices_test, close_prices_test, features_test)
            #             print("================Test portfolio: %.2f won============================" % test_portfolio)
            #             if best_portfolio_test < test_portfolio:
            #                 best_portfolio_test = test_portfolio
            #                 policy.save_model("LFD_project4-{}_test".format(epoch))
    print(final_portfolios[-1])


## This is for validation and test
def run(policy, initial_budget, initial_num_stocks, open_prices, close_prices, features):
    budget = initial_budget
    num_stocks_list = initial_num_stocks

    for i in range(len(open_prices)):
        current_state = np.asmatrix(features[i])  # np.asmatrix(np.hstack((features[i], num_stocks_list)))
        action = policy.select_action(current_state, is_training=False)
        budget, num_stocks_list = do_action(policy.actions, action, budget, num_stocks_list, open_prices[i])
        print('Day {}'.format(i + 1))
        print('action {} / budget {} / shares {}'.format(action, budget, num_stocks_list))
        print('portfolio with  open price : {}'.format(budget + sum(num_stocks_list * open_prices[i])))
        print('portfolio with close price : {}\n'.format(budget + sum(num_stocks_list * close_prices[i])))

    portfolio = budget + sum(num_stocks_list * close_prices[-1])

    print('Finally, you have')
    print('budget: %.2f won' % budget)
    print('Share : {}'.format(num_stocks_list))
    print('Share value : {} won'.format(close_prices[-1]))
    print()
    return portfolio