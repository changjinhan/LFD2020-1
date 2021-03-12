from sklearn import neural_network as NN
import pickle
import DataGenerator
import numpy as np
import warnings


def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)

    start_date = '2012-01-02'
    end_date = '2020-04-06'
    random_state = 0

    training_x, training_y = DataGenerator.make_features(start_date, end_date, is_training=True)

    # TODO: set model parameters
    model = NN.MLPRegressor(random_state=random_state, shuffle=False, hidden_layer_sizes= (64,32),
                        max_iter=2000, early_stopping=True, activation='tanh', solver='sgd', learning_rate='adaptive')
    model.fit(training_x, training_y)

    # TODO: fix pickle file name
    filename = 'team11_model.pkl'
    pickle.dump(model, open(filename, 'wb'))
    print('saved {}'.format(filename))


if __name__ == "__main__":
    main()





