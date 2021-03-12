from hmmlearn.hmm import GaussianHMM
import pickle
import DataGenerator
import numpy as np

def main():

    start_date = '2010-01-01'
    end_date = '2020-04-18'

    training_x = DataGenerator.make_features(start_date, end_date, is_training=True)


    # TODO: set model parameters
    n_components = 11
    model = GaussianHMM(n_components=n_components, covariance_type='tied', n_iter=100, random_state=0)
    model.fit(training_x)

    # TODO: fix pickle file name
    filename = 'team10_model.pkl'
    pickle.dump(model, open(filename, 'wb'))
    print('saved {}'.format(filename))


if __name__ == "__main__":
    main()



