from sklearn import svm
import pickle
import DataGenerator

def main():
    test_day = ['2020-01-19', '2020-02-01', '2020-02-02', '2020-02-08']
    # test_day = ['2020-02-09', '2020-02-15', '2020-02-16', '2020-02-22']
    training_x, training_y = DataGenerator.get_data(test_day, is_training=True)

    # ================================ train SVM model=========================================
    # TODO: set parameters
    print('start training model')
    model = svm.SVC(random_state=0, C=10, kernel='rbf', gamma=0.1,
                    coef0=0.0, shrinking=True, probability=False, cache_size=600, tol=0.001, class_weight='balanced',
                    verbose=False, max_iter=-1, decision_function_shape='ovr')


    model.fit(training_x, training_y)
    print('completed training model')

    # TODO: fix pickle file name
    filename = 'team10_model.pkl'
    pickle.dump(model, open(filename, 'wb'))
    print('save complete')

if __name__ == '__main__':
    main()

