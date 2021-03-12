import pickle
import DataGenerator
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score

def inrace_pred(pred_y, test_date, test_rc):
    win = list()
    test_date = np.array(test_date)
    test_rc = np.array(test_rc)
    # initialize
    temp_lst = [pred_y[0]]
    ref_date = test_date[0]
    ref_rc = test_rc[0]
    # loop
    for i in range(1, len(pred_y)):
        if ref_date == test_date[i] and ref_rc == test_rc[i]:
            temp_lst.append(pred_y[i])
        else:
            win.extend(discrete(temp_lst))  # 매 경기마다 상대평가 진행
            temp_lst = [pred_y[i]]
            ref_date = test_date[i]
            ref_rc = test_rc[i]
        if i == len(pred_y) - 1:  # 마지막 date, race의 결과 list에 추가
            win.extend(discrete(temp_lst))
    return win

def discrete(lst):  # 4등까지 1, 이외 0
    leng = len(lst)
    temp = sorted(range(len(lst)), key=lambda k: lst[k])
    for i in range(leng):
        if i < leng - 4:
            lst[temp[i]] = 0
        else:
            lst[temp[i]] = 1
    return lst

def main():

    test_day = ['2020-01-19', '2020-02-01', '2020-02-02', '2020-02-08']
    # test_day = ['2020-02-09', '2020-02-15', '2020-02-16', '2020-02-22']
    test_x, test_y, test_date, test_rc  = DataGenerator.get_data(test_day, is_training=False)
    filename = 'team10_model.pkl'
    model = pickle.load(open(filename, 'rb'))
    print('load complete')
    print(model.get_params())
    
    # model inrace test
    pred_prob = model.decision_function(test_x)  
    pred_y = inrace_pred(pred_prob,test_date, test_rc) 
    print('accuracy: {}'.format(accuracy_score(test_y, pred_y)))
    print('recall: {}'.format(recall_score(test_y, pred_y)))
    print('f1-score: {}'.format(f1_score(test_y, pred_y)))    
    


if __name__ == '__main__':
    main()