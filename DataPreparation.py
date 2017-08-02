import pandas as pd
import numpy as np
import pickle


def prep_train_test(dataset, percent_test =0.15):
    conv_np = np.array(dataset)
    np.random.shuffle(conv_np)
    test_size = int(percent_test * (len(dataset)))
    train_x = conv_np[:,:-1][:-test_size]
    train_y = conv_np[:,-1][:-test_size]
    test_x = conv_np[:,:-1][-test_size:]
    test_y = conv_np[:,-1][-test_size:]
    return train_x, train_y, test_x, test_y

def y_conversion(tr_y, te_y):
    y = []
    for item in [tr_y, te_y]:
        each_item =[]
        for val in item:
            # if target is 0.0
            if val == 0:
                each_item.append([0,1])
            # if target in 1.0
            else:
                each_item.append([1,0])
        y.append(each_item)
    return y

if __name__ == '__main__':
    dataset = pd.read_csv('train.csv')
    dataset.drop('Time', 1, inplace=True)
    print(dataset.head(2))
    print(dataset.shape)
    train_x, train_y, test_x, test_y = prep_train_test(dataset)
    y = y_conversion(train_y, test_y)
    train_y = y[0]
    test_y = y[1]
    print(len(train_x), len(test_x))
    with open('credit_card.pickle', 'wb') as s:
        pickle.dump([train_x, train_y, test_x, test_y], s)






















