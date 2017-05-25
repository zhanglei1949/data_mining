import numpy as np
import csv
import h5py

from keras.models import Sequential,load_model
from keras.layers import Dense, LSTM
from keras.utils import np_utils

def load_data(filename,train_ratio):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    dataList_x = []
    dataList_y = []
    raw_data = csv.reader(file(filename))
    raw_len = 0
    for row in raw_data:
        cur = []
        for i in range(len(row)-1):
            cur.append(float(row[i]))
        dataList_x.append(cur)
        dataList_y.append(float(row[len(row)-1]))
    raw_len = len(dataList_y)
    train_part = int(train_ratio*raw_len)
    x_train = dataList_x[:train_part]
    x_test = dataList_x[train_part:]
    y_train = dataList_y[:train_part]
    y_test = dataList_y[train_part:]

    print 'total samples', raw_len
    print 'train', len(y_train)
    print 'test', len(y_test)
    return x_train, y_train, x_test, y_test

def main():
    #filename = 'valid.csv'
    filename = '../data/new/TK_rb0000_1733_29_valid.csv'
    timesteps = 20
    feature_num = 8
    x_train, y_train, x_test, y_test = load_data(filename,0)
    #x_test = np.reshape(x_test, (len(x_test), timesteps, feature_num, 1))
    x_test = np.reshape(x_test, (len(x_test), 10, 16, 1))
    y_test = np_utils.to_categorical(y_test,2)
    model = load_model('2017-5-22-2.h5')
    classes = model.predict(x_test)
    up = 0
    down = 0
    #flat = 0
    print classes[:100]
    for i in range(len(classes)):
        # incorrect way
        if (classes[i][0] > classes[i][1]):
            down+=1
        else:
            up+=1
    print up,down
main()
