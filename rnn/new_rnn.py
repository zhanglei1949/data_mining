#new rnn using three LSTM layers
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM
from keras.layers import Conv2D, MaxPooling2D, Reshape
from keras.optimizers import SGD
from keras.utils import np_utils
import csv
import h5py

import os

#from data_preprocess5.py import get_file_list
def get_file_list(direction):
    if (os.path.isfile(direction)):
        return []
    else:
        res = []
        for s in os.listdir(direction):
            res.append(s)
    return res
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
    print 'reading', filename
    print 'total samples', raw_len
    print 'train', len(y_train)
    print 'test', len(y_test)
    return x_train, y_train, x_test, y_test
def train_and_evaluate(filename_list, model_name,train_ratio, timesteps = 20, feature_num = 8, epoches = 3):
    #first build the model
    model = Sequential()
   
    model.add(LSTM(56, input_shape = (timesteps, feature_num), dropout = 0.25, return_sequences = True))
    model.add(LSTM(28, dropout = 0.25, return_sequences = True))
    model.add(LSTM(7, dropout = 0.25))
    model.add(Dense(2, activation='softmax'))

    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #a batch for a file
    #use the last one as test set
    for filename in filename_list[:-1]:
        x_train,y_train, x_test, y_test = load_data('../data/new/'+filename, 1)
        x_train = np.reshape(x_train, (len(x_train), timesteps, feature_num))
        #x_test = np.reshape(x_test, (len(x_test), dim_x, dim_y, 1))
        y_train = np_utils.to_categorical(y_train,2)
        #y_test = np_utils.to_categorical(y_test,2)
        for i in range(epoches): 
            print filename
            
            print len(y_train)
            model.train_on_batch(x_train, y_train)
            #model.fit(x_train, y_train, batch_size=128, epochs=6)
    x_train, y_train, x_test, y_test = load_data('../data/new/'+filename_list[-1], 0)
    print len(y_train)
    print len(y_test)
    x_test = np.reshape(x_test, (len(x_test), timesteps, feature_num))
    y_test = np_utils.to_categorical(y_test,2)
    score = model.evaluate(x_test, y_test, batch_size=32)
    print score
    model.save(model_name)
def main():
    filename_list = get_file_list(direction = '../data/new')
    train_and_evaluate(filename_list, model_name = 'rnn_new.h5',train_ratio = 0.7)

main()
# Generate dummy data
