#new rnn using three LSTM layers
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM
from keras.layers import Conv2D, MaxPooling2D, Reshape
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import regularizers
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
def train_and_evaluate(filename_list, model_name,train_ratio, timesteps = 20, feature_num = 1, epoches = 4, batch_size = 240):
    #first build the model
    model = Sequential()
   
    model.add(LSTM(16,input_shape = (timesteps, feature_num), dropout = 0.25, return_sequences = True))
    #model.add(LSTM(32, recurrent_dropout = 0.5,activation = 'relu',return_sequences = True))
    model.add(LSTM(16, dropout = 0.5, activation = 'relu', kernel_regularizer=regularizers.l2(0.001), return_sequences = True))
    model.add(LSTM(8, dropout = 0.5, activation = 'relu'))
    #model.add(LSTM(4,dropout = 0.25, activation = 'relu'))
    model.add(Dense(10, activation='softmax',kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(2, activation='softmax'))
    
    sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    #a batch for a file
    #use the last one as test set
    for i in range(epoches):
    	print 'epoches',i
	for filename in filename_list[:-1]:
	    x_train,y_train, x_test, y_test = load_data('../data/svm/'+filename, 1)
	    x_train = np.reshape(x_train, (len(x_train), timesteps, feature_num))
            y_train = np_utils.to_categorical(y_train,2)
	         
	    print filename
	            
	    total = len(y_train)
	    steps = total/batch_size
	    print len(y_train), 'samples'
	    print batch_size,'batch size'
            print steps, 'steps'
	            
	    for i in range(steps-1):
	        loss = model.train_on_batch(x_train[i*batch_size : (i+1)*batch_size], y_train[i*batch_size:(i+1)*batch_size])
                print loss
	    model.train_on_batch(x_train[steps*batch_size:], y_train[steps*batch_size:])
            print loss
    x_train, y_train, x_test, y_test = load_data('../data/svm/'+filename_list[-1], 0)
    print len(y_train)
    print len(y_test)
    x_test = np.reshape(x_test, (len(x_test), timesteps, feature_num))
    y_test = np_utils.to_categorical(y_test,2)
    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    print score
    model.save(model_name)
    classes = model.predict(x_test, batch_size = batch_size)
    print classes[:150]
def main():
    filename_list = get_file_list(direction = '../data/svm')
    train_and_evaluate(filename_list, model_name = 'rnn_new_7.h5',train_ratio = 0.7, batch_size = 4000, epoches = 3)

main()
# Generate dummy data
