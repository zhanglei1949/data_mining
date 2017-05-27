import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras import regularizers
from keras.utils import np_utils
import csv
import h5py


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

    return x_train, y_train, x_test, y_test
def train_and_evaluate(filename, model_name,train_ratio = 0.7, batch_size = 320,epochs = 6):
    model = Sequential()

    model.add(Dense(120, activation='relu', input_shape=(80, )))
    model.add(Dropout(0.25))
    model.add(Dense(640,kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.25))
    #model.add(Dense(100,activation = 'relu', kernel_regularizer=regularizers.l2(0.01)))
    #model.add(Dense(48, activation = 'relu', kernel_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l1(0.001)))
    #model.add(Dropout(0.25))
    model.add(Dense(480, activation = 'relu',kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.25))
    model.add(Dense(240,activation = 'relu',kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.25))

    model.add(Dense(60,activation = 'relu',kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.25))

    model.add(Dense(2, activation='softmax'))

    sgd = SGD(lr=0.025, decay=1e-8, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    #a batch for a file
    #use the last one as test set
    x_train, y_train, x_test, y_test = load_data(filename, train_ratio)
    x_train = np.reshape(x_train, (len(x_train), 80))
    x_test = np.reshape(x_test, (len(x_test), 80))
    y_train = np_utils.to_categorical(y_train,2)
    y_test = np_utils.to_categorical(y_test, 2)
    model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, shuffle = True, validation_split = 0.2)
    loss = model.evaluate(x_test, y_test, batch_size = 64)
    print loss
    model.save(model_name)

def main():
    filename = '../data/generated_10_tick_1_tick/5-27.csv'
    train_and_evaluate(filename, model_name = 'dense_5_27_0.h5',train_ratio = 0.7, epochs = 14,batch_size = 1320)

main()
# Generate dummy data
