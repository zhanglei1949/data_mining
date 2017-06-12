#new rnn using three LSTM layers
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM
from keras.layers import Conv2D, MaxPooling2D, Reshape
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
from keras import regularizers
import csv
import h5py
import tensorflow as tf
import os
from keras.backend.tensorflow_backend import set_session
import keras.backend.tensorflow_backend as KTF

#from data_preprocess5.py import get_file_list

def get_session(gpu_fraction=0.7):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = 0
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def generator(fileList, timesteps, feature_num, batch_size):
    while 1:
    	for filename in fileList:
    	    raw_data = csv.reader(open(filename))
            print filename
	    x = []
    	    y = []
            cnt = 0
            for row in raw_data:
                cur = []
            	for i in range(len(row)-1):
                    cur.append(float(row[i]))
            	tmp_y = float(row[len(row)-1])
            	x.append(cur)
            	y.append(tmp_y)
            	cnt+=1
            	if (cnt==batch_size):
            	    cnt = 0
	            x = np.reshape(x, (batch_size,timesteps, feature_num))
    	            y = np_utils.to_categorical(y, 2)
	            yield(x,y)
	            x = []
	            y = []
            #x = np.reshape(x, (batch_size,timesteps, feature_num))
    	    #y = np_utils.to_categorical(y, 2)
            #yield(x,y)
def train_and_evaluate(trainFileList, testFileList, model_name, timesteps = 10, feature_num = 8, epochs = 4, steps_per_epoch = 300,batch_size = 10000):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    KTF.set_session(tf.Session(config=config))
    model = Sequential()
    model.add(LSTM(248, input_shape = (timesteps, feature_num), activation = 'sigmoid'))
    #model.add(LSTM(240, input_shape = (timesteps, feature_num),activation = 'relu', return_sequences = True,kernel_regularizer = regularizers.l1(0.001)))
    #model.add(Dropout(0.2))
    #model.add(LSTM(960, activation = 'relu', return_sequences = True, kernel_regularizer = regularizers.l1(0.001), dropout = 0.2))
    #model.add(Dropout(0.2))
    #model.add(LSTM(100,activation = 'relu',kernel_regularizer = regularizers.l1(0.001), dropout = 0.2))
    #model.add(Dropout(0.2))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation = 'sigmoid'))
    sgd = SGD(lr=0.002, decay=1e-7, momentum=0.9, nesterov=True)
    rmsprop = RMSprop(lr = 0.00002, rho=0.9, epsilon=1e-08, decay=1e-9)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #with tf.device('/gpu:3'):
    
    model.fit_generator(generator(trainFileList, timesteps, feature_num, batch_size = batch_size),epochs = epochs, steps_per_epoch = steps_per_epoch)
    model.save(model_name)
    loss = model.evaluate_generator(generator(testFileList, timesteps, feature_num, batch_size = batch_size),steps = steps_per_epoch)
    print loss
    #classes = model.predict(x_test,batch_size =32)
    
    #with tf.device('/gpu:3'):
    classes = model.predict_generator(generator(testFileList,timesteps, feature_num,batch_size),steps = steps_per_epoch)
    res = [0,0]
    print len(classes), 'samples'
    print classes[:50]
    for i in range(len(classes)):
        if(classes[i][0] < classes[i][1]):
            res[0]+=1
        else:
            res[1]+=1
    print res
    

    #print classes[:150]
def get_file_list(dir):
    if (os.path.isfile(dir)):
        return []
    else:
        res = []
        for s in os.listdir(dir):
            res.append(dir+s)
    return res
def main():
    fileList = get_file_list('../../data/generated/m/')
    trainFileList = fileList[:-1]
    testFileList = [fileList[-1]]
    print 'train',trainFileList
    print 'test',testFileList
    train_and_evaluate(trainFileList, testFileList, model_name = 'rnn_train_on_batch.h5',steps_per_epoch = 300, epochs = 30,batch_size = 4000)
    #epochs = 8
main()