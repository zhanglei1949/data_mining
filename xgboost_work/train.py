#!/usr/bin/python

from __future__ import division

import numpy as np
import xgboost as xgb
import csv
# label need to be 0 to num_class -1
#data = np.loadtxt('./dermatology.data', delimiter=',',
#        converters={33: lambda x:int(x == '?'), 34: lambda x:int(x)-1})

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

def main():
	filename = '../data/generated_10_tick_1_tick/5-27.csv'
	train_ratio = 0.7
	vali_split = 0.8
	x_train, y_train, x_test, y_test = load_data(filename,train_ratio)
	x_train = np.reshape(x_train, (len(x_train), 80,))
	x_test = np.reshape(x_test, (len(x_test), 80,))
	y_train = np.reshape(y_train, (len(y_train),))
	y_test = np.reshape(y_test, (len(y_test),))

	x_train_tr = x_train[:int(vali_split*len(x_train))]
	x_vali = x_train[int(vali_split*len(x_train)):]
	y_train_tr = y_train[:int(vali_split*len(y_train))]
	y_vali = y_train[int(vali_split*len(y_train)):]
	print x_train_tr.shape
	print y_train_tr.shape
	print x_vali.shape
	print y_vali.shape
	xg_train = xgb.DMatrix(x_train_tr, label = y_train_tr)
        xg_vali = xgb.DMatrix(x_vali, label = y_vali)
	xg_test = xgb.DMatrix(x_test, label = y_test)


	# setup parameters for xgboost
	param = {}
	# use softmax multi-class classification
	param['objective'] = 'binary:logitraw'
	# scale weight of positive examples
	param['eta'] = 0.05
        #0.1
	param['max_depth'] = 5
        param['subsample'] = 0.8
	param['silent'] = 1
        param['eval_metric'] = 'auc'
        param['alpha'] = 1
	#param['nthread'] = 4
	#param['num_class'] = 2

	watchlist = [(xg_train, 'train'), (xg_vali, 'val')]
	num_round = 45
	bst = xgb.train(param, xg_train, num_round, watchlist)
	# get prediction
	pred = bst.predict(xg_test)
	#error_rate = np.sum(pred != y_test) / y_test.shape[0]
	#print('Test error using softmax = {}'.format(error_rate))
	cnt = 0
        for i in range(len(pred)):
    	    if (pred[i]>0 and y_test[i] == 1):
    	        cnt+=1
    	    elif (pred[i] <0 and y_test[i]==0):
    		cnt+=1
        print float(cnt)/len(pred) 
	# do the same thing again, but output probabilities
	param['objective'] = 'binary:logistic'
	bst = xgb.train(param, xg_train, num_round, watchlist)
	pred_prob = bst.predict(xg_test)
        print pred_prob[:20]
	cnt = 0
        for i in range(len(pred_prob)):
            if (pred[i] > 0.5 and y_test[i]==1):
                cnt+=1
            elif (pred[i] <0.5 and y_test[i]==0):
                cnt+=1
        print float(cnt)/len(pred_prob)
main()