#!/usr/bin/python

from __future__ import division

import numpy as np
import xgboost as xgb
import csv
# label need to be 0 to num_class -1
#data = np.loadtxt('./dermatology.data', delimiter=',',
#        converters={33: lambda x:int(x == '?'), 34: lambda x:int(x)-1})

def load_data(fileList,train_ratio):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    dataList_x = []
    dataList_y = []
    for filename in fileList:
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
        x_train.append(dataList_x[:train_part])
        x_test.append(dataList_x[train_part:])
        y_train.append(dataList_y[:train_part])
        y_test.append(dataList_y[train_part:])
        print 'reading', filename
        print 'total samples', raw_len
    print 'train', len(y_train)
    print 'test', len(y_test)
    return x_train, y_train, x_test, y_test

def main():

    bst = xgb.Booster()
    bst.load_model('6_9_649.model')
    
    xg_train = xgb.DMatrix('train_xgb')
    print xg_train.num_row()
    xg_vali = xgb.DMatrix('vali_xgb')
    print xg_vali.num_row()
    xg_test = xgb.DMatrix('test_xgb')
    print xg_test.num_row()

    loss_acc = bst.eval(xg_test,name='test')
    print loss_acc
    loss_acc = bst.eval(xg_train,name='test')
    print loss_acc
    loss_acc = bst.eval(xg_vali,name='test')
    print loss_acc


main()