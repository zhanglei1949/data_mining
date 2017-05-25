# svm model for machine learning project
import numpy as np
import scipy.io as scio
import csv
from sklearn import svm
from sklearn.externals import joblib
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
    dataList_x = dataList_x[:20000]
    dataList_y = dataList_y[:20000]
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
    filename = '../../data/new/TK_rb0000_1722_0_valid.csv'
    x_train, y_train, x_test, y_test = load_data(filename, 0.7)
    clf = svm.NuSVC(cache_size = 2000)
    clf.fit(x_train, y_train)
    #
    joblib.dump(clf,'cvm1.pkl')
    cnt = 0
    predicted = clf.predict(x_test)
    for i in range(len(y_test)):
        if (y_test[i]==predicted[i]):
            cnt+=1
    score = float(cnt)/len(y_test)
    print score
    print clf.score(x_test, y_test)
main()
