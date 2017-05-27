# for batch data reading and saving
import csv
import os
from sklearn import preprocessing
#from data_preprocess4 import extract_file_on_batch
def extract_file_on_batch(filename,target_filename, timeSteps = 10, predict_fur = 1):
    raw_data = csv.reader(open(filename))
    writeFile = file(target_filename, 'a')
    FileWriter = csv.writer(writeFile)

    raw_data_len = 0
    feature_list = [4, 6, 8, 13, 14, 23, 24, 29]
    feature_num = len(feature_list)
    lable_set = [13,14]
    #used to count trends
    up = 0
    down = 0

    raw_data = csv.reader(open(filename))
    header = next(raw_data)
    
    dataList = []
    yList = []
    resList = []
    # extract data to dataList
    for row in raw_data:
        cur = []
        for i in range(feature_num):
            cur.append(float(row[feature_list[i]]))
        yList.append( (float(row[lable_set[0]]) + float(row[lable_set[1]]) )/ 2)
        dataList.append(cur)

    print len(dataList), len(yList)
    print dataList[0]
    print dataList[1]
    # scaler
    min_max_scaler = preprocessing.MinMaxScaler()
    dataList_scaled = min_max_scaler.fit_transform(dataList)

    if (len(dataList_scaled) >= 2*timeSteps):
        cur = []
        y = 0
        flag = 0
        #forms first [timeSteps], then iteratively
        for i in range(0,timeSteps):
            for j in range(0,feature_num):
                cur.append(dataList_scaled[i][j])
        if (yList[timeSteps-1] < yList[timeSteps-1+predict_fur]):
            up+=1
            y = 1
        elif (yList[timeSteps-1] == yList[timeSteps-1+predict_fur]):
            #print 'flat', timeSteps-1, 2*timeSteps-1
            t = timeSteps+1
            while (t < len(yList) and yList[t]==yList[timeSteps-1]):
                t+=1
            if (t==len(yList)):
                flag = 1
            elif (yList[t] > yList[timeSteps-1]):
                up+=1
                y = 1
            else:
                down+=1
                y = 0
            #print 'finally,',t
            #if equals to len(dataList), do nothing
        else:
            down+=1
            y = 0
        if (flag==0):
            resList.append(cur+[y])
        # the rest
        flag = 0
        for i in range(timeSteps, len(dataList_scaled)-predict_fur):
            #i is the appended one
            cur = cur[feature_num:]
            for j in range(feature_num):
                cur.append(dataList_scaled[i][j])
            if (yList[i+predict_fur] > yList[i]):
                up+=1
                y = 1
            elif (yList[i+predict_fur] < yList[i]):
                y = 0
                down+=1
            else:
                #print 'flat', i, i+timeSteps
                t = i+predict_fur+1
                while (t < len(yList) and yList[t]==yList[i]):
                    t+=1
                if (t==len(yList)):
                    flag=1
                elif (yList[t] > yList[i]):
                    up+=1
                    y = 1
                else:
                    down+=1
                    y = 0
                #print 'finally', t

            if (flag==0):
                resList.append(cur+[y])
    
    print len(resList), 'generated'
    FileWriter.writerows(resList)
    print resList[0]
    print resList[1]
    print resList[2]
    print 'ups:',up, 'downs', down
    writeFile.close()
    #close file


def get_file_list(dir):
    if (os.path.isfile(dir)):
        return []
    else:
        res = []
        for s in os.listdir(dir):
            res.append(s)
    return res
def main():
    file_list = get_file_list('raw/')
    
    print len(file_list)
    target_filename = 'generated_10_tick_1_tick/5-27.csv'
    for file_name in file_list[:1]:
        #target_filename = 'generated_10_tick_1_tick/'+file_name
    	file_name = 'raw/'+file_name
        
        print file_name
        print target_filename
        extract_file_on_batch(file_name,target_filename)
        
        #break

main()
