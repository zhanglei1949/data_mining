#########################
#time step = 20,  generate valid data
# no flat,
#########################
import csv
import random
import math

def extract_file_on_batch(filename,target_filename, batch_size = 10000, timeSteps = 20):
    raw_data = csv.reader(open(filename))
    raw_data_len = 0
    num_batches = 0
    feature_list = [4,5,6,7,8,23,24,29]
    feature_num = len(feature_list)
    lable_set = [13,14]
    #used to count trends
    up = 0
    down = 0

    #1. first read the data by column, store the feature mean and variance
    header = next(raw_data)
    column_data = zip(*raw_data)
    raw_data_len = len(column_data[0])

    feature_var = [0.0]*feature_num
    feature_mean = [0.0]*feature_num

    for i in range(feature_num):
        for j in range(raw_data_len):
            feature_mean[i]+= float(column_data[feature_list[i]][j])
        print i, 'th feature mean before div', feature_mean[i]
        feature_mean[i] /= raw_data_len

        for j in range(raw_data_len):
            feature_var[i] += (float(column_data[feature_list[i]][j])- feature_mean[i])**2

        print i, 'th feature var before div', feature_var[i]
        feature_var[i] /= raw_data_len
        print i, 'th feature mean', feature_mean[i]
        print i, 'th feature var', feature_var[i]
    print raw_data_len,'samples'


    #2. reload again, , open target file for each iteration, read in some data, process and save
    raw_data = csv.reader(open(filename))
    header = next(raw_data)
    start_pos = 0
    end_pos = 0
    writeFile = file(target_filename, 'w+')
    FileWriter = csv.writer(writeFile)
    #FileWriter.writerows(towrite)
    ############################
    #raw_data_len = 130
    #timeSteps = 20
    #batch_size = 70
    ##############################

    print 'raw data length', raw_data_len
    print 'batch size', batch_size
    #print 'need ', num_batches, 'batches'
    print 'feature number', feature_num
    ind = 0
    while (ind < raw_data_len):
        dataList = []
        resList = []
        cnt = 0
        #print 'start from', ind
        while (cnt < batch_size and ind < raw_data_len):
            cnt+=1
            ind+=1
            line = next(raw_data)
            tmp = []
            for i in range(feature_num):
                #tmp.append(ind)
                tmp.append((float(line[feature_list[i]])- feature_mean[i])/feature_var[i])
            tmp.append(( float(line[lable_set[0]])+float(line[lable_set[1]]) )/2)
            #print tmp
            dataList.append(tmp)
        #combine serval instance together forms a training instance
        #print 'end at', ind-1
        print len(dataList)
        if (len(dataList) < 2*timeSteps):
            break
            #can not form a train sample
        else:
            cur = []
            y = 0
            flag = 0
            #forms first [timeSteps], then iteratively
            for i in range(0,timeSteps):
                for j in range(0,feature_num):
                    cur.append(dataList[i][j])
            if (dataList[timeSteps-1][feature_num] < dataList[2*timeSteps-1][feature_num]):
                up+=1
                y = 1
            elif (dataList[timeSteps-1][feature_num] == dataList[2*timeSteps-1][feature_num]):
                #print 'flat', timeSteps-1, 2*timeSteps-1
                t = 2*timeSteps
                while (t < len(dataList) and dataList[t][feature_num]==dataList[timeSteps-1][feature_num]):
                    t+=1
                if (t==len(dataList)):
                    flag = 1
                elif (dataList[t][feature_num] > dataList[timeSteps-1][feature_num]):
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
            for i in range(timeSteps, len(dataList)-timeSteps):
                #i is the appended one
                cur = cur[feature_num:]
                for j in range(feature_num):
                    cur.append(dataList[i][j])
                if (dataList[i+timeSteps][feature_num] > dataList[i][feature_num]):
                    up+=1
                    y = 1
                elif (dataList[i+timeSteps][feature_num] < dataList[i][feature_num]):
                    y = 0
                    down+=1
                else:
                    #print 'flat', i, i+timeSteps
                    t = i+timeSteps+1
                    while (t < len(dataList) and dataList[t][feature_num]==dataList[i][feature_num]):
                        t+=1
                    if (t==len(dataList)):
                        flag=1
                    elif (dataList[t][feature_num] > dataList[i][feature_num]):
                        up+=1
                        y = 1
                    else:
                        down+=1
                        y = 0
                    #print 'finally', t

                if (flag==0):
                    resList.append(cur+[y])
        #print len(resList[0])
        #print len(resList[1])
        #print len(resList[2])
            FileWriter.writerows(resList)
    print 'ups:',up, 'downs', down
    writeFile.close()
    #close file

def main():

    #filename = 'TK_m0000[s20160101 00000000_e20160110 00153000]20170410_1735_0.csv'
    filename = 'TK_rb0000[s20160101 00000000_e20160110 00153000]20170410_1722_0.csv'
    #writeFileName = 'valid.csv'
    target_filename = 'rb0000_valid_only_up_down.csv'
    extract_file_on_batch(filename, target_filename, batch_size = 10000, timeSteps = 20)


#main()
