# for batch data reading and saving
import csv
import os
from data_preprocess4 import extract_file_on_batch
def get_file_list(dir):
    if (os.path.isfile(dir)):
        return []
    else:
        res = []
        for s in os.listdir(dir):
            res.append(s)
    return res
def main():
    file_list = get_file_list('../data/')
    print len(file_list)
    for file_name in file_list:
        
        target_filename = '../data/new/'+file_name[:-4]+'_valid.csv'
	file_name = '../data/'+file_name
        if (file_name=='../data/new'):
            print 'directory'
            continue

        print file_name
        print target_filename
        if (os.path.isfile(target_filename)):
            print 'exits'
        else:
            extract_file_on_batch(file_name,target_filename)
        
        #break

main()
