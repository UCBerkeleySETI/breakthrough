import pandas as pd
from astropy.time import Time
import csv


print("Paste_file_location")
file_location = input()

data = pd.read_csv (file_location)
data.columns = ['id', 'project', 'time', 'target_name', 'ra', 'decl', 'center_freq', 'file_type', 'size', 'md5sum', 'url']

data = data.loc[data['project'] == 'GBT']
data = data.loc[data['file_type'] == 'HDF5']
data.to_csv('database_gbt_h5.CSV', sep=',')

data_numpy = data.values[:,1:]

links_list =[]
for i in range(0,data_numpy.shape[0]): 
  string = data_numpy[i,9]
  string = string.replace('fine.h5','mid.h5')
  string = string.replace('time.h5','mid.h5')
  links_list.append(string)
url_list=set(links_list)

out = csv.writer(open("database_midres_h5.csv","w"), delimiter=',',quoting=csv.QUOTE_ALL)
out.writerow(url_list)