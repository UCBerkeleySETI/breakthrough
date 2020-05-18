from google.colab import drive
import wget
drive.mount('/content/drive/')
import pandas as pd
import requests
from DeepSeti import DeepSeti
import os 


data = pd.read_csv (r'/content/drive/My Drive/Deeplearning/SETI/database.csv')
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
short_list = []

for k in range(0,int(len(links_list)/3)):
  short_list.append(links_list[k*3])


DeepSeti = DeepSeti()

for i in range(100,200):
  try:
    print("Downloading "+ str(i))
    file_download = wget.download(short_list[i])
    print(file_download)
    print("finished downloading")
    DeepSeti.prediction(model_location='/content/encoder_injected_model_Cudda.h5', 
                    test_location='/content/'+file_download, 
                    anchor_location='/content/GBT_58402_66967_HIP66130_mid.h5', 
                    top_hits=1, target_name=file_download,
                    output_folder='/content/drive/My Drive/Deeplearning/SETI/output_folder/')
    os.remove('/content/'+file_download)
    print("Search Execution Complete")
  except:
    try:
      os.remove('/content/'+short_list[i].replace('http://blpd7.ssl.berkeley.edu/dl2/', ''))
    except:
      print("Execution stack cleanered")
    print("Dataset "+ short_list[i].replace('http://blpd7.ssl.berkeley.edu/dl2/', '')+" doesn't exist --------- skipped!")