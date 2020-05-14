import numpy as np
from random import seed
from random import random
import time
from blimpy import Waterfall
import pylab as plt
from astropy import units as u
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cupy as cp
import h5py

class DataProcessing(object):
    """
    Note: ONLY ACCEPTS MID RES FILES!!!!!!!

    Facilitates the data manipulation for the Deep learning algorithm. 
    Takes in  [tchans, 1, fchans] H5 or Filterbank file=> 
    - splits tchans into sections of 32 
    - splits fchans into sections of 32 
    
    returns numpy array
    """
    def __init__(self):
        self.name = 'name'
        self.f_stop = 0
        self.f_start = 0
        self.n_chans =0

    def load_data(self, file_location, normalize = True):
        start = time.time()
        """
        Single data file load. 

        - Splits channels into 32x1x32

        Optional to normalize data. => Data is normalized by factor of 
        5906562547.712 => empircally tested value to speed training. 

        Returns numpy array
        """

        if file_location == 'none':
            return None
        else:
            file_path =file_location
            obs = Waterfall(file_path, load_data =False)
            h5 = h5py.File(file_path, 'r')
            self.chan_stop_idx = obs.container.chan_stop_idx
            self.chan_start_idx = obs.container.chan_start_idx
            self.t_start = obs.container.t_start
            self.t_stop = obs.container.t_stop

            data = h5["data"][self.t_start:self.t_stop,:,self.chan_start_idx:self.chan_stop_idx]
            self.f_stop = obs.container.f_stop
            self.f_start = obs.container.f_start
            self.n_chans =obs.header[b'nchans']
            
            data_temp = np.zeros((8,32, 1, data.shape[2]))
            intervals = [0,32,64,96,128,160,192,224]
            count=0
            for k in intervals: 
                data_temp[count,:,:,:]=data[k:k+32,:,:]
                count+=1
            total_samples = 8*int(data.shape[2]/32)
            data = np.zeros((8*int(data.shape[2]/32),32,1,32))
            
            i=0
            count=0
            
            while i<total_samples:
                # print(data_temp[0,:,:,32*i:(i+1)*32].shape)
                data[i,:,:,:]= data_temp[0,:,:,32*count:(1+count)*32]
                data[i+1,:,:,:]= data_temp[1,:,:,32*count:(1+count)*32]
                data[i+2,:,:,:]= data_temp[2,:,:,32*count:(1+count)*32]
                data[i+3,:,:,:]= data_temp[3,:,:,32*count:(1+count)*32]
                data[i+4,:,:,:]= data_temp[4,:,:,32*count:(1+count)*32]
                data[i+5,:,:,:]= data_temp[5,:,:,32*count:(1+count)*32]
                data[i+6,:,:,:]= data_temp[6,:,:,32*count:(1+count)*32]
                data[i+7,:,:,:]= data_temp[7,:,:,32*count:(1+count)*32]
                count+=1
                i=i+8
            
            print("single Data load Execution: "+str(time.time()-start)+ " Sec")
            
            if normalize:
                data = data/ 5906562547.712
            return data
    
    def load_data_cupy(self, file_location, normalize = True):
        start = time.time()
        """
        Single data file load. THIS USES CUDA AND CUPY FOR DATA PROCESSING 

        - Splits channels into 32x1x32

        Optional to normalize data. => Data is normalized by factor of 
        5906562547.712 => empircally tested value to speed training. 

        Returns numpy array
        """

        if file_location == 'none':
            return None
        else:
            file_path =file_location
            obs = Waterfall(file_path, max_load=1)
            data = cp.array(obs.data)
            self.f_stop = obs.container.f_stop
            self.f_start = obs.container.f_start
            self.n_chans =obs.header[b'nchans']
            
            data_temp = cp.zeros((8,32, 1, data.shape[2]))
            intervals = [0,32,64,96,128,160,192,224]
            count=0
            for k in intervals: 
                data_temp[count,:,:,:]=data[k:k+32,:,:]
                count+=1
            total_samples = 8*int(data.shape[2]/32)
            data = cp.zeros((8*int(data.shape[2]/32),32,1,32))
            
            i=0
            count=0
            
            while i<total_samples:
                # print(data_temp[0,:,:,32*i:(i+1)*32].shape)
                data[i,:,:,:]= data_temp[0,:,:,32*count:(1+count)*32]
                data[i+1,:,:,:]= data_temp[1,:,:,32*count:(1+count)*32]
                data[i+2,:,:,:]= data_temp[2,:,:,32*count:(1+count)*32]
                data[i+3,:,:,:]= data_temp[3,:,:,32*count:(1+count)*32]
                data[i+4,:,:,:]= data_temp[4,:,:,32*count:(1+count)*32]
                data[i+5,:,:,:]= data_temp[5,:,:,32*count:(1+count)*32]
                data[i+6,:,:,:]= data_temp[6,:,:,32*count:(1+count)*32]
                data[i+7,:,:,:]= data_temp[7,:,:,32*count:(1+count)*32]
                count+=1
                i=i+8
            
            print("single Data load Execution: "+str(time.time()-start)+ " Sec")
            
            if normalize:
                data = data/ 5906562547.712
            return data

    def load_multiple_files(self, list_directory, normalize=None, test_image = False):
        """
        Load multiple data files. Uses single loading method to handle data processing
        Takes in a list that contains multiple directories. 
        returns a numpy array with normalized data
        """
        num_files = len(list_directory)
        tchans = 256
        data = np.zeros((1,32,1,32))
        for i in range(0, num_files):
            data_temp = self.load_data( list_directory[i], normalize = True)
            data = np.concatenate((data_temp,data), axis=0)
        
        if test_image:
            fig = plt.figure(figsize=(10, 6))
            plt.imshow(data[78848-1000,:,:,:],  aspect='auto')
            plt.colorbar()
        
        unsupervised_temp = shuffle(data[1:,:,:,:], random_state=3)
        X_train_unsupervised, X_test_unsupervised, y_train_unsupervised, y_test_unsupervised = train_test_split(unsupervised_temp, unsupervised_temp, test_size=0.2, random_state=2)
        return X_train_unsupervised, X_test_unsupervised