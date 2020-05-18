import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import load_model
import os, os.path
from scipy.io import wavfile
from keras.models import Model
from keras import backend as K
from random import random

from  copy import deepcopy
import cupy as cp

class predict(object):

    def __init__(self, test, model_loaded, anchor=None):
        self.anchor = anchor
        self.test = test
        self.f_start = 0
        self.f_stop = 0 
        self.n_chan_width = 0
        self.encoder_injected = model_loaded
        self.values= np.zeros(self.test.shape[0])
    
    def compute_distance_preloaded(self, anchor):
        """
        Method helps compute the MSE between two N-d vectors and is used to make the
        Helps facilitate fast computation.                 
        """
        check = self.encoder_injected.predict(self.test)
        for j in range(0, self.test.shape[0]-1):
            # index = int(random()*10)
            index = 0
            self.values[j]=(np.square(np.subtract(anchor[index:index+1,:], check[j:j+1,:]))).mean()   
        return self.values

    def compute_distance(self):
        """
        Method helps compute the MSE between two N-d vectors and is used to make the
        Helps facilitate fast computation.                 
        """
        
        check = self.encoder_injected.predict(self.test)
        anchor = self.encoder_injected.predict(self.anchor)
        for j in range(0, self.test.shape[0]-1):
            # index = int(random()*10)
            index = 0
            self.values[j]=(np.square(np.subtract(anchor[index:index+1,:], check[j:j+1,:]))).mean()   
        return self.values

    def compute_distance_cupy(self):
        """
        Method helps compute the MSE between two N-d vectors and is used to make the
        Helps facilitate fast computation.                 
        """
        check = cp.array(self.encoder_injected.predict(self.test))
        anchor = cp.array(self.encoder_injected.predict(self.anchor))
        for j in range(0, self.test.shape[0]-1):
            # index = int(random()*10)
            index = 0
            self.values[j]=(cp.square(cp.subtract(anchor[index:index+1,:], check[j:j+1,:]))).mean()   
        return self.values

    def convert_np_to_mhz(self, np_index, f_stop,f_start, n_chans):
        width = (f_stop-f_start)/n_chans
        return width*np_index + f_start

    def max_index(self, f_start, f_stop, n_chan_width, top=3):
        self.f_start = f_start
        self.f_stop = f_stop 
        self.n_chan_width = n_chan_width
        top_hits = []
        f_chan_low = [1190,2290]
        f_chan_high = [1350, 2360]
        copy = deepcopy(self.values)
        i=0
        while i <top:
            hit = np.argmax(copy)
            hit_freq = self.convert_np_to_mhz(hit, self.f_start,self.f_start, self.n_chan_width)
            if hit_freq< f_chan_low[0] or (hit_freq> f_chan_high[0] and hit_freq< f_chan_low[1]) or hit_freq>f_chan_high[1]:
                top_hits.append(hit)
                i+=1
            copy[hit]=0
        return top_hits
    def max_index_nofilter(self, top=3):
        top_hits = []
        copy = self.values
        for i in range(0, top):
            hit = np.argmax(copy)
            top_hits.append(hit)
            copy[hit]=0
        return top_hits

    def min_index(self, top=3):
        """
        This method finds the minimum distance - used for reverse image search for signals.
        This implements the same logic as the max_index search.
        """
        top_hits = []
        copy = deepcopy(self.values)
        for i in range(0, top):
            hit = np.argmin(copy)
            top_hits.append(hit)
            copy[hit]=0
        return top_hits