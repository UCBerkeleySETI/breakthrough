import numpy as np
from random import seed
from random import random
import time
from blimpy import Waterfall
import pylab as plt
from astropy import units as u
from matplotlib import pyplot as plt
import setigen as stg
from sklearn.model_selection import train_test_split

class synthetic(object):
    def __init__(self):
        self.name = "synthetic generator"
    def generate(self, total_num_samples, data, intensity = 0.7, test=False, labels= True):
        start_time = time.time()
        num_channels = 32
        # Prepare the target set
        fchans = num_channels
        tchans = 32
        df = 2.7939677238464355*u.Hz
        dt = 18.25361108*u.s
        fch1 = 6095.214842353016*u.MHz
        tune_base = intensity
        generated_signals = np.zeros((total_num_samples,data.shape[1],1, num_channels), dtype=float)

        for i in range(0,total_num_samples):
            base = data[i,:,0,:] 
            start = int(random()*(fchans-15))
            artifical_radio = np.zeros((32,32))
            period = random()
            drifrate = (random()-0.5)*10**int(random()*-2) *u.Hz/u.s
            choose = random()
            amplitude  = random()
            tune = tune_base * (random()+0.3)

            for blur in range(0,2):
                frame = stg.Frame(fchans=fchans,
                            tchans=tchans,
                            df=df,
                            dt=dt,
                            fch1=fch1)
                signal = frame.add_signal(stg.constant_path(f_start=frame.fs[start+10],
                                                            drift_rate=drifrate),
                                        stg.constant_t_profile(level=tune*blur),
                                        stg.gaussian_f_profile(width=(20-5*blur)*u.Hz),
                                        stg.constant_bp_profile(level=tune*blur))
                artifical_radio = np.add(frame.get_data(), artifical_radio)

            artifical_radio = np.add(base, artifical_radio)
            artifical_radio = np.reshape(artifical_radio, (tchans,1,fchans))
            generated_signals[i,:,:,:]= artifical_radio

            if i %int(total_num_samples/2) ==0 and test:
                fig = plt.figure(figsize=(10, 6))
                plt.imshow(generated_signals[i,:,0,:], aspect='auto')
                plt.colorbar()
 
        # Label the dataset
        supervised_true =  np.concatenate((np.ones((generated_signals.shape[0],1),dtype='int64'),np.zeros((generated_signals.shape[0],1),dtype='int64')), axis=1)
        supervised_false = np.concatenate((np.zeros((total_num_samples,1),dtype='int64'),np.ones((total_num_samples,1),dtype='int64')), axis=1)
        label = np.concatenate((supervised_true, supervised_false))
        supervised_dataset = np.concatenate((generated_signals, data[0:total_num_samples,:,:,:]))
        print(label.shape)
        print(supervised_dataset.shape)
        print("Synethtic Generation Execution Time: "+ str(time.time()-start_time))

        X_train_supervised, X_test_supervised, y_train_supervised, y_test_supervised = train_test_split(supervised_dataset, label, test_size=0.2, random_state=2)
        return X_train_supervised, X_test_supervised, y_train_supervised, y_test_supervised 

       
    