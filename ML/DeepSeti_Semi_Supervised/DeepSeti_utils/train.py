import keras
from keras.models import Sequential 
from keras.layers.core import Activation, Flatten
from keras.optimizers import SGD,RMSprop,adam
from keras.models import load_model
from keras.losses import binary_crossentropy
from keras.utils import np_utils
import numpy as np
from keras import losses
from keras.models import Model
from keras import backend as K
from  keras.backend import expand_dims
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

class train(object):
    def __init__(self):
        self.name="name"
        
    def train_model(self, epoch, inputs, encode, feature_encode, decoder, latent_encode, 
                X_train_unsupervised, X_test_unsupervised, X_train_supervised, 
                X_test_supervised, y_train_supervised, y_test_supervised, batch_size=4096):

        encoder_final = Model(inputs, feature_encode(encode(inputs)), name='encoder_training')
        AutoEncoder= Model(inputs, decoder(latent_encode(encode(inputs))), name='autoencoder')
        history_encoder_tracker = np.zeros((epoch))
        history_unsupervised_tracker = np.zeros((epoch))
        sgd_encoder = SGD(lr=0.1, clipnorm=1, clipvalue=0.5)
        sgd_unsupervised = SGD(lr=0.1, clipnorm=1, clipvalue=0.5)
        encoder_final.compile(loss='binary_crossentropy', optimizer=sgd_encoder,  metrics=['acc'])
        AutoEncoder.compile(loss='mean_squared_error', optimizer=sgd_unsupervised,  metrics=['acc'])

        for i in range(0,epoch):
        
        # encoder_final.compile(loss='mean_squared_error', optimizer=sgd_encoder,  metrics=['acc'])
            if i%2==0:
                print("--------------ENCODER TRAIN--------------" + str(i))
                mc = ModelCheckpoint('encoder_model.h5', monitor='val_loss', mode='min', save_best_only=True)
                history_encoder = encoder_final.fit(X_train_supervised, y_train_supervised, batch_size=1024, epochs=40, 
                    validation_data=(X_test_supervised, y_test_supervised), callbacks=[mc])
                print()
                print()
            # history_encoder_tracker[i] = history_encoder.history['loss']
            print("--------------UNSUPERVISED TRAIN--------------" + str(i))
            # AutoEncoder.compile(loss='mean_squared_error', optimizer=sgd_unsupervised,  metrics=['acc'])

            mc = ModelCheckpoint('model.h5', monitor='val_loss', mode='min', save_best_only=True)
            history_unsupervised = AutoEncoder.fit(X_train_unsupervised, X_train_unsupervised,  batch_size=batch_size, epochs=1, 
                validation_data=(X_test_unsupervised, X_test_unsupervised), callbacks=[mc])
        # encoder_injected = Model(inputs, feature_encode(encoder(inputs)), name='Generator')
        # model_freezed = self.freeze_layers(encoder_injected)
        return  Model(inputs, feature_encode(encode(inputs)), name='Generator')

