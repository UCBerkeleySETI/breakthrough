import tensorflow as tf 
import keras
from keras.models import load_model
from keras.utils import np_utils
import os, os.path
from keras.models import Model
from keras import backend as K


class save_model(object):
    def __init__(self):
        self.name ="name"

    def freeze_layers(self, model):
        for i in model.layers:
            i.trainable = False
            if isinstance(i, Model):
                self.freeze_layers(i)
        return model

    def save(self, model):
        freeze = self.freeze_layers(model)
        freeze.save('encoder_injected_model.h5')