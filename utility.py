import numpy as np
import random as rand
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializations
from keras.models import Sequential, Model, load_model, save_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten
from keras.optimizers import SGD
from keras.regularizers import l2
from Dataset import Dataset


def get_model(num_items):
    # Input variables
    user_input = Input(shape=(8,), dtype='float32', name = 'user_input')
    item_input = Input(shape=(8,), dtype='float32', name = 'item_input')

    
    
    
    # Element-wise product of user and item embeddings 
    predict_vector = merge([user_input, item_input], mode = 'mul')
    
    # Final prediction layer
    #prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name = 'prediction')(predict_vector)
    
    model = Model(input=[user_input, item_input],output=prediction)

    
    return model
