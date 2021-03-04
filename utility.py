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


def init_normal(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)


def get_model(num_items):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(8,), dtype='float32', name = 'item_input')

    MF_Embedding_User = Embedding(input_dim = num_items, output_dim = 8, name = 'user_embedding',
                                  init = init_normal, W_regularizer = l2(0.001), input_length=1)
    
    user_latent = Flatten()(MF_Embedding_User(user_input))
    
    # Element-wise product of user and item embeddings 
    predict_vector = merge([user_latent, item_input], mode = 'mul')
    
    # Final prediction layer
    #prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name = 'prediction')(predict_vector)
    
    model = Model(input=[user_input, item_input],output=prediction)

    
    return model
