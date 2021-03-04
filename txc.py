from pyopp import cSimpleModule, cMessage, EV
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

from WeightsMessage import WeightsMessage
from dataMessage import dataMessage
import utility as util



def create_user_embedding(factors = 8):
    
    user_input = []
    for i in range(factors):
        user_input.append(rand.uniform(-1,1))
    return user_input



class Node(cSimpleModule):
    def initialize(self):
        self.positives_nums = 0
        self.vector = np.empty(0)
        self.labels = np.empty(0)
        self.item_input = np.empty(0) 
        if self.getName() == 'mafihech':
            self.send(cMessage('tnakaat'), 'nl',0)

    def handleMessage(self, msg):
        if msg.getName() == 'PreparationPhase': 
            self.vector = msg.user_ratings
            self.id_user = msg.id_user
            self.item_input, self.labels, self.user_input = self.my_dataset(msg.items_embeddings)
            EV << 'got a '<< msg.getName() <<'  message from server\n' 
            
        elif msg.getName() == 'FirstRound': 
        
            self.model = util.get_model(self.item_input.shape[0]) # giving the size of items as par
            self.model.set_weights(msg.weights)
            self.model.compile(optimizer=SGD(lr=0.001), loss='binary_crossentropy')
            hist = self.model.fit([self.user_input, self.item_input], #input
                        np.array(self.labels), # labels 
                        batch_size=2, nb_epoch=1, verbose=1, shuffle=True)
       
            weights = WeightsMessage('Node_weights')
            weights.weights.append(self.model.get_weights())
            EV << 'len of weights : ' << len(weights.weights[0])
            weights.positives_nums = self.positives_nums            
            self.send(weights, 'nl$o',0)
        
        elif msg.getName() == 'Round': 
              # advanced rounds 
            
            self.model.set_weights(msg.weights)
            hist = self.model.fit([self.user_input, self.item_input], #input
                        np.array(self.labels), # labels 
                        batch_size=2, nb_epoch=1, verbose=1, shuffle=True)
            weights = WeightsMessage('Node_weights')
            weights.weights.append(self.model.get_weights())
            weights.positives_nums = self.positives_nums            
            self.send(weights, 'nl$o',0)
        
    
    
    
    
    def my_dataset(self,items_embeddings):
        item_input = []
        labels = []
        positives_nums = 0
        user_input = []
        for i in self.vector:
            item_input.append(items_embeddings[i])
            labels.append(1)
            user_input.append(self.id_user)
            positives_nums = positives_nums + 1
        self.positives_nums = positives_nums 
        for i in range(items_embeddings.shape[0]):
            if i not in self.vector:
                user_input.append(self.id_user)
                item_input.append(items_embeddings[i])
                labels.append(0)

        return np.array(item_input), np.array(labels), np.array(user_input)

    