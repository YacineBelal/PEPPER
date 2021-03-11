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
from keras.optimizers import SGD, Adam
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
   

    def handleMessage(self, msg):
        if msg.getName() == 'PreparationPhase': 
            self.vector = msg.user_ratings
            self.id_user = msg.id_user
            self.num_items = msg.num_items
            self.num_users = msg.num_users
            self.item_input, self.labels, self.user_input = self.my_dataset(True)
           
            
        elif msg.getName() == 'FirstRound': 
            
            self.model = util.get_model(self.num_items,self.num_users) # giving the size of items as par
            self.model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy')
            self.model.set_weights(msg.weights)
            hist = self.model.fit([self.user_input, self.item_input], #input
                        np.array(self.labels), # labels 
                        batch_size=32, nb_epoch=1, verbose=1, shuffle=True)
    
            weights = WeightsMessage('Node_weights')
            weights.weights.append(self.model.get_weights())
            #EV << "I'm the node " << self.id_user << '\n'
            #EV << 'My weights : ' << weights.weights[0]
            weights.positives_nums = self.positives_nums            
            self.send(weights, 'nl$o',0)
        
        elif msg.getName() == 'Round': 
              # advanced rounds 
            self.item_input, self.labels, self.user_input = self.my_dataset(True)
            self.model.set_weights(msg.weights)
            hist = self.model.fit([self.user_input, self.item_input], #input
                        np.array(self.labels), # labels 
                        batch_size=32, nb_epoch=1, verbose=1, shuffle=True)
            weights = WeightsMessage('Node_weights')
            weights.weights.append(self.model.get_weights())
            #EV << "I'm the node " << self.id_user << '\n'
            #EV << 'My weights : ' << weights.weights[0]
            weights.positives_nums = self.positives_nums            
            self.send(weights, 'nl$o',0)
        
    
    
    
    
    def my_dataset(self,num_negatives = 4,sample = True):
        item_input = []
        labels = []
        positives_nums = 0
        user_input = []
        for i in self.vector:
            item_input.append(i)
            labels.append(1)
            user_input.append(self.id_user)
            positives_nums = positives_nums + 1
            if sample:
                for i in range(num_negatives):
                    j = np.random.randint(self.num_items)
                    while j in self.vector:
                        j = np.random.randint(self.num_items)
                    user_input.append(self.id_user)
                    item_input.append(j)
                    labels.append(0)
        if not sample:
            for i in range(self.num_items):
                if i not in item_input:
                    user_input.append(self.id_user)
                    item_input.append(i)
                    labels.append(0)

        
        self.positives_nums = positives_nums 
        
        return np.array(item_input), np.array(labels), np.array(user_input)

    