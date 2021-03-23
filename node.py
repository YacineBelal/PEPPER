from pyopp import cSimpleModule, cMessage, EV, simTime
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
import multiprocessing as mp
from WeightsMessage import WeightsMessage
from dataMessage import dataMessage
import utility as util
import random
from evaluate import evaluate_model
dataset = Dataset('ml-100k')
train ,testRatings, testNegatives= dataset.trainMatrix, dataset.testRatings, dataset.testNegatives


class Node(cSimpleModule):
    def initialize(self):
        self.rounds = 80
        self.positives_nums = 0
        self.vector = np.empty(0)
        self.labels = np.empty(0)
        self.item_input = np.empty(0)
        self.age = 0
        self.period_message = cMessage('period_message')
        self.peers = []
        for i in range(self.gateSize("no")):
            if self.gate("no$o",i).isConnected():
                self.peers.append(i)
   

    def handleMessage(self, msg):
        if msg.getName() == 'PreparationPhase': 
            self.vector = msg.user_ratings
            self.id_user = msg.id_user
            self.num_items = msg.num_items
            self.num_users = msg.num_users
            self.item_input, self.labels, self.user_input = self.my_dataset()
            self.model = util.get_model(self.num_items,self.num_users) # each node initialize its own model 
            self.model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy')
            self.scheduleAt(simTime() + 4,self.period_message)
            topK = 10
            evaluation_threads = mp.cpu_count()
            (hits, ndcgs) = evaluate_model(self.model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
            print('node : ')
            print(self.getName())
            print('HR =  \n')
            print(hr)
            print(' NDCG = \n') 
            print(ndcg)

        elif msg.getName() == 'period_message':
            
            peer = random.randint(0,len(self.peers)-1)
            
            weights = WeightsMessage('Model')
            weights.weights = self.model.get_weights().copy()
            weights.positives_nums = self.positives_nums
            weights.age = self.age       
            
            self.send(weights, 'no$o',self.peers[peer])
            if self.rounds > 0:
                self.scheduleAt(simTime() + 4,self.period_message)
                self.rounds = self.rounds - 1
            else:
                topK = 10
                evaluation_threads = mp.cpu_count()
                (hits, ndcgs) = evaluate_model(self.model, testRatings, testNegatives, topK, evaluation_threads)
                hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
                print('node : ')
                print(self.getIndex())
                print('HR =  \n')
                print(hr)
                print(' NDCG = \n') 
                print(ndcg)
                    
            weights.positives_nums = self.positives_nums     
        elif msg.getName() == 'Model': 
            self.merge(msg)
            self.update()
            self.age = self.age + 1
    
    
    
    def update(self):
        hist = self.model.fit([self.user_input, self.item_input], #input
                        np.array(self.labels), # labels 
                        batch_size=8, nb_epoch=1, verbose=0, shuffle=True)
    
    def merge(self,message_weights):
        weights = message_weights.weights
        local_weights = self.model.get_weights().copy()
        local_weights[:] =  [ (a + b) / 2 for a,b in zip(local_weights,weights)]
        # a classic and simple averaging for a start
        # local_weights[:] = [ x / 2 for x in local_weights]
        self.model.set_weights(local_weights)

    def my_dataset(self,num_negatives = 4):
        item_input = []
        labels = []
        user_input = []
        self.positives_nums = 0
        for i in self.vector:
            item_input.append(i)
            labels.append(1)
            user_input.append(self.id_user)
            self.positives_nums = self.positives_nums + 1
            for i in range(num_negatives):
                j = np.random.randint(self.num_items)
                while j in self.vector:
                    j = np.random.randint(self.num_items)
                user_input.append(self.id_user)
                item_input.append(j)
                labels.append(0)            
        return np.array(item_input), np.array(labels), np.array(user_input)

    