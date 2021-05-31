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
train ,testRatings, testNegatives,validationRatings, validationNegatives,= dataset.trainMatrix, dataset.testRatings, dataset.testNegatives,dataset.validationRatings, dataset.validationNegatives



def get_user_vector(train,user = 0):
    positive_instances = []
    for (u,i) in train.keys():
        if u == user:
            positive_instances.append(i)
        if u  > user :
            break

    return positive_instances

def get_user_test_set(testRatings,testNegatives,user):
    personal_testRatings = []
    personal_testNegatives = []
    for i in range(len(testRatings)):
        idx = testRatings[i][0]
        if idx == user:
            personal_testRatings.append(testRatings[i])
            personal_testNegatives.append(testNegatives[i])
        elif idx > user:
            break
    return personal_testRatings,personal_testNegatives




class Node(cSimpleModule):
    def initialize(self):
        # initialization phase in which number of rounds, model age, data is read, model is created, connecter peers list is created
        self.rounds = 800
        self.positives_nums = 0
        self.vector = np.empty(0)
        self.labels = np.empty(0)
        self.item_input = np.empty(0)
        self.age = 1
        self.num_items = 1682 # TO DO train.shape[1] automate this, doesn't work since the validation data has been added because one item is present there and not in training
        self.num_users = train.shape[0]
        self.id_user = self.getIndex()
        self.vector = get_user_vector(train,self.id_user)
        self.testRatings, self.testNegatives = get_user_test_set(testRatings,testNegatives,self.id_user)
        self.validationRatings, self.validationNegatives = get_user_test_set(validationRatings,validationNegatives,self.id_user)
        self.item_input, self.labels, self.user_input = self.my_dataset()
        self.model = util.get_model(self.num_items,self.num_users) # each node initialize its own model 
        self.model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy')
        self.period_message = cMessage('period_message')
        self.peers = []
        for i in range(self.gateSize("no")):
            if self.gate("no$o",i).isConnected():
                self.peers.append(i)
        self.update()
        self.scheduleAt(simTime() + random.randint(1,6),self.period_message)
   

    def handleMessage(self, msg):
        # periodic self sent message used as a timer by each node to diffuse its model 
        if msg.getName() == 'period_message':
            
            if self.rounds > 0 :
                self.diffuse_to_peer()
                self.rounds = self.rounds - 1
                self.scheduleAt(simTime() + random.randint(1,6),self.period_message)
            else: # if number of rounds has been acheived, we can evaluate the model both locally and globally
                lhr, lndcg = self.evaluate_local_model(False,False)
                print('node : ')
                print(self.getIndex())
                print('local HR =  ')
                print(lhr)
                print(' local NDCG = ') 
                print(lndcg)

                hr, ndcg = self.evaluate_local_model(True)
                print('node : ')
                print(self.getIndex())
                print('global HR =  ')
                print(hr)
                print(' global NDCG = ') 
                print(ndcg)

        # messsage containing a neighbour's mode is received here    
        elif msg.getName() == 'Model': 
            # aggregating the weights received with the local weigths (ponderated by the model's age aka number of updates) before making some gradient steps 
            self.merge(msg)
            self.item_input, self.labels, self.user_input = self.my_dataset()
            self.update()
            self.age = max(self.age,msg.age)
    
    # evaluation method : can be on the whole dataset for a general hit ratio but is usually on the local dataset.
    # locally, it can be either on validation data (during the aggregation) or on test data at the end of training
    def evaluate_local_model(self,all_dataset = False, validation=True):
        topK = 20
        evaluation_threads = 2 #mp.cpu_count()
        if not all_dataset:
            if validation :
                if len(self.validationRatings) > 2:
                    (hits, ndcgs) = evaluate_model(self.model, self.validationRatings, self.validationNegatives, topK, evaluation_threads)
                else:
                    return 0,0
            else:
                if len(self.testRatings) > 1:
                    (hits, ndcgs) = evaluate_model(self.model, self.testRatings, self.testNegatives, topK, evaluation_threads)
                else:
                    return 0,0 
        else:
            (hits, ndcgs) = evaluate_model(self.model, testRatings, testNegatives, topK, evaluation_threads)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
       
        return hr, ndcg


    # select a random peer and send its model weights and its age to it
    def diffuse_to_peer(self):
        if(len(self.peers) >= 1):
            peer = random.randint(0,len(self.peers)-1)
            weights = WeightsMessage('Model')
            weights.weights = self.model.get_weights().copy()
            weights.age = self.age       
            self.send(weights, 'no$o',self.peers[peer])
    
    # making 4 gradient steps
    def update(self):
        hist = self.model.fit([self.user_input, self.item_input], #input
                        np.array(self.labels), # labels 
                        batch_size=len(self.user_input), nb_epoch=4, verbose=2, shuffle=True)
        self.age = self.age + 1
    
    def merge(self,message_weights):
        weights = message_weights.weights
        local_weights = self.model.get_weights().copy()
        local_weights[:] =  [ (a * self.age + b * message_weights.age) / (self.age + message_weights.age) for a,b in zip(local_weights,weights)]
        self.model.set_weights(local_weights)

    # using the data at disposition (which is the positives ratings) we create the negative ratings that goes with, in order to have a small local dataset
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

    