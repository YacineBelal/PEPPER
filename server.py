from pyopp import cSimpleModule, cMessage, EV, simTime
import numpy as np
from Dataset import Dataset
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializations
from keras.models import Sequential, Model, load_model, save_model
from keras.layers import Embedding, Flatten
from keras.regularizers import l2
import functools 
from dataMessage import dataMessage
from WeightsMessage import WeightsMessage
import utility as util
from evaluate import evaluate_model
import random

dataset = Dataset('ml-100k')
train ,testRatings, testNegatives= dataset.trainMatrix, dataset.testRatings, dataset.testNegatives


def get_user_vector(train,user = 0):
    positive_instances = []
    for (u,i) in train.keys():
        if u == user:
            positive_instances.append(i)
        if u  > user :
            break

    return positive_instances

def get_items_instances(train):
    item_input = []
    for i in range(train.shape[1]):
        item_input.append(i)
    
    return item_input

def create_structure(weights):
        struct = []
        for i in weights:
            struct2 = []
            for j in i:
                struct2.append(0)
            struct.append(np.array(struct2))
            
        return struct 



class Server(cSimpleModule):
    def initialize(self):
        self.number_rounds = 1000
        self.message_round = cMessage('message_round')
        self.message_averaging = cMessage('StartAveraging')
        self.global_weights = []
        self.total_samples = [] 
        self.all_participants = [i for i in range(self.gateSize('sl'))]
        self.oldparticipants = []
        self.num_items = train.shape[1]
        self.num_users = train.shape[0]
        self.model = util.get_model(self.num_items,self.num_users) 
        
        EV << 'global model weights at the start :' << self.model.get_weights()[0]
       
        topK = 10
        
        
        (hits, ndcgs) = evaluate_model(self.model, testRatings, testNegatives, topK, 1,self.num_items)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        print('HR =  \n')
        print(hr)
        print(' NDCG = \n') 
        print(ndcg)
        

        self.global_weights.append(self.model.get_weights())
        if self.getName() == 'server':
            self.diffuse_message('PreparationPhase')
             
       

    def handleMessage(self, msg):
        if msg.isSelfMessage():
            if msg.getName() == 'StartAveraging':
                self.fedAvg()
            else: 
                self.diffuse_message('FirstRound')
                self.global_weights = []   
       
        elif msg.getName() == 'Node_weights':
            # averaging weight , taking account the number of items the user has rated to weight the average
            # an averaging based on fedavg
            self.global_weights.append(msg.weights[0])
            self.total_samples.append(msg.positives_nums)
            
    def diffuse_message(self,str,sample = False):
        participants = self.sampling()  if sample else self.all_participants
        # participants = self.all_participants
        if str =='FirstRound':
            print('******************** FirstRound *************************')
            for i in participants:
                weights = WeightsMessage(str)
                weights.weights = self.model.get_weights()
                self.send(weights,'sl$o',i)
            self.scheduleAt(simTime() + self.gateSize('sl'),self.message_averaging)
            self.number_rounds = self.number_rounds - 1
        elif str == 'PreparationPhase':
            for i in participants:
                msg = dataMessage('PreparationPhase')
                msg.user_ratings = np.array(get_user_vector(train,i))
                msg.num_items = self.num_items
                msg.num_users = self.num_users
                msg.id_user = i
                self.send(msg, 'sl$o',i)
            self.scheduleAt(simTime() + 2,self.message_round)
        else:
            print('******************** Round number ************************* \n')
            print(self.number_rounds)
            for i in participants:
                weights = WeightsMessage(str)
                weights.weights = self.model.get_weights()
                self.send(weights,'sl$o',i)
            self.scheduleAt(simTime() + self.gateSize('sl'),self.message_averaging)
            self.number_rounds = self.number_rounds - 1
            
       

    
    def fedAvg(self):
            sum = functools.reduce(lambda a,b : a+b,self.total_samples)
            j = 0    
            # averaging
            for w in self.global_weights:
                for i in range(len(w)):
                    w[i] = self.total_samples[j] * w[i] / sum
                j = j + 1
            
            # summing and then combining in one entity of weights
            new_weights = self.global_weights[0].copy()
            for i in range(1,len(self.global_weights)):
                new_weights = [ np.add(x,y) for x, y in zip(self.global_weights[i], new_weights)]
            self.global_weights = []
            self.model.set_weights(new_weights)
            EV << 'global model weights in the middle :' << self.model.get_weights()[0]
            self.total_samples = []
            if self.number_rounds > 0:
                self.diffuse_message('Round',True)
            else:
                #self.model.save_weights('model.h5', overwrite=True)
                
              
                topK = 10
                
                (hits, ndcgs) = evaluate_model(self.model, testRatings, testNegatives, topK, 1,self.num_items)
                hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
                print('HR =  \n')
                print(hr)
                print(' NDCG = \n') 
                print(ndcg)
        

               


    def sampling(self,num_samples = 200):
        if num_samples > self.gateSize('sl'):
            raise Exception("ERROR : size of sampling set is bigger than total samples of clients") 
        else:
            size =  self.gateSize('sl')
            participants = []
            for i in range(num_samples):
                p = random.randint(0,size-1)
                while p in participants or p in self.oldparticipants:
                    p = random.randint(0,size-1)
                participants.append(p)
        self.oldparticipants = participants.copy()
        return participants



             
    
