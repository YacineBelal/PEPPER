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

# a peer sampling service, used to initialize data distribution and then eventually to provide nodes with a peers (not yet the case)
class Service(cSimpleModule):
    def initialize(self):
        self.num_items = train.shape[1]
        self.num_users = train.shape[0]
        if self.getName() == 'service':
            self.diffuse_message('PreparationPhase')
             
       

    def handleMessage(self, msg):
        
        if msg.isSelfMessage():
            print("self message")
            
    def diffuse_message(self,str,sample = False):
    
        if str == 'PreparationPhase':
            for i in  range(self.gateSize('sl')):
                msg = dataMessage('PreparationPhase')
                msg.user_ratings = np.array(get_user_vector(train,i))
                msg.num_items = self.num_items
                msg.num_users = self.num_users
                msg.id_user = i
                self.send(msg, 'sl$o',i)
           



             
    
