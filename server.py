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

dataset = Dataset('small')
train = dataset.trainMatrix


def get_user_vector(train,user = 0):
    positive_instances = []
    for (u,i) in train.keys():
        if u == user:
            positive_instances.append(i)
            print(i)
        if u  > user :
            break

    return positive_instances

def get_items_instances(train):
    item_input = []
    for i in range(train.shape[1]):
        item_input.append(i)
    
    return item_input

def init_normal(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)

def get_embeddings(input,factors = 8):
    dim = len(input)
    model = Sequential()
    model.add(Embedding(input_dim=dim,output_dim=factors,name = 'item_embedding',
                                  init = init_normal, W_regularizer = l2(0), input_length=1))
    model.add(Flatten())
    model.compile('rmsprop', 'mse')
    output_array = model.predict(np.array(input))
    return output_array     

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
        self.number_rounds = 5
        self.message_round = cMessage('message_round')
        self.message_averaging = cMessage('StartAveraging')
        self.global_weights = []
        self.total_samples = [] 
        
        item_input = get_items_instances(train)
        embeddings = get_embeddings(item_input)    
        
        model = util.get_model(len(item_input)) 
        self.global_weights.append(model.get_weights())
        if self.getName() == 'server':
            
            for i in range(self.gateSize('sl')):
                msg = dataMessage('PreparationPhase')
                msg.user_ratings = np.array(get_user_vector(train,i))
                msg.items_embeddings = embeddings
                self.send(msg, 'sl$o',i)
            self.scheduleAt(simTime() + 2,self.message_round)
       

    def handleMessage(self, msg):
        if msg.isSelfMessage():
            if msg.getName() == 'StartAveraging':
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
                self.global_weights.append(new_weights)
                if self.number_rounds > 0:
                    self.diffuse_weights() 

               
            else: 
                EV << 'Server : starting a round of training'
                self.diffuse_weights('FirstRound')
                self.global_weights = []   
       
        elif msg.getName() == 'Node_weights':
            # averaging weight , taking account the number of items the user has rated to weight the average
            # an averaging based on fedavg
            self.global_weights.append(msg.weights[0])
            self.total_samples.append(msg.positives_nums)
            
    def diffuse_weights(self,str = 'Round'):
        
        for i in range (self.gateSize('sl')):
            weights = WeightsMessage(str)
            weights.weights = self.global_weights[0]
            self.send(weights,'sl$o',i)
        self.scheduleAt(simTime() + self.gateSize('sl'),self.message_averaging)
        self.number_rounds = self.number_rounds - 1    

             
    
