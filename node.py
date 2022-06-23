from pyopp import cSimpleModule, cMessage, EV, simTime
import numpy as np
import random
from keras.optimizers import Adam, SGD
from Dataset import Dataset
from WeightsMessage import WeightsMessage
import utility as util
from evaluate import evaluate_model
import random
import sys

from evaluate import evaluate_model
import multiprocessing as mp
from numpy import linalg as LA
from sklearn.preprocessing import normalize


import time



topK = 20
dataset_name = "ml-100k" #foursquareNYC   ml-1m_version 
num_items =  1682 # 38333  
dataset = Dataset(dataset_name)
train ,testRatings, testNegatives,validationRatings, validationNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives,dataset.validationRatings, dataset.validationNegatives

testRatings = testRatings[:1000] #  2453 1000
testNegatives= testNegatives[:1000]

epochs = 2
number_peers = 3
# batch_size = 32s


def get_items_per_class_file():
    with open("H_"+dataset_name+".txt","r") as input, open("M_"+dataset_name+".txt","r") as input1, open("T_"+dataset_name+".txt","r") as input2:
        H = input.readlines()
        H = [ i.strip() for i in H]
        M = input1.readlines()
        M = [ i.strip() for i in M]
        T = input2.readlines()
        T = [ i.strip() for i in T]

    
    return H, M, T

def get_user_vector(train,user = 0):
    positive_instances = []
    # nb_user = 0
    # last_u = list(train.keys())[0]
    
    for (u,i) in train.keys():
        # if(u != last_u):
        #     nb_user +=1
        #     last_u = u
        if u == user:
            positive_instances.append(i)
        if u  > user :
            break

    return positive_instances

def get_user_test_set(testRatings,testNegatives,user):
    personal_testRatings = []
    personal_testNegatives = []
    # nb_user = 0
    # last_u = testRatings[0][0]
    for i in range(len(testRatings)):
        
        # if(testRatings[i][0] != last_u):
        #     nb_user +=1
        #     last_u = testRatings[i][0]
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
        self.rounds = 180
        self.init_rounds = 0

        self.mse_ponderations = 0
        self.nb_mse = 0
        self.accuracy_rank_ponderations = 0
        self.nb_accuracy_rank = 0
        self.vector = np.empty(0)
        self.labels = np.empty(0)
        self.item_input = np.empty(0)
        self.age = 1
        self.alpha = 0.4
        self.num_items = num_items #train.shape[1] #1682 #3900  TO DO automate this, doesn't work since the validation data has been added because one item is present there and not in training
        self.num_users = train.shape[0] #100 
        self.id_user = self.getIndex()  
        self.period =  0 #np.random.exponential(1)
        self.vector = get_user_vector(train,self.id_user)
        self.testRatings, self.testNegatives = get_user_test_set(testRatings,testNegatives,self.id_user)
        self.validationRatings, self.validationNegatives = get_user_test_set(validationRatings,validationNegatives,self.id_user)
        
        self.local_vector = [ 0 for _ in range(self.num_items)]
        for x in self.vector:
            self.local_vector[x] = 1

        self.positives_nums = len(self.vector)

        self.item_input, self.labels, self.user_input = self.my_dataset()
        self.model = util.get_model(self.num_items,self.num_users) # each node initialize its own model 
        self.model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy')
        self.period_message = cMessage('period_message')
        self.best_hr = 0.0
        self.best_ndcg = 0.0
        self.best_model = []

        self.update()
        self.peers = []
        self.neighbours = dict()

        # for i in range(self.gateSize("no")):
        #     if self.gate("no$o",i).isConnected():
        #         self.peers.append(i)
    
        self.peer_sampling()
        self.performances = {}
        self.scheduleAt(simTime() + self.period,self.period_message)


    def handleMessage(self, msg):
        # periodic self sent message used as a timer by each node to diffuse its model 
        if msg.getName() == 'period_message':
            if self.rounds > 0 :
                if self.rounds != 1:
                    lhr, lndcg = self.evaluate_local_model(False,False)
                    self.diffuse_to_server(lhr, lndcg)
                else:
                    self.model.set_weights(self.best_model)
                    lhr, lndcg = self.evaluate_local_model(False, False)
                    self.diffuse_to_server(lhr,lndcg)

                self.diffuse_to_peer()
                if self.rounds % 10 == 0:
                    # self.peer_sampling()
                    self.peer_sampling_enhanced()
               

                self.rounds = self.rounds - 1
                self.scheduleAt(simTime() + self.period,self.period_message)
            
            elif self.rounds == 0: # if number of rounds has been acheived, we can evaluate the model both locally and globally
                hr, ndcg = self.evaluate_local_model(False,False)
                print('node : ',self.id_user)
                print('Local HR =  ', hr)
                print('Local NDCG =  ',ndcg)
                sys.stdout.flush()

                
             
        # messsage containing a neighbour's model is received here    
        elif msg.getName() == 'Model': 
            # if(self.training_rounds > 0):
            # dt = self.merge(msg)
            dt = self.DKL_mergeJ(msg)
            self.init_rounds +=1
            # dt = self.FullAvg(msg)
                
            self.delete(msg)
            

    def finish(self):
        pass
        
    def evaluate_local_model(self,all_dataset = False, validation=True, topK = topK):
        evaluation_threads = mp.cpu_count()
        if not all_dataset:
            if validation :
                (hits, ndcgs) = evaluate_model(self.model, self.validationRatings, self.validationNegatives, topK, evaluation_threads)               
            else:
                (hits, ndcgs) = evaluate_model(self.model, self.testRatings, self.testNegatives, topK, evaluation_threads)
        else:
            (hits, ndcgs) = evaluate_model(self.model, testRatings, testNegatives, topK, evaluation_threads)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        return hr, ndcg
    
    def get_model(self):
        return self.model.get_layer("item_embedding").get_weights().copy()
        # return self.model.get_weights().copy()
    
    def get_DP_model(self):
        return self.dp_model 

    def add_noise(self):
        sensitivity = 2
        epsilon = 1
        delta = 10e-5  
        sigma =  sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        self.dp_model = self.get_model()
        norm = LA.norm(self.dp_model)
        # self.dp_model = normalize(self.dp_model, axis = 1, norm="l2")
        self.dp_model = np.divide(self.dp_model, norm)     
        self.dp_model = np.add(self.dp_model,np.random.normal(loc = 0, scale = sigma, size = self.dp_model.shape))

        return self.dp_model
    
    
    def set_model(self, weights):
        self.model.get_layer("item_embedding").set_weights(weights)
        # self.model.set_weights(weights)



    def peer_sampling(self):
        size = self.gateSize("no") - 1
        self.peers = []
        for _ in range(number_peers):
            p = random.randint(0,size - 1)
            while(p in self.peers):
                p = random.randint(0,size - 1)
            self.peers.append(p)


    def get_gate(self, peer):
        idx = self.getIndex()
        if peer < idx:
            return peer
        else:
            return peer - 1 

    def peer_sampling_enhanced(self): #alpha is the exploration/exploitation ratio where alpha = 0 exclusively 
        # keeps the same actual peers and alpha = 1 change all the peers 
      
        size = self.gateSize("no") - 1 
        self.peers = []
        exploitation_peers = int(number_peers * (1 - self.alpha))
        self.performances = sorted(self.performances.items(), key=lambda x: x[1],reverse=True)  
        keys = [x[0] for x in self.performances]
        i = 0

        while i < exploitation_peers and i < len(keys):
            p = keys[i]
            p = self.get_gate(p)
            self.peers.append(p)
            i += 1
                
        self.performances = {}
       
        exploration_peers = number_peers - i

        for _ in range(exploration_peers):
            p = random.randint(0,size - 1)
            while(p in self.peers):
                p = random.randint(0,size - 1)
            self.peers.append(p)
        
        sys.stdout.flush()

    # select a random peer and send its model weights and its age to it  
    def diffuse_to_peer(self,nb_peers = 3):
        peers = self.peers.copy()
        for _ in range(nb_peers):
            peer = random.randint(0,len(peers)-1)
            weights = WeightsMessage('Model')
            weights.weights = self.get_model()
            weights.dp_weights = self.get_DP_model()
            weights.age = self.age       
            # weights.dist = self.local_dist
            weights.samples = self.positives_nums 
            weights.id = self.getIndex()
            self.send(weights, 'no$o',peers[peer])
            peers.pop(peer)

    def diffuse_to_server(self,hr,ndcg):
        if self.rounds != 1:
            weights = WeightsMessage('Performance')
        else:
            weights = WeightsMessage('FinalPerformance')
            weights.mse_performances = self.mse_ponderations
            weights.accuracy_rank = self.accuracy_rank_ponderations
        weights.user_id = self.id_user
        weights.round = self.rounds
        weights.hit_ratio = hr
        weights.ndcg = ndcg
        self.send(weights, 'nl$o',0)
    
    def update(self):
        hist = self.model.fit([self.user_input, self.item_input], #input
                        np.array(self.labels), # labels 
                        batch_size=len(self.labels), nb_epoch=epochs, verbose=2, shuffle=True) 
        
        hr, _ = self.evaluate_local_model()
        if hr >= self.best_hr:
            self.best_hr = hr
            self.best_model = self.model.get_weights().copy()
        
        self.age = self.age + 1
        print("Node : ",self.getIndex())
        print("Rounds Left : ",self.rounds)
        sys.stdout.flush()

        self.add_noise()

    def merge(self,message_weights):
        weights = message_weights.weights
        local_weights = self.get_model()
        local_weights[:] =  [ (a * self.age + b * message_weights.age) / (self.age + message_weights.age) for a,b in zip(local_weights,weights)]
        self.age = max(self.age,message_weights.age)
        self.set_model(local_weights)

        self.update()
        self.item_input, self.labels, self.user_input = self.my_dataset()

        return 0

        
    def simple_merge(self,weights):
        local = self.get_model()
        local[:] =  [ (a + b) / 2 for a,b in zip(local,weights)]
        self.set_model(local)


    def FullAvg(self, message_weights):
        local_weights = self.get_model()
        local_weights [:] = [(self.positives_nums * a + message_weights.samples * b) / (message_weights.samples + self.positives_nums) for a,b in zip(local_weights,message_weights.weights)]
        self.set_model(local_weights)
        start_time = time.process_time()
        self.update()
        delta = time.process_time() - start_time
        self.time_update += delta
        self.item_input, self.labels, self.user_input = self.my_dataset()

        return delta

    def DKL_mergeJ(self,message_weights):  
        local = self.get_model()
        hrs = []
        lhr, _ =self.evaluate_local_model()
        hrs.append(lhr)
        self.set_model(message_weights.dp_weights)
        dp_hr, _ = self.evaluate_local_model()
        self.performances[message_weights.id] = dp_hr
        hrs.append(dp_hr)

        self.set_model(message_weights.weights)
        hr, _ = self.evaluate_local_model()
        
        self.nb_mse += 1        
        self.mse_ponderations +=  (dp_hr - hr) ** 2 / self.nb_mse 
        
        # mse or something to compare weights with and without DP

        hrs_total = sum(hrs)
        
        if(hrs_total) == 0:
            self.set_model(local)
            return 0

        #normalize hit ratios  
        norm = [ (float(i))/hrs_total for i in hrs]
            

        local[:] = [w * norm[0] for w in local]
        message_weights.weights[:] = [w * norm[1] for w in message_weights.weights]
        
        self.nb_accuracy_rank += 1
        if (hr > lhr and dp_hr > lhr) or (hr < lhr and dp_hr < lhr) or (hr == dp_hr == lhr):
            self.accuracy_rank_ponderations += 1 / self.nb_accuracy_rank 


        # average weights
        local[:] = [a + b for a,b in zip(local,message_weights.weights)]
        self.set_model(local)
        self.item_input, self.labels, self.user_input = self.my_dataset()

        self.update()
        
        return 0
        
    # using the data at disposition (which is the positives ratings) we create the negative ratings that goes with, in order to have a small local dataset
    def my_dataset(self,num_negatives = 4):
        item_input = []
        labels = []
        user_input = []
        for i in self.vector:
            item_input.append(i)
            labels.append(1)
            user_input.append(self.id_user)
            for i in range(num_negatives):
                j = np.random.randint(self.num_items)
                while j in self.vector:
                    j = np.random.randint(self.num_items)
                user_input.append(self.id_user)
                item_input.append(j)
                labels.append(0)            
        return np.array(item_input), np.array(labels), np.array(user_input)

    