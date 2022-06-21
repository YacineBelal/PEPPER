from collections import defaultdict



from sklearn import neighbors

from pyopp import cSimpleModule, cMessage, EV, simTime
import numpy as np
import random
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from Dataset import Dataset
from WeightsMessage import WeightsMessage
import utility as util
import random
import math
import sys
from evaluate import evaluate_model, User_Popularity_Deviation
from scipy.spatial.distance import cosine

import multiprocessing as mp
from numpy import linalg as LA
from sklearn.metrics  import mean_squared_error
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
genre = 1 # action


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


def get_genreattacked_prop(vector):
    infos = []
    with open("u.item",'r', encoding="ISO-8859-1") as info:
        line = info.readline()
        while(line and line!=''):
            arr = line.split("|")
            temp = arr[-19:]
            infos.append(temp)
            line = info.readline()
    prop = 0
    for item in vector:
        if infos[item][genre] == "1":
            prop += 1
    return prop /  len(vector)

def create_profile(idx):
    infos = []
    vector = []
    with open("u.item",'r', encoding="ISO-8859-1") as info:
        line = info.readline()
        while(line and line!=''):
            arr = line.split("|")
            temp = arr[-19:]
            infos.append(temp)
            line = info.readline()
    
    for it in range(num_items):
        if infos[it][genre] == "1":
            vector.append(it)
             # action

    n = len(vector)
    u = [idx] * n
    testRatings = [x for x in  zip(u,vector[:int(0.1 * n)])]
    validationRatings = [x for x in  zip(u,vector[int(0.1 * n):int(0.2 * n)])]
    testNegatives = []
    validationNegatives = []
    for _ in testRatings:
        testNegatives.append(list(np.random.randint(0,num_items, size = 99)))
    for _ in validationRatings:
        validationNegatives.append(list(np.random.randint(0,num_items, size = 99)))
    vector =  vector[int(0.2 *n):]


    return vector, testRatings, testNegatives, validationRatings, validationNegatives
    

def get_distribution_by_genre(vector):
    infos = []
    with open("u.item",'r', encoding="ISO-8859-1") as info:
        line = info.readline()
        while(line and line!=''):
            arr = line.split("|")    embeddings = normalize(embeddings, axis = 1, norm="l2")

            temp = arr[-19:]
            infos.append(temp)
            line = info.readline()
    
    dist = [0 for _ in range(19)]
    for item in vector:
        for i in range(len(dist)):
            dist[i] += int(infos[item][i]) 
        
    summ = sum(dist)
    dist = [elem / summ for elem in dist]
    
    return dist

def get_global_distribution_by_genre():
    infos = []
    with open("u.item",'r', encoding="ISO-8859-1") as info:
        line = info.readline()
        while(line and line!=''):
            arr = line.split("|")
            temp = arr[-19:]
            infos.append(temp)
            line = info.readline()
    with open("ml-100k.train.rating") as base:
        dist = [0 for _ in range(19)]
        line = base.readline()
        while (line and line !=''):
            arr = line.split("\t")
            item = int(arr[1])  
            for i in range(len(dist)):
                dist[i] += int(infos[item][i]) 
            line = base.readline()
    summ = sum(dist)
    dist = [elem / summ for elem in dist]
    return dist


def jaccard_similarity(list1, list2):
        s1 = set(list1)
        s2 = set(list2)
        return float(len(s1.intersection(s2)) / len(s1.union(s2)))




class Node(cSimpleModule):
    def initialize(self):
        # initialization phase in which number of rounds, model age, data is read, model is created, connecter peers list is created
        self.rounds = 600
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

        # if self.id_user == 100:
        #     self.vector, self.testRatings, self.testNegatives, self.validationRatings, self.validationNegatives = create_profile(self.id_user)
        #     print(len(self.vector))
        #     print(self.vector)
        #     sys.stdout.flush()
        # else:
        self.vector = get_user_vector(train,self.id_user)
        self.testRatings, self.testNegatives = get_user_test_set(testRatings,testNegatives,self.id_user)
        self.validationRatings, self.validationNegatives = get_user_test_set(validationRatings,validationNegatives,self.id_user)
        
        # self.local_dist = get_distribution_by_genre(vector=self.vector)
        # self.global_dist = get_global_distribution_by_genre()
        # self.global_vector = [1 for _ in range(self.num_items)]
        self.local_vector = [ 0 for _ in range(self.num_items)]
        for x in self.vector:
            self.local_vector[x] = 1

        self.time_update = 0
        self.num_updates = 0 
        self.aggregation_time = 0
        self.transfer = 0
        self.peer_sampling_time = 0

        self.positives_nums = len(self.vector)
        self.received = 0

        self.item_input, self.labels, self.user_input = self.my_dataset()
        self.model = util.get_model(self.num_items,self.num_users) # each node initialize its own model 
        self.model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy')
        self.period_message = cMessage('period_message')
        self.best_hr = 0.0
        self.best_ndcg = 0.0
        self.best_model = []

        self.init_rounds = 500
        # self.training_rounds = self.init_rounds
        self.update()
        self.peers = []
        self.neighbours = dict()

        # for i in range(self.gateSize("no")):
        #     if self.gate("no$o",i).isConnected():
        #         self.peers.append(i)
    
        self.peer_sampling()
        self.performances = {}
        self.average = 0
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

                # start_time = time.process_time()
                self.diffuse_to_peer()
                # self.transfer += 1
                # delta = time.process_time() - start_time
                if self.rounds % 10 == 0:
                    # start_time = time.process_time()
                    # self.peer_sampling()
                    self.peer_sampling_enhanced()
                    # delta = time.process_time() - start_time
                    # self.peer_sampling_time += delta
               

                self.rounds = self.rounds - 1
                self.scheduleAt(simTime() + self.period,self.period_message)
            
            elif self.rounds == 0: # if number of rounds has been acheived, we can evaluate the model both locally and globally
                # if self.id_user != 99:
                hr, ndcg = self.evaluate_local_model(False,False)
                print('node : ',self.id_user)
                print('Local HR =  ', hr)
                print('Local NDCG =  ',ndcg)
                sys.stdout.flush()

                # print("My Correspondance = ",get_genreattacked_prop(self.vector))
            
                # else:
                #     print("Profiles Found \n")
                #     best_profiles = sorted(list(self.neighbours.items()),key=lambda x: x[1]) 
                #     print(best_profiles)

                
             
        # messsage containing a neighbour's model is received here    
        elif msg.getName() == 'Model': 
            # if(self.training_rounds > 0):
            # self.received += 1
            # aggregating the weights received with the local weigths (ponderated by the model's age aka number of updates) before making some gradient steps 
            # start_time = time.process_time()
            # dt = self.merge(msg)
            dt = self.DKL_mergeJ(msg)
            # dt = self.FullAvg(msg)
            # delta = time.process_time() - start_time - dt
            # self.aggregation_time += delta

            # lhr, lndcg = self.evaluate_local_model(False,True)
            # if lhr >= self.best_hr:
            #     self.best_hr = lhr
            #     self.best_ndcg = lndcg
            #     self.best_model = self.model.get_weights().copy()
            
            # if self.id_user == 99:
                #     self.find_profiles(msg)
                
            self.delete(msg)
            

    def finish(self):
        pass
        # self.meta_update()
        # util.save_whole_cost(self.transfer,"total_transfer_nb"+str(self.id_user))
        # util.save_whole_cost(self.peer_sampling_time,"total_peersampling_time"+str(self.id_user))
        # util.save_whole_cost(self.aggregation_time,"total_aggregation_time"+str(self.id_user))
        # util.save_whole_cost(self.aggregation_time / self.received,"average_aggregation_time"+str(self.id_user))
        # util.save_whole_cost(self.time_update,"total_update_time"+str(self.id_user))
        # util.save_whole_cost(self.time_update / self.received, "average_update_time"+str(self.id_user))
        
        
    def find_profiles(self, msg):

        items_embeddings = msg.weights[0]
        self_items_embeddings = self.get_model()[0]
        distance = 0
        for i in range(len(self_items_embeddings)):
            if i in self.vector: # added to consider only items in the local set of the user
                distance += abs(cosine(self_items_embeddings[i],items_embeddings[i]))
        
        distance /= len(self.vector)
        if(self.neighbours.get(msg.id) == None):
            self.neighbours[msg.id] = distance
        else:
            self.neighbours[msg.id] = distance  if distance  < self.neighbours[msg.id] else self.neighbours[msg.id]



    # evaluation method : can be on the whole dataset for a general hit ratio but is usually on the local dataset.
    # locally, it can be either on validation data (during the aggregation) or on test data at the end of training
    def evaluate_local_model(self,all_dataset = False, validation=True, topK = topK):
        evaluation_threads = mp.cpu_count()
        if not all_dataset:
            if validation :
                (hits, ndcgs) = evaluate_model(self.model, self.validationRatings, self.validationNegatives, topK, evaluation_threads)               
            else:
                (hits, ndcgs) = evaluate_model(self.model, self.testRatings, self.testNegatives, topK, evaluation_threads)
                # print("HITS ; TEST ITEMS")
                # for i in range(len(hits)):
                #     print(hits[i]," ; ",self.testRatings[i])
                # sys.stdout.flush()
        else:
            (hits, ndcgs) = evaluate_model(self.model, testRatings, testNegatives, topK, evaluation_threads)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        return hr, ndcg
    
    def evaluate_local_model_upd(self):
        evaluation_threads = 2 #mp.cpu_count()
        evaluate_model(self.model, self.testRatings, self.testNegatives, topK, evaluation_threads)
        self.trainprop = self.get_user_train_distribution_vector()
        upd = User_Popularity_Deviation(self.id_user, self.trainprop, self.H, self.M, self.T, self.num_items)   
        return upd


    def get_model(self):
        return self.model.get_layer("item_embedding").get_weights().copy()
        # return self.model.get_weights().copy()
    
    def get_DP_model(self):
        return self.dp_model 

    def add_noise(self):
        sensitivity = 2
        epsilon = 0.1
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
            weights.local_vector = self.local_vector
            weights.samples = self.positives_nums 
            weights.id = self.getIndex()
            self.send(weights, 'no$o',peers[peer])
            peers.pop(peer)

    def diffuse_to_server(self,hr,ndcg):
        if self.rounds != 1:
            weights = WeightsMessage('Performance')
        else:
            weights = WeightsMessage('FinalPerformance')
            weights.mse_peformances = self.mse_ponderations
            weights.accuracy_rank = self.accuracy_rank_ponderations
        weights.user_id = self.id_user
        weights.round = self.rounds
        weights.hit_ratio = hr
        weights.ndcg = ndcg
        self.send(weights, 'nl$o',0)
    
    # making 4 gradient steps
    def update(self):
        hist = self.model.fit([self.user_input, self.item_input], #input
                        np.array(self.labels), # labels 
                        batch_size=len(self.labels), nb_epoch=epochs, verbose=2, shuffle=True) 
        
        # want to keep best model according to validation set (used also for weighting)
        hr, _ = self.evaluate_local_model()
        if hr >= self.best_hr:
            self.best_hr = hr
            self.best_model = self.model.get_weights().copy()
        
        self.age = self.age + 1
        print("Node : ",self.getIndex())
        print("Rounds Left : ",self.rounds)
        # self.training_rounds -= 1
        sys.stdout.flush()

        self.add_noise()

        
    def meta_update(self):
        start_time = time.process_time()
        meta_lr0 = 0.1
        meta_epochs = 120 
        before_weights = self.model.get_weights()
        
        self.meta_model = util.get_model(self.num_items,self.num_users) 
        self.meta_model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy')
        self.meta_model.set_weights(self.model.get_weights())
        
        # outer update
        for ep in range(meta_epochs) :
            hist = self.meta_model.fit([self.user_input, self.item_input], #input
                        np.array(self.labels), # labels 
                        batch_size=len(self.labels), nb_epoch=1, verbose = 0, shuffle=True)
            after_weights = self.meta_model.get_weights()
            # meta-step 
            delta = []
            for i in range(len(after_weights)):
                delta.append(np.subtract(after_weights[i],before_weights[i]))
        
            meta_lr = meta_lr0 * (1 - ep / meta_epochs)
            for i in range(len(before_weights)):             
                before_weights[i] = before_weights[i] + meta_lr * delta[i]
            self.meta_model.set_weights(before_weights)

        end_time = time.process_time() - start_time
        util.save_per_round_cost(end_time, "total_metaupdate_time"+str(self.id_user))
        
        self.model.set_weights(self.meta_model.get_weights())
        lhr, lndcg = self.evaluate_local_model(False,False)
        print("Node :",self.id_user)
        print("Local META HR20: ",lhr)
        print("Local META NDCG20 :",lndcg)
        self.diffuse_to_server(lhr,lndcg)
        
        lhr, lndcg = self.evaluate_local_model(False, False, 10)
        print("Node :",self.id_user)
        print("Local META HR10: ",lhr)
        print("Local META NDCG10 :",lndcg)
        lhr, lndcg = self.evaluate_local_model(False, False, 5)
        print("Node :",self.id_user)
        print("Local META HR5: ",lhr)
        print("Local META NDCG5 :",lndcg)
        


        sys.stdout.flush()

    def merge(self,message_weights):
        weights = message_weights.weights
        local_weights = self.get_model()
        local_weights[:] =  [ (a * self.age + b * message_weights.age) / (self.age + message_weights.age) for a,b in zip(local_weights,weights)]
        self.age = max(self.age,message_weights.age)
        self.set_model(local_weights)

        # start_time = time.process_time()
        self.update()
        # delta = time.process_time() - start_time
        # self.time_update += delta
        self.num_updates += 1
        self.item_input, self.labels, self.user_input = self.my_dataset()

        # util.save_whole_cost(delta, "Local_Update_Time"+str(self.id_user))
        # util.save_whole_cost(delta, "Local_Update_Time_Total")
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
        self.num_updates += 1
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

        # start_time = time.process_time()
        self.update()
        # delta = time.process_time() - start_time
        # self.time_update += delta
        self.num_updates += 1
        
        return 0
        
    
    def Meta_Learning_Approach(self,message_weights):
        if(self.rounds  > 650):
            self.merge(message_weights)
        
        local = self.model.get_weights().copy()
        
        if len(self.validationRatings) < 2:
            self.simple_merge(message_weights.weights,local)
            return

        
        hrs = []
    
        #evaluate local model hit ratio and append it to hrs
        hr, _ = self.evaluate_local_model()
   
        hrs.append(math.exp(hr))
        

     
        self.model.set_weights(message_weights.weights)
        hr, _ = self.evaluate_local_model()
        self.performances[message_weights.id] = hr
        hrs.append(math.exp(hr))
                

        hrs_total = sum(hrs)
    
        norm = [ (float(i))/hrs_total for i in hrs]
             

        local[:] = [w * norm[0] for w in local]
        message_weights.weights[:] = [w * norm[1] for w in message_weights.weights]
      
        
        # average weights
        old_local = local.copy()

        local[:] = [a + b for a,b in zip(local,message_weights.weights)]



        for l in range(len(local)-1):
            for k in range(len(local[l])):
                for i in range(len(local[l][k])):
                    old_local[0][k][i] +=  0.1 * (local[0][k][i] - old_local [0][k][i]) 

        old_local[3] +=  0.1 * (local[3] - old_local[3]) 


        self.model.set_weights(old_local)
        hrs = []
   
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

    
    def get_user_train_distribution_vector(self):
        h = m = t = 0   
        for i in self.vector:
            if  str(i) in self.H:
                h += 1
            elif str(i) in self.M:
                m += 1
            else:
                t += 1
            
        train_prop = [h,m,t]
        train_prop = [item / sum(train_prop) for item in train_prop]
        return train_prop