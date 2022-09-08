import numpy as np
from pyopp import cSimpleModule, cMessage, simTime
from keras.optimizers import Adam, SGD
from Dataset import Dataset
from WeightsMessage import WeightsMessage
import utility as util
import random
import sys
from evaluate import evaluate_model
from scipy.spatial.distance import cosine

import multiprocessing as mp



import time

# attack parameters
# attacker_id = 49

topK = 20
dataset_name = "ml-100k" #foursquareNYC   ml-1m_version 
num_items =  1682 # 38333  
dataset = Dataset(dataset_name)
train ,testRatings, testNegatives,validationRatings, validationNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives,dataset.validationRatings, dataset.validationNegatives

testRatings = testRatings[:1000] #  2453 1000
testNegatives= testNegatives[:1000]

number_peers = 3
genre = 1 # action


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
        # action movies
        if infos[it][genre] == "1":
            vector.append(it)

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
            arr = line.split("|")

            temp = arr[-19:]
            infos.append(temp)
            line = info.readline()
    
    dist = [0 for _ in range(19)]
    for item in vector:
        for i in range(len(dist)):
            dist[i] += int(infos[item - 1][i]) 
        
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
        self.rounds = 199
        self.current_round = 0
        self.vector = np.empty(0)
        self.labels = np.empty(0)
        self.item_input = np.empty(0)
        self.age = 1
        self.alpha = 0.4
        self.num_items = num_items #train.shape[1] #1682 #3900  TO DO automate this, doesn't work since the validation data has been added because one item is present there and not in training
        self.num_users = train.shape[0] #100 
        self.id_user = self.getIndex()  
        self.period =  0 #np.random.exponential(1)

        # used to create fake profile here
        self.vector = get_user_vector(train,self.id_user)
        self.testRatings, self.testNegatives = get_user_test_set(testRatings,testNegatives,self.id_user)
        self.validationRatings, self.validationNegatives = get_user_test_set(validationRatings,validationNegatives,self.id_user)
        
        self.positives_nums = len(self.vector)

        self.item_input, self.labels, self.user_input = self.my_dataset()
        self.model = util.get_model(self.num_items,self.num_users) # each node initialize its own model 
        self.model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy')
        self.period_message = cMessage('period_message')

        self.update(epochs= 2, batch_size = 32)
        self.peers = []
        self.neighbours = dict()
    
        self.peer_sampling()
        self.performances = {}
        self.scheduleAt(simTime() + self.period,self.period_message)


    def handleMessage(self, msg):
        # periodic self sent message used as a timer by each node to diffuse its model 
        if msg.getName() == 'period_message':
            if self.rounds >= 0 :
                if  self.rounds % 25 == 0:
                    lhr, lndcg = self.evaluate_local_model(False,False)
                    self.diffuse_to_server(lhr, lndcg)
                # elif self.rounds == 1:
                #     self.model.set_weights(self.best_model)
                #     lhr, lndcg = self.evaluate_local_model(False, False)
                #     self.diffuse_to_server(lhr,lndcg)

                self.diffuse_to_peer()
                if self.rounds % 10 == 0:
                    # self.peer_sampling()
                    self.peer_sampling_enhanced()
               

                self.rounds = self.rounds - 1
                self.scheduleAt(simTime() + self.period,self.period_message)
            
            elif self.rounds == 0: # if number of rounds has been acheived, we can evaluate the model both locally
                    hr, ndcg = self.evaluate_local_model(False,False)
                    print('node : ',self.id_user)
                    print('Local HR =  ', hr)
                    print('Local NDCG =  ',ndcg)
                    print("Profiles Found \n")
                    best_profiles = sorted(list(self.neighbours.items()),key=lambda x: x[1]) 
                    # print(best_profiles)
                    sys.stdout.flush()
             
                
             
        elif msg.getName() == 'Model': 
            # dt = self.merge(msg)
            dt = self.DKL_mergeJ(msg)
            # self.id_user == attacker_id 
            self.find_profiles(msg)
                
            ## added the pull possibility
            if msg.type == "push":
                self.diffuse_to_specific_peer(msg.id)
                self.current_round += 1

            
            self.delete(msg)
            

    def finish(self):
        pass
        
    def find_profiles(self, msg):
        items_embeddings = msg.weights[0]
        self_items_embeddings = self.get_model()[0]
        distance = 0

        for i in range(len(self_items_embeddings)):
            if i in self.vector: # added to consider only items in the local set of the user
                distance += abs(cosine(self_items_embeddings[i], items_embeddings[i]))
        
        distance /= len(self.vector)
        if(self.neighbours.get(msg.id) == None):
            self.neighbours[msg.id] = distance
        else:
            self.neighbours[msg.id] = distance  if distance  < self.neighbours[msg.id] else self.neighbours[msg.id]



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
       
    def set_model(self, weights):
        self.model.get_layer("item_embedding").set_weights(weights)
        # self.model.set_weights(weights)

    def get_gate(self, peer):
        idx = self.getIndex()
        if peer < idx:
            return peer
        else:
            return peer - 1 

    def peer_sampling(self):
        size = self.gateSize("no") - 1
        old_peers = self.peers.copy()
        self.peers = []
        for _ in range(number_peers):
            p = random.randint(0,size - 1)
            while(p in self.peers or p in old_peers or p == self.id_user):
                p = random.randint(0,size - 1)
            self.peers.append(p)



    def peer_sampling_enhanced(self):       
        print("****** f le peer sampling enhanced")

        size = self.gateSize("no") - 1 
        self.peers = []
        exploitation_peers = int(number_peers * (1 - self.alpha))
        self.performances = sorted(self.performances.items(), key=lambda x: x[1],reverse=True)  
        keys = [x[0] for x in self.performances]
        i = 0

        while i < exploitation_peers and i < len(keys):
            p = self.get_gate(keys[i])
            self.peers.append(p)
            i += 1
                
        self.performances = {}

        exploration_peers = number_peers - i

        for _ in range(exploration_peers):
            p = random.randint(0,size - 1)
            while(self.get_gate(p) in self.peers):
                p = random.randint(0,size - 1)
            self.peers.append(p)

        print("****** ani kharedj")
        sys.stdout.flush()

    def diffuse_to_specific_peer(self, id):
        weights = WeightsMessage('Model')
        weights.weights = self.get_model()
        weights.age = self.age       
        weights.samples = self.positives_nums 
        weights.type = "pull"
        weights.id = self.getIndex()

        self.send(weights, 'no$o', self.get_gate(id))
    
    # select a random peer and send its model weights and its age to it  
    def diffuse_to_peer(self,nb_peers = 3):
        peers = self.peers.copy()
        for _ in range(nb_peers):
            peer = random.randint(0,len(peers)-1)
            weights = WeightsMessage('Model')
            weights.weights = self.get_model()
            weights.age = self.age       
            weights.samples = self.positives_nums 
            weights.id = self.id_user
            weights.type = "push"
            self.send(weights, 'no$o',self.get_gate(peers[peer]))
            peers.pop(peer)

    def diffuse_to_server(self,hr,ndcg):
        if self.rounds != 1:
            weights = WeightsMessage('Performance')
        else:
            weights = WeightsMessage('FinalPerformance')
          # if self.id_user == attacker_id:

        neighbours = list(self.neighbours.items())
        neighbours.sort(key= lambda x:x[1])
        weights.cluster_found = [x[0] for x in neighbours]
        weights.user_id = self.id_user
        weights.round = self.rounds
        weights.hit_ratio = hr
        weights.ndcg = ndcg
        self.send(weights, 'nl$o',0)
    
    def update(self, epochs = 2, batch_size = None):
        batch_size = len(self.labels) if batch_size == None else batch_size
        hist = self.model.fit([self.user_input, self.item_input], #input
                        np.array(self.labels), # labels 
                        batch_size= batch_size, nb_epoch=epochs, verbose=2, shuffle=True) 
        
        # hr, _ = self.evaluate_local_model()
        # if hr >= self.best_hr:
            # self.best_hr = hr
            # self.best_model = self.model.get_weights().copy()
        
        self.age = self.age + 1
        print("Node : ",self.getIndex())
        print("Rounds Left : ",self.rounds)
        sys.stdout.flush()


        
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
        local = self.get_model().copy()
        hrs = []
        lhr, _ =self.evaluate_local_model()
        hrs.append(lhr)
        self.set_model(message_weights.weights)
        hr, _ = self.evaluate_local_model()
        self.performances[message_weights.id] = hr
        hrs.append(hr)

        hrs_total = sum(hrs)
        
        if(hrs_total) == 0:
            self.set_model(local)
            return 0

        norm = [ (float(i))/hrs_total for i in hrs]
            

        local[:] = [w * norm[0] for w in local]
        message_weights.weights[:] = [w * norm[1] for w in message_weights.weights]
        
        local[:] = [a + b for a,b in zip(local,message_weights.weights)]
        self.set_model(local)
        self.item_input, self.labels, self.user_input = self.my_dataset()
        self.update()
        
        return 0
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