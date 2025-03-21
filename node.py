import numpy as np
from pyopp import cSimpleModule, cMessage, simTime
from keras.optimizers import Adam, SGD
from Dataset import Dataset
from WeightsMessage import WeightsMessage
import utility as util
import random
import sys
from evaluate import evaluate_model
import csv 


topK = 20 # Rank to consider when evaluating recommended items 
dataset_name = "ml-100k" #foursquareNYC   
num_items =  1682 # 38333  
dataset = Dataset(dataset_name)
train ,testRatings, testNegatives, \
validationRatings, validationNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives, \
    dataset.validationRatings, dataset.validationNegatives
    
# testRatings = testRatings[:1000] #   if one wishes to evaluate the models globally & for only 100 users; not a problem for a personalized-data metric measurement
# testNegatives= testNegatives[:1000]

number_peers = 3


def get_user_vector(train,user = 0):
    positive_instances = []    
    for (u,i) in train.keys():
        if u == user:
            positive_instances.append(i)
        if u  > user :
            break

    return positive_instances

def get_user_test_set(testRatings, testNegatives, user):
    personal_testRatings = []
    personal_testNegatives = []
    
    for i in range(len(testRatings)):    
        idx = testRatings[i][0]
        if idx == user:
            personal_testRatings.append(testRatings[i])
            personal_testNegatives.append(testNegatives[i])
        elif idx > user:
            break
        
    return personal_testRatings, personal_testNegatives

def get_individual_set(user, ratings, negatives):
    personal_Ratings = []
    personal_Negatives = []

    for i in range(len(ratings)):
        idx = ratings[i][0]
        if idx == user:
            personal_Ratings.append(ratings[i].copy())
            personal_Negatives.append(negatives[i].copy())
        elif idx > user:
            break

    return personal_Ratings, personal_Negatives


class Node(cSimpleModule):
    def initialize(self):
        # initialization phase in which number of rounds, model age, data is read, model is created, connecter peers list is created
        self.rounds = 600   
        self.vector = np.empty(0)
        self.labels = np.empty(0)
        self.item_input = np.empty(0)
        self.age = 1
        self.alpha = 0.4
        self.num_items = 1682 #train.shape[1] #1682 ml-100k #3900 foursquare; TODO automate this, get number of items from all sets (train, val..) before building the model
        self.num_users = 100 #to consider only 100 users, .ned file needs to be altered too when modying number of users in the system 
        self.id_user = self.getIndex()  
        self.period = 0 # np.random.exponential(0.1) #*** for different periods per node, TODO exploring periodicity impact***

        self.vector = get_user_vector(train,self.id_user) # positive samples; relevant items
        self.testRatings, self.testNegatives = get_user_test_set(testRatings, testNegatives, self.id_user) # final model testing data 
        self.validationRatings, self.validationNegatives = get_user_test_set(validationRatings,validationNegatives,self.id_user) # D_ weighting
        self.best_hr = 0.0
        self.best_ndcg = 0.0
        self.best_model = []

        self.positives_nums = len(self.vector)

        self.item_input, self.labels, self.user_input = self.my_dataset()
        self.model = util.get_model(self.num_items,self.num_users) # *** each node initialize its own model ***
        self.model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy')
        
        self.period_message = cMessage('period_message')
        self.update()
        self.peers =  []
        self.peer_sampling()
        self.performances = {} # *** used for the personnalized peer-sampling protocol *** 
        
        self.scheduleAt(simTime() + self.period,self.period_message)
        

    def handleMessage(self, msg):
        # periodic self sent message used as a timer by each node to diffuse its model 
        if msg.getName() == 'period_message':
            
            if self.rounds > 0 :

                if self.rounds % 10 == 0 or self.rounds == 1:
                    if self.rounds % 10 == 0:
                        lhr, lndcg = self.evaluate_local_model(False,False)   
                    else:
                        if(len(self.best_model) > 0):
                            self.model.set_weights(self.best_model)
                        lhr, lndcg = self.evaluate_local_model(False,False)
                    
                    print('node : ',self.id_user)
                    print('Local HR =  ', lhr)
                    print('Local NDCG =  ',lndcg)
                    print('Round left = ', self.rounds)
                    sys.stdout.flush()
                    self.diffuse_to_server(lhr, lndcg)

            
                # diffuse model periodically, either to one peer or the whole neighborhood 
                self.diffuse_to_peer()
                # self.broadcast()
                
                # peer-sampling call; 
                if self.rounds % 10 == 0:
                    # self.peer_sampling()
                    self.peer_sampling_enhanced()
               
                self.rounds = self.rounds - 1
                self.scheduleAt(simTime() + self.period,self.period_message)
            

                
             
        elif msg.getName() == 'Model':
           
            # *** aggregation function call ***
            # TODO: Automate choice of aggregation function
             
            # self.model_age(msg) 
            # self.FullAvg(msg) # Decentralized FedAvg
            dt = self.Performance_based(msg) # Performance based 
        

            # Evaluation on validation set
            hr, ndcg = self.evaluate_local_model(False, True)
            if hr >= self.best_hr:
                self.best_hr = hr
                self.best_ndcg = ndcg
                self.best_model = self.model.get_weights().copy()


            ## *** the pull possibility to flatten the degree distribution of input for clients; expensive. 
            # if msg.type == "pull":
            #     self.diffuse_to_specific_peer(msg.id)
    

            self.delete(msg)
            

    def finish(self):
        pass
    
    def evaluate_local_model(self,all_dataset = False, validation = True, topK = topK):
        evaluation_threads = 1 #mp.cpu_count()
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
        # if we're sharing only items' embeddings : 
        # return self.model.get_layer("item_embedding").get_weights().copy()
        return self.model.get_weights()
        
    def set_model(self, weights):
        # similarly
        # self.model.get_layer("item_embedding").set_weights(weights)
        self.model.set_weights(weights)

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

    # peer-sample considering the performances of received models in the last period;
    def peer_sampling_enhanced(self):       
        size = self.gateSize("no") - 1 
        self.peers = []
        exploitation_peers = int(number_peers * (1 - self.alpha))
        self.performances = sorted(self.performances.items(), key= lambda x: x[1] ,reverse=True)  
        keys = [x[0] for x in self.performances]
        i = 0

        while i < exploitation_peers and i < len(keys):
            self.peers.append(keys[i])
            i += 1
                
        self.performances = {}

        exploration_peers = number_peers - i

        for _ in range(exploration_peers):
            p = random.randint(0,size - 1)
            while(p in self.peers or p == self.id_user):
                p = random.randint(0,size - 1)
            self.peers.append(p)


    def diffuse_to_specific_peer(self, id):
        weights = WeightsMessage('Model')
        weights.weights = self.get_model()
        weights.age = self.age       
        weights.samples = self.positives_nums 
        weights.type = "push"
        weights.id = self.getIndex()

        self.send(weights, 'no$o', self.get_gate(id))
    
    def broadcast (self):
        for p in self.peers:
            weights = WeightsMessage('Model')
            weights.weights = self.get_model()
            weights.age = self.age       
            weights.samples = self.positives_nums 
            weights.id = self.id_user
            weights.type = "pull"
            self.send(weights, 'no$o',self.get_gate(p))

    # select random peers and send its model weights and its age to it  
    def diffuse_to_peer(self,nb_peers = 3, type = "pull"): 
        peers = self.peers.copy()
        for _ in range(nb_peers):
            peer = random.choice(peers)
            weights = WeightsMessage('Model')
            weights.weights = self.get_model()
            weights.age = self.age       
            weights.samples = self.positives_nums 
            weights.id = self.id_user
            weights.type = type
            self.send(weights, 'no$o',self.get_gate(peer))
            peers.remove(peer)
                
    
    def diffuse_to_server(self,hr,ndcg):
        weights = WeightsMessage('Performance')
        weights.user_id = self.id_user
        weights.round = self.rounds
        weights.hit_ratio = hr # hit ratio 
        weights.ndcg = ndcg # normalized cumulative discounted gain
        self.send(weights, 'nl$o',0)

    
    def update(self, epochs = 2, batch_size = None):
        batch_size = len(self.labels) if batch_size == None else batch_size
        hist = self.model.fit([self.user_input, self.item_input], #input
                        np.array(self.labels), # labels 
                        batch_size= batch_size, nb_epoch=epochs, verbose=0, shuffle=True) 
   
        self.age = self.age + 1
        sys.stdout.flush()
        
        # different aggregation functions :
    def model_age(self,message_weights):
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
        weights = message_weights.weights
        local_weights = self.get_model()
        local_weights [:] = [(self.positives_nums * a + message_weights.samples * b) / (message_weights.samples + self.positives_nums) for a,b in zip(local_weights, weights)]
        normalized_weight = (message_weights.samples / (message_weights.samples + self.positives_nums))
        self.to_csv_file(message_weights.id, "no", self.id_user, normalized_weight, normalized_weight, self.rounds, "FedAvg")
        self.set_model(local_weights)
        self.update()
        self.item_input, self.labels, self.user_input = self.my_dataset()

        return 0

    def to_csv_file(self, sender, isAttacker, receiver, none_normalized_weights, normalized_weights, round, setting):
        with open("list_weights_given.csv","a") as output:
            writer = csv.writer(output, delimiter=",")
            writer.writerow([sender, isAttacker, receiver, none_normalized_weights, normalized_weights, round, setting])


    def Performance_based(self,message_weights): 
        if len(self.validationRatings) < 2: #D_weighting
            self.simple_merge(message_weights.weights)
            self.item_input, self.labels, self.user_input = self.my_dataset()
            self.update()
            return 0
         
        local = self.get_model()
        ndcgs = []
        lhr,lndcg =self.evaluate_local_model()
        ndcgs.append(lhr * lndcg)
        self.set_model(message_weights.weights)
        hr, ndcg = self.evaluate_local_model()
        self.performances[message_weights.id] = hr * ndcg
        ndcgs.append(hr * ndcg)
        none_normalized_weight = hr * ndcg 
        
        ndcg_total = sum(ndcgs)
        
        if(ndcg_total) == 0:
            self.set_model(local)
            return 0

        norm = [ (float(i))/ndcg_total for i in ndcgs]
            

        local[:] = [w * norm[0] for w in local]
        message_weights.weights[:] = [w * norm[1] for w in message_weights.weights]
        
        normalized_weight = norm[1]
        # self.to_csv_file(message_weights.id, "no", self.id_user, none_normalized_weight, normalized_weight, self.rounds, "Pepper")

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

   
    

    # self.vector  contains relevant items for user
    # user 0 : [55,89,22] 
    # at each update
    # we generate a set of negative items :  [55,(41,9,1,3),..]
