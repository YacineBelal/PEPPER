from cProfile import label
import enum
from re import X
from pyopp import cSimpleModule
import sys 
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from collections import defaultdict
# import matplotlib.pyplot as plt 
import Dataset
import numpy as np
import wandb
import os 

sync_ = 1
name_ = "Performance-Based TensorFlow"
dataset_ = "ml-100k" #foursquareNYC   
topK = 20
clustersK= 7
attacker_id = 49


dataset = Dataset("ml-100k")

def get_user_vector(user):
    positive_instances = []
    
    for (u,i) in dataset.trainMatrix.keys():
        if u == user:
            positive_instances.append(i)
        if u  > user :
            break

    return positive_instances
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
            dist[i] += int(infos[item][i]) 
        
    summ = sum(dist)
    dist = [elem / summ for elem in dist]
    
    return dist

def indic(data):
    max = np.max(data, axis=1)
    std = np.std(data, axis=1)
    return max, std


def cdf(data, metric):
    data_size=len(data)
     
    # Set bins edges
    data_set=sorted(set(data))
    bins=np.append(data_set, data_set[-1]+1)

    
    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(data, bins=bins, density=False)

   
    counts=counts.astype(float)/data_size   
    
    # Find the cdf
    cdf = np.cumsum(counts)    
    idx = np.arange(cdf.shape[0])
    data = [[x, y] for (x, y) in zip(bin_edges[idx], cdf[idx])]
    table = wandb.Table(data=data, columns = [metric+"@"+str(topK), "CDF"])
    wandb.log({metric+"@"+"CDF" : wandb.plot.line(table, metric+"@"+str(topK), "CDF", stroke="dash",
           title = metric+"@"+str(topK)+" cumulative distribution")})


if  sync_:
    wandb_config = {
                "config": "B",
                "dataset": dataset_,
                "implementation": "TensorFlow",
                "rounds": 400,
                "learning_rate": 0.01,
                "epochs": 2,
                "batch_size": "Full",
                "topK": topK,
                "Epsilon": "0.1",
                "Delta": 10e-5,
                "Number of clusters": clustersK,
                "Attacker id": attacker_id
                }

    os.environ["WANDB_API_KEY"] = "334fd1cd4a03c95f4655357b92cdba2b7d706d4c"
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_START_METHOD"] = "thread"
    wandb.init(project="DecentralizedGL", entity="drimfederatedlearning", name = name_, config = wandb_config)
   

class Server(cSimpleModule):
    
    def initialize(self):
        self.all_participants = [i for i in range(self.gateSize('sl'))]
        self.num_participants = len(self.all_participants)
        self.hit_ratios = defaultdict(list)
        self.ndcgs = defaultdict(list)
        self.accuracy_rank = defaultdict(list)
        self.mse_performances = defaultdict(list)
        self.cluster_found = []
    
    def handleMessage(self, msg):
        if msg.getName() == "Performance":            
            self.hit_ratios[msg.round].append(msg.hit_ratio) # hit ratio ~ recall for recsys 
            self.ndcgs[msg.round].append(msg.ndcg) # ~ accuracy
        elif msg.getName() == "FinalPerformance":
            self.hit_ratios[msg.round].append(msg.hit_ratio) # hit ratio ~ recall for recsys 
            self.ndcgs[msg.round].append(msg.ndcg) # ~ accuracy
            self.accuracy_rank[msg.round].append(msg.accuracy_rank)
            self.mse_performances[msg.round].append(msg.mse_performances)     
            if msg.cluster_found != None:
                self.cluster_found = msg.cluster_found  

        self.delete(msg)
            
    
    def finish(self):
        global wandb
        nb_rounds = max(self.hit_ratios.keys())
        for round in self.hit_ratios.keys():
            if(len(self.hit_ratios[round]) == self.num_participants): # if we receive all the users' model we evaluate them and then compute an average perf
                print("round = ", round)
                avg_hr = sum(self.hit_ratios[round]) / self.num_participants
                print("Average Test HR = ",avg_hr)
                avg_ndcg = sum(self.ndcgs[round]) / self.num_participants
                print("Average Test NDCG = ",avg_ndcg)
                sys.stdout.flush()
                avg_mse = sum(self.mse_performances[round]) / self.num_participants
                print("Average MSE on weights = ",avg_mse)
                avg_acc = sum(self.accuracy_rank[round]) / self.num_participants
                print("Average Accuracy of model ranking = ",avg_acc)
                sys.stdout.flush()
                if sync_:
                    wandb.log({"Average HR": avg_hr,"Average NDCG": avg_ndcg, "Round ": nb_rounds - round})

        acc = self.groundTruth_Clustering()
        if sync_:
            wandb.log({"Average MSE (weights delta)": avg_mse,
                    "Average Accuracy (of model ranking)": avg_acc, "Attack Accuracy": acc})
            cdf(self.hit_ratios[1],"Local HR")      
            cdf(self.ndcgs[1],"Local NDCG")
            cdf(self.mse_performances[1],"MSE per node (weight difference)")
            cdf(self.accuracy_rank[1],"Ranking accuracy per node (Normalized weight ranking)")
            wandb.finish()

    
    def groundTruth_Clustering(self):
        users = []
        for u in range(self.all_participants):
            vector = get_user_vector(u)
            users.append(get_distribution_by_genre(vector))

        users = np.array(users)

        ### kelbow_visualizer(KMeans(random_state = 0),users,k=(2,30))

        model = KMeans(n_clusters=clustersK,random_state = 0).fit(users) 
        _labels = list(model.labels_)
        centroids = model.cluster_centers_
        users_distances = []
        for i,u in enumerate(users):
            users_distances.append(euclidean(centroids[_labels[i]],u))

        ## putting outliers in a cluster
        
        # threshold_users_keep = 0.9
        # threshold_max_distance = (sorted(users_distances, reverse=True)[int((1-threshold_users_keep) * len(users_distances)):])[0]
        # outliers =[x for x in range(len(users)) if users_distances[x] > threshold_max_distance]
        # for x in outliers:x
        #     model.labels_[x] = clustersK


        cluster_knn = []
        for u,c in enumerate(_labels):
            if c == _labels[attacker_id] and u!= attacker_id:
                cluster_knn.append(u)
        
        print("True cluster :",cluster_knn)
        print("Cluster found:", self.cluster_found)
        intersection = set(cluster_knn) & set(self.cluster_found)
        print("Intersection: ", intersection)
        acc = len(intersection)/ len(cluster_knn)
        print("Accuracy :", acc)

        return acc


        


        
        






