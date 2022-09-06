from re import X
from pyopp import cSimpleModule
import sys 
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from collections import defaultdict
# import matplotlib.pyplot as plt 
from Dataset import Dataset
import numpy as np
import wandb
import os 

sync_ = 1
name_ = "Performance-Based Attacked"
dataset_ = "ml-100k" #foursquareNYC   
topK = 20
clustersK= 9


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
            dist[i] += int(infos[item - 1][i]) 
        
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
                "Dataset": dataset_,
                "Implementation": "TensorFlow",
                "Rounds": 200,
                "Learning_rate": 0.01,
                "Epochs": 2,
                "Batch_size": "Full",
                "TopK": topK,
                "Epsilon": np.inf,
                "Delta": np.inf,
                "Number of clusters": clustersK,
                "Attacker id": "all",
                "Distance_Metric": "cosine"
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
        self.cluster_found = defaultdict(list)
        self.att_acc = defaultdict(list)
        self.att_recall = defaultdict(list)

    def handleMessage(self, msg):
        self.hit_ratios[msg.round].append(msg.hit_ratio) # hit ratio ~ recall for recsys 
        self.ndcgs[msg.round].append(msg.ndcg) # ~ accuracy
        self.cluster_found[msg.user_id].append(msg.cluster_found[:clustersK])
        
        # if msg.getName() == "Performance":            
        # elif msg.getName() == "FinalPerformance":

        self.delete(msg)
            
    
    def finish(self):
        global wandb
        nb_rounds = max(self.hit_ratios.keys())
        for round in self.hit_ratios.keys():               
            if(len(self.hit_ratios[round]) == self.num_participants):
                print("round = ", round)
                avg_hr = sum(self.hit_ratios[round]) / self.num_participants
                print("Average Test HR = ",avg_hr)
                avg_ndcg = sum(self.ndcgs[round]) / self.num_participants
                print("Average Test NDCG = ",avg_ndcg)
                sys.stdout.flush()

                if sync_:
                    wandb.log({"Average HR": avg_hr,"Average NDCG": avg_ndcg, "Round ": nb_rounds - round})
        
        nb_rounds = max(self.hit_ratios.keys())
        clusters = self.groundTruth_Clustering()
        idx_round = 0
        for round in self.hit_ratios.keys(): 
            avg_acc = 0
            avg_recall = 0
            for attacker in self.cluster_found.keys():
                acc, recall = self.Accuracy_Clustering_Attack(clusters, attacker, idx_round)
                self.att_acc[round].append(acc)
                self.att_recall[round].append(recall)
                avg_acc += acc
                avg_recall += recall
            idx +=1
            avg_acc = avg_acc / self.num_participants
            avg_recall = avg_recall / self.num_participants
             
            if sync_:
                    wandb.log({"Average Accuracy": avg_acc ,"Average Recall": avg_recall,
                     "Round ": nb_rounds - round})


        
        
        if sync_:
            cdf(self.hit_ratios[1],"Local HR")      
            cdf(self.ndcgs[1],"Local NDCG")
            cdf(self.att_acc[1],"Attack accuracy in the last round")
            cdf(self.att_recall[1],"Attack recall in the last round")
            
            wandb.finish()

    
    def groundTruth_Clustering(self):
        users = []
        for u in range(len(self.all_participants)):
            vector = get_user_vector(u)
            users.append(get_distribution_by_genre(vector))

        users = np.array(users)
        model = KMeans(n_clusters=clustersK,random_state = 0).fit(users) 
        _labels = list(model.labels_)
        # centroids = model.cluster_centers_  
        # print("True cluster :",cluster_knn)
        # print("Cluster found:", self.cluster_found)
        # print("Intersection: ", intersection)
        # print(len(cluster_knn))
        # print(len(intersection))
        # acc = len(intersection)/ len(cluster_knn)
        # print("Accuracy :", acc)
        # sys.stdout.flush()

        return _labels
    
    def Accuracy_Clustering_Attack(self, clusters, attacker_id, idx):
        cluster_user = []
        for u,c in enumerate(clusters):
            # find the cluster of attacker_id and users that belong to it
            if c == clusters[attacker_id] and u!= attacker_id:
                cluster_user.append(u)
       
        intersection = set(cluster_user) & set(self.cluster_found[attacker_id][idx])
        if len(intersection == 0):
            acc = 1
            recall = 0
        else:
            acc = len(intersection) / len(self.cluster_found[attacker_id][idx]) 
            recall = len(intersection) / len(cluster_user)

        return acc, recall
        


        
        






