from random import sample
from pyopp import cSimpleModule
import sys 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from Dataset import Dataset
import numpy as np
import wandb
import os 

sync_ = 1
name_ = "Model_Age_Based Attacked"
dataset_ = "ml-100k" #foursquareNYC   
topK = 20
topK_clustering = 10
clustersK= 7


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
        
    # summ = sum(dist)
    # dist = [elem / summ for elem in dist]
    
    return dist


def indic(data):
    max = np.max(data, axis=1)
    std = np.std(data, axis=1)
    return max, std


def cdf(data, metric, topK = topK):
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
    if topK == None:
        table = wandb.Table(data=data, columns = [metric+" CDF"])
        wandb.log({metric+"@"+"CDF" : wandb.plot.line(table, metric+"@"+str(topK), "CDF", stroke="dash",
               title = metric+"@"+str(topK)+" cumulative distribution")})

    else: 
        table = wandb.Table(data=data, columns = [metric+"@"+str(topK), "CDF"])
        wandb.log({metric + "CDF" : wandb.plot.line(table, metric, "CDF", stroke="dash",
               title = metric +" cumulative distribution")})
    

if  sync_:
    wandb_config = {
                "Dataset": dataset_,
                "Implementation": "TensorFlow",
                "Rounds": 40,
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
        self.cluster_found[msg.user_id].append(msg.cluster_found)
        
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
            idx_round +=1
            avg_acc = avg_acc / self.num_participants
            avg_recall = avg_recall / self.num_participants
             
            if sync_:
                    wandb.log({"Average Accuracy": avg_acc ,"Average Recall": avg_recall})


        
        
        if sync_:
            cdf(self.hit_ratios[1],"Local HR")      
            cdf(self.ndcgs[1],"Local NDCG")
            print("************* problem for att acc graph here?", self.att_acc.items())
            sys.stdout.flush()
            cdf(self.att_acc[1],"Attack accuracy (Last round)", topK_clustering)
            cdf(self.att_recall[1],"Attack recall (Last round)", topK_clustering)
            
            wandb.finish()

    def groundTruth_Clustering(self):
        users = []
        for u in range(len(self.all_participants)):
            vector = get_user_vector(u)
            users.append(get_distribution_by_genre(vector))
             
        users = np.array(users)
        scaler = StandardScaler(with_mean=False)
        users = scaler.fit_transform(users)

        model = KMeans(n_clusters = clustersK, init="k-means++", random_state=1245).fit(users) 
        _labels = list(model.labels_)
        
        self.silhouette_avg = silhouette_score(users, _labels)
        print("For n_clusters = ", clustersK, "Average silhouette score is ", self.silhouette_avg)
        sample_silhouette_values = silhouette_samples(users, _labels)
        print("Silhouette values :", sample_silhouette_values)
        sys.stdout.flush()

        if sync_:
            cdf(sample_silhouette_values,"Silhouette score value", topK = None)
        
        
        return _labels
    
    def Accuracy_Clustering_Attack(self, clusters, attacker_id, idx):
        if idx > len(self.cluster_found[attacker_id]):
            return 1 , 0 
            
        cluster_user = []
        for u,c in enumerate(clusters):
            # find the cluster of attacker_id and users that belong to it
            if c == clusters[attacker_id] and u!= attacker_id:
                cluster_user.append(u)
       
        interacted_with_fair_recall = []
        for u in cluster_user:
            if u in cluster_user and u in self.cluster_found[attacker_id][idx]:
                interacted_with_fair_recall.append(u) 

        found_and_relevant = set(cluster_user) & set(self.cluster_found[attacker_id][idx][:topK_clustering])
        acc = len(found_and_relevant) / len(self.cluster_found[attacker_id][idx][:topK_clustering])
        if len(interacted_with_fair_recall) == 0:
            recall = 1
        else:
            recall = len(found_and_relevant) / len(interacted_with_fair_recall)

        return acc, recall
        


        
        






