from re import I
from pyopp import cSimpleModule, cMessage
import sys 
from collections import defaultdict
# import matplotlib.pyplot as plt 
import numpy as np
import wandb
import os 

sync_ = 1
name_ = "Performance-Based TensorFlow-DP"
dataset_ = "ml-100k" #foursquareNYC   
topK = 20

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
                "rounds": 180,
                "learning_rate": 0.01,
                "epochs": 2,
                "batch_size": "Full",
                "topK": topK,
                "Epsilon": 1,
                "Delta": 10e-5,
                
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
    
    def handleMessage(self, msg):
        if msg.getName() == "Performance":            
            self.hit_ratios[msg.round].append(msg.hit_ratio) # hit ratio ~ recall for recsys 
            self.ndcgs[msg.round].append(msg.ndcg) # ~ accuracy
        elif msg.getName() == "FinalPerformance":
            self.hit_ratios[msg.round].append(msg.hit_ratio) # hit ratio ~ recall for recsys 
            self.ndcgs[msg.round].append(msg.ndcg) # ~ accuracy
            self.accuracy_rank[msg.round].append(msg.accuracy_rank)
            self.mse_performances[msg.round].append(msg.mse_performances)        
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

        if sync_:
            wandb.log({"Average MSE (weights delta)": avg_mse,
                    "Average Accuracy (of model ranking)": avg_acc})
            cdf(self.hit_ratios[1],"Local HR")      
            cdf(self.ndcgs[1],"Local NDCG")
            cdf(self.mse_performances[1],"MSE per node (weight difference)")
            cdf(self.accuracy_rank[1],"Ranking accuracy per node (Normalized weight ranking)")
            wandb.finish()