from pyopp import cSimpleModule, cMessage
import sys 
from collections import defaultdict
import matplotlib.pyplot as plt 
import numpy as np
import wandb
import os 

sync_ = 1
name_ = "Model-Age-Based"
dataset_ = "ml-100k" #foursquareNYC   
topK = 20

def cdf(data, metric):
    linestyle = "dashed"
    marker = "x"
    data_size=len(data)
     
    # Set bins edges
    data_set=sorted(set(data))
    bins=np.append(data_set, data_set[-1]+1)

    
    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(data, bins=bins, density=False)

   
    counts=counts.astype(float)/data_size   
    
    # Find the cdf
    cdf = np.cumsum(counts)
    

    # Plot the cdf
    # Save the cdf
    plt.plot(bin_edges[0:-1],cdf,linestyle = linestyle, linewidth=2)
    idx = np.arange(cdf.shape[0])
    plt.scatter(bin_edges[idx],cdf[idx], marker= marker, label = name_)
    plt.xlim((0,1))
    plt.ylabel("CDF")
    plt.xlabel(metric+str(topK))
    plt.grid(True)
    plt.legend()

if sync_:
    wandb_config = {
                "dataset": dataset_,
                "rounds": 400,
                "learning_rate": 0.01,
                "epochs": 2,
                "batch_size": 128,
                "topK": topK
                }
    os.environ["WANDB_API_KEY"] = "334fd1cd4a03c95f4655357b92cdba2b7d706d4c"
    os.environ["WANDB_MODE"] = "offline"
    run = wandb.init(project="DecentralizedGL", entity="drimfederatedlearning", name = name_, config = wandb_config)
   

class Server(cSimpleModule):
    
    def initialize(self):
        self.all_participants = [i for i in range(self.gateSize('sl'))]
        self.num_participants = len(self.all_participants)
        self.hit_ratios = defaultdict(list)
        self.ndcgs = defaultdict(list)

    
    def handleMessage(self, msg):
        if msg.getName() == "Performance":            
            self.hit_ratios[msg.round].append(msg.hit_ratio) # hit ratio ~ recall for recsys 
            self.ndcgs[msg.round].append(msg.ndcg) # ~ accuracy
            self.delete(msg)
    
    def finish(self):
        nb_rounds = max(self.hit_ratios.keys())
        for round in self.hit_ratios.keys():
            if(len(self.hit_ratios[round]) == self.num_participants): # if we receive all the users' model we evaluate them and then compute an average perf
                print("round = ", round)
                avg_hr = sum(self.hit_ratios[round]) / self.num_participants
                print("Average Test HR = ",avg_hr)
                avg_ndcg = sum(self.ndcgs[round]) / self.num_participants
                print("Average Test NDCG = ",avg_ndcg)
                sys.stdout.flush()
                if sync_:
                    run.log({"Average HR": avg_hr,"Average NDCG": avg_ndcg, "Round ": nb_rounds - round})

        # cdf(self.hit_ratios[1],"Local Hit Ratio")
        # wandb.log({"HR CDF": plt})
        # plt.clf()
        # cdf(self.ndcgs[1],"Local NDCG")
        # wandb.log({"NDCG CDF": plt})
        run.finish()
                    

