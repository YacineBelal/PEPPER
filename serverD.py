from pyopp import cSimpleModule
import sys
from collections import defaultdict
from Dataset import Dataset
import numpy as np
import wandb
import os

sync_ = 1
name_ = "Pepper (FedAtt with 10% attackers)" #(FedAtt with 10% attackers)"  # "Model_Age_Based Attacked" #  "Pepper Attacked"
dataset_ = "ml-100k"  
topK = 20
dataset = Dataset("ml-100k")
Attackers_ratio = 0 #0.1
Worst_ratio = 0.2

def get_user_vector(user):
    positive_instances = []

    for (u, i) in dataset.trainMatrix.keys():
        if u == user:
            positive_instances.append(i)
        if u > user:
            break

    return positive_instances


def get_distribution_by_genre(vector):
    infos = []
    with open("u.item", 'r', encoding="ISO-8859-1") as info:
        line = info.readline()
        while line and line != '':
            arr = line.split("|")
            temp = arr[-19:]
            infos.append(temp)
            line = info.readline()

    dist = [0 for _ in range(19)]
    for item in vector:
        for i in range(len(dist)):
            dist[i] += int(infos[item][i])

    return dist

def cdf(data, metric, sync=sync_, topK=topK):
    data_size = len(data)

    # Set bins edges
    data_set = sorted(set(data))
    bins = np.append(data_set, data_set[-1] + 1)

    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(data, bins=bins, density=False)

    counts = counts.astype(float) / data_size

    # Find the cdf
    cdf = np.cumsum(counts)
    idx = np.arange(cdf.shape[0])
    data = [[x, y] for (x, y) in zip(bin_edges[idx], cdf[idx])]
    if sync:
        if topK == None:
            table = wandb.Table(data=data, columns=[metric, "CDF"])
            wandb.log({metric + " CDF ": wandb.plot.line(table, metric, "CDF", stroke="dash",
                                                         title=metric + " last round cumulative distribution")})

        else:
            table = wandb.Table(data=data, columns=[metric + "@" + str(topK), "CDF"])
            wandb.log(
                {metric + "@" + str(topK) + " CDF": wandb.plot.line(table, metric, "CDF", stroke="dash",
                                                                               title=metric + " last round cumulative distribution")})
if sync_:
    wandb_config = {
        "Dataset": dataset_,
        "Implementation": "TensorFlow",
        "Rounds": 300,
        "Nodes": 100,
        "Learning_rate": 0.01,
        "Epochs": 2,
        "Batch_size": "Full",
        "TopK": topK,
        "Attacker ratio": "10%",
        "Pull": False,
        "Epsilon": np.inf,
        "Delta": np.inf,
    }

    os.environ["WANDB_API_KEY"] = "334fd1cd4a03c95f4655357b92cdba2b7d706d4c"
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_START_METHOD"] = "fork"
    # os.environ["WANDB_SILENT"] = "true"
    wandb.init(project="DecentralizedGL", entity="drimfederatedlearning", name=name_, config=wandb_config)

def zero():
    return 0

class Server(cSimpleModule):

    def initialize(self):
        self.all_participants = [i for i in range(self.gateSize('sl'))]
        self.num_participants = len(self.all_participants) #int(len(self.all_participants) - (len(self.all_participants) * Attackers_ratio))
        self.hit_ratios = defaultdict(list)
        self.ndcgs = defaultdict(list)
        self.attackers = defaultdict(list)


    def handleMessage(self, msg):

        if msg.attacker == 0:
            self.hit_ratios[msg.round].append(msg.hit_ratio)  # hit ratio ~ recall for recsys
            self.ndcgs[msg.round].append(msg.ndcg)  # ~ accuracy

        self.delete(msg)

    def finish(self):
        global wandb
        nb_rounds = max(self.hit_ratios.keys())
        best_round = 0
        best_avg_hr = 0
        for round in self.hit_ratios.keys():
            avg_hr = sum(self.hit_ratios[round]) / self.num_participants
            print("Average Test HR = ", avg_hr)
            if avg_hr > best_avg_hr:
                best_round = round
                best_avg_hr = avg_hr
                
            avg_ndcg = sum(self.ndcgs[round]) / self.num_participants
            print("Average Test NDCG = ",avg_ndcg)
            sys.stdout.flush()
            worst_hr = sorted(self.hit_ratios[round])[:int(self.num_participants * Worst_ratio)] # to have 10% here, might change later
            worst_avg_hr = sum(worst_hr) / int(self.num_participants * Worst_ratio)
            worst_ndcg = sorted(self.ndcgs[round])[:int(self.num_participants * Worst_ratio)] 
            worst_avg_ndcg = sum(worst_ndcg) / int(self.num_participants * Worst_ratio)      
            if sync_ :
                wandb.log({"Average HR": avg_hr, "Average NDCG": avg_ndcg, "Worst 20% Average HR": worst_avg_hr, "Worst 20% Average NDCG": worst_avg_ndcg, \
                           "Round ": nb_rounds - round})
                if round == 0:
                    wandb.log({"Final Average HR": avg_hr, "Final Average NDCG": avg_ndcg,
                     "Final Worst 20% Average HR": worst_avg_hr, "Final Worst 20% Average NDCG": worst_avg_ndcg})

        if sync_:
            cdf(self.hit_ratios[best_round], "Local HR")
            cdf(self.ndcgs[best_round], "Local NDCG")
            cdf(worst_hr, "Worst 20\% Local HR")
            cdf(worst_ndcg, "Worst 20\% Local NDCG")
            wandb.finish()
       
