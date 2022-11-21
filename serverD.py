from audioop import avg
import random
from pyopp import cSimpleModule
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cosine, euclidean
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from Dataset import Dataset
import numpy as np
import wandb
import os
import matplotlib.pyplot as plt

sync_ = 1 # ** synch on the wandb cloud **
dataset = Dataset("ml-100k")
name_ = "Pepper"  # "Model_Age_Based"
dataset_ = "ml-100k"  # foursquareNYC
topK = 20


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


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))


def indic(data):
    max = np.max(data, axis=1)
    std = np.std(data, axis=1)
    return max, std


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
        "Rounds": 600,
        "Nodes": 100,
        "Learning_rate": 0.01,
        "Epochs": 2,
        "Batch_size": "Full",
        "TopK": topK,
        "Pull": False,
        "Epsilon": np.inf,
        "Delta": np.inf,
    }

    os.environ["WANDB_API_KEY"] = "334fd1cd4a03c95f4655357b92cdba2b7d706d4c"
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_START_METHOD"] = "fork"
    wandb.init(project="DecentralizedGL", entity="drimfederatedlearning", name=name_, config=wandb_config)


def cosine_similarity(list1, list2):
    return 1 - cosine(list1, list2)


class Server(cSimpleModule):

    def initialize(self):
        self.all_participants = [i for i in range(self.gateSize('sl'))]
        self.num_participants = len(self.all_participants)
        self.hit_ratios = defaultdict(list)
        self.ndcgs = defaultdict(list)


    def handleMessage(self, msg):

        self.hit_ratios[msg.round].append(msg.hit_ratio)  # hit ratio ~ recall for recsys
        self.ndcgs[msg.round].append(msg.ndcg)  # ~ accuracy

        self.delete(msg)

    def finish(self):
        global wandb
        nb_rounds = max(self.hit_ratios.keys())
        best_hr = 0.0
        best_ndcg = 0.0
        best_round = 0
        for round in self.hit_ratios.keys():
            avg_hr = sum(self.hit_ratios[round]) / self.num_participants
            print("Average Test HR = ", avg_hr)
            avg_ndcg = sum(self.ndcgs[round]) / self.num_participants
            print("Average Test NDCG = ",avg_ndcg)
            sys.stdout.flush()
            if avg_ndcg > best_ndcg:
                best_ndcg = avg_ndcg
                best_hr = avg_hr
                best_round = round

            if sync_:
                wandb.log({"Average HR": avg_hr, "Average NDCG": avg_ndcg, 
                           "Round ": nb_rounds - round})
        
        if sync_:
            wandb.log({"Best Average HR": best_hr, "Best Average NDCG": best_ndcg})
            cdf(self.hit_ratios[best_round], "Local HR")
            cdf(self.ndcgs[best_round], "Local NDCG")
            wandb.finish()

       
