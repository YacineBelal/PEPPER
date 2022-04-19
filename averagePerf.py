import pickle
import re
from collections import defaultdict


def get_density_per_user():

    with open("pinterest-dense.train.rating") as f:
        lines = f.readlines()
        items = {}
        users = {}
        for l in lines:
            arr = l.split("\t")
            item = int(arr[1])
            user = int(arr[0])
            if item not in items:
                items[item] = 1
            else:
                items[item] += 1  
    
            if user not in users:
                users[user] = []
            users[user].append(item)

        user_density = {}
        for user in users:
            density = 0
            for item in users[user]:
                # print("item =",item)
                # print("rated by =",items[item])
                # print("density =",1 / items[item])
                density += items[item] / 4531
            user_density[user] = density/len(users[user])                
            
        
        return user_density

def averagePerf(logs = "Output.txt", output="ML_Pinterest.txt"):
    with open(logs,"r") as f:
        nodes_performance = {} #defaultdict(list)
        prec_line = f.readline()
        for line in f:
            nextline = re.match("^temporary local HR.*([0-9]\.[0-9]+)",line)
            if nextline:
                # f.readline()
                # line = f.readline()
                # line = f.readline()
                # nextline =  re.match("^[Tt]emporary Precision.*([0-9]\.[0-9]+)",line)
                
                nodes_performance[re.findall("[0-9]+",prec_line.strip())[0]] = float(nextline.group(1)) if float(nextline.group(1)) > nodes_performance.get(re.findall("[0-9]+",prec_line.strip())[0],-1) else  nodes_performance[re.findall("[0-9]+",prec_line.strip())[0]]
            prec_line = line
      

        # print(nodes_performance.items())
        # print(len(nodes_performance))
        # print(results)
        # print(len(results))
        # print(sum(results) / len(results))
        # print("average performance = ",sum(nodes_performance.values())/ len(nodes_performance))
        return nodes_performance



def extract_profilesAndEvaluateThem(input = "MDJ_ml-100k20_onlyItems.txt"):
    with open(input,"r") as f:
        correspondances = []
        profiles = []
        for line in f:
            if re.match("^node :.*(\d+)",line):
                nodeid = re.findall("[0-9]+",line.strip())[0]
                for _ in range(5):
                    line = f.readline()
                    
                if correspondance_line:=re.match("My Correspondance =.*(0\.[0-9]+)", line):
                    # print("node : ", nodeid)
                    # print("correspondance", correspondance_line.group(1))
                    correspondances.append((int(nodeid),float(correspondance_line.group(1))))
            elif re.match("Profiles Found", line):
                for _ in range(2):
                    profiles_line = f.readline()
                    lst = re.findall("\(([0-9]+, 0\.[0-9]+)\)",profiles_line)
                    for l in lst:
                        node, similarity = l.split(",")
                        profiles.append((int(node),float(similarity.strip())))

        correspondances =  sorted(correspondances,key = lambda x : x[1], reverse = True)
        print("Average correspondance in dataset :",sum([x[1] for x in correspondances if x[0] in [y[0] for y in profiles]])/ len(correspondances))

        correspondances_nodes = [x[0] for x in correspondances]
        # print(len(profiles))
        
        profiles = profiles[:30]
        summ = 0
        average = 0


        for n,s in profiles:
            if correspondances_nodes.index(n) < 20:
                summ += 1
            average += correspondances[correspondances_nodes.index(n)][1]
            print("node : ", n , " similarity : ", s , " correspondance rank : ", correspondances_nodes.index(n), " correspondance = ", correspondances[correspondances_nodes.index(n)][1])

        print("Average correspondance for the top 20 found users : ",average/20)
        print("number of top 20 corresponding nodes detected in top 20 :",summ)
        print("****************************************************** \n")

        return profiles, correspondances

# My Correspondance = 
#  0.47014851476767516
# 0.44166953787030544

# extract_profilesAndEvaluateThem()

nodes_performance_allparams = averagePerf("MDJ_ml-100k20_allparams.txt")
nodes_performance_onlyitems = averagePerf("MDJ_ml-100k20_onlyitems.txt")
# size = 0    
print("average performance all params = ", sum(nodes_performance_allparams.values()) / len(nodes_performance_allparams.values()))
print("average performance only items = ", sum(nodes_performance_onlyitems.values()) / len(nodes_performance_onlyitems.values()))


# print(nodes_performance)
# print(nodes_performance.items())
size = 0
avg = 0
nodes = []
for user, perf in nodes_performance_onlyitems.items():
    if nodes_performance_allparams[user] > perf:
        size += 1
        # avg += perf
        # nodes.append((user,perf))
        avg += perf
        nodes.append(user)

    # nodes.append((user,perf - nodes_performance_allparams[user]))

# print(nodes_performance_onlyitems)
print(nodes)
print(len(nodes))
# print(size)
# print(avg/size)
with open("outliers","rb") as input:
    outliers = pickle.load(input)

print(outliers)
print(len(outliers))
print(len([x for x in nodes if int(x) in outliers]))


# 40% of nodes have the same performance
# 20% regressed 72% : if we consider the 10% max outliers 90% of them are included here; if we consider 15% max outliers, 66% of thems are included in these; still need to find an explanation for others but a good start neverthless 
# 40% improved, these nodes have an average perf of 89%




