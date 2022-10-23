from cProfile import label
import pickle
from random import random, seed
import re
from collections import defaultdict
from configparser import ConfigParser
from click import style
import numpy as np
import matplotlib.pyplot as plt 
import random
from regex import P 
from scipy.stats import expon




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

def averagePerf(logs = "Alpha08_ml-100k20.txt", output="MDJ8.txt", output1="DMetaL_NDCG_5.txt"):
    with open(logs,"r") as f:
        with open(output,"w") as out, open(output1,"w") as out1:
            nodes_performance = [] #defaultdict(list)
            nodes_performance_meta = [] #defaultdict(list)
            for line in f:
                if  nextline := re.match("^Local META HR5:.*([0-9]\.[0-9]+)",line):
                    nodes_performance_meta.append(float(nextline.group(1)))
                    # out.write(nextline.group(1)+"\n")
                    # nodes_performance[re.findall("[0-9]+",prec_line.strip())[0]] = float(nextline.group(1)) if float(nextline.group(1)) > nodes_performance.get(re.findall("[0-9]+",prec_line.strip())[0],-1) else  nodes_performance[re.findall("[0-9]+",prec_line.strip())[0]]
                
                elif nextline := re.match("^Local HR =.*([0-9]\.[0-9]+)",line):
                    nodes_performance.append(float(nextline.group(1)))
                    out.write(nextline.group(1)+"\n")
                
                elif nextline := re.match("^Local META NDCG5 :.*([0-9]\.[0-9]+)",line):
                    out1.write(nextline.group(1)+"\n")
                    

# (sum(nodes_performance) / len(nodes_performance) ,
    print(len(nodes_performance))
    return  sum(nodes_performance) / len(nodes_performance)


print(averagePerf())
exit(1)
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



# plots = []
# legends = ["Model-age-based","Pepper","Decentralized Reptile"]
# files = ["overheadJ.ini","overheadDJ.ini","overheadJ.ini"]

def cdf(data, i):
    linestyle = ["solid","dashed","dashdot","dotted"]
    markers = ["v","s","8"]
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
    plots.append([bin_edges,cdf])
    plt.plot(bin_edges[0:-1],cdf, linestyle = linestyle[3], linewidth=2)
    # idx = random.sample(range(cdf.shape[0]),20)
    # print(cdf.shape[0])
    idx = np.arange(cdf.shape[0],step=70)
    plt.scatter(bin_edges[idx],cdf[idx], marker= markers[i], alpha=0.7, label = legends[i])
    # plt.xlim((0,1))
    plt.ylabel("CDF")
    plt.xlabel("Execution time (seconds)")
    plt.grid(True)


# extract_profilesAndEvaluateThem()


def overhead(config, style_idx = 0): 
    cost_file_name = config
    config_object = ConfigParser()
    config_object.read(cost_file_name)
    performance = config_object["performance"]
    min_local_update = 1000
    max_local_update = 0
    min_total_transfer_nb = 1000 
    max_total_transfer_nb = 0
    min_total_peersampling_time = 1000 
    max_total_peersampling_time = 0
    min_aggregation_time = 1000
    max_aggregation_time = 0
    min_total_metaupdate_time = 1000
    max_total_metaupdate_time = 0
    total_update_time = 0
    total_aggregation_time = 0
    data = []
    num_perf = 0
    for i in range(100):
        
        min_local_update = float(performance["total_update_time"+str(i)]) if float(performance["total_update_time"+str(i)]) < min_local_update else min_local_update  
        max_local_update = float(performance["total_update_time"+str(i)]) if float(performance["total_update_time"+str(i)]) > max_local_update else max_local_update  
        min_total_peersampling_time = float(performance["total_peersampling_time"+str(i)]) if float(performance["total_peersampling_time"+str(i)]) < min_total_peersampling_time else min_total_peersampling_time  
        max_total_peersampling_time = float(performance["total_peersampling_time"+str(i)]) if float(performance["total_peersampling_time"+str(i)]) > max_total_peersampling_time else max_total_peersampling_time
        min_total_transfer_nb = float(performance["total_transfer_nb"+str(i)]) if float(performance["total_transfer_nb"+str(i)]) < min_total_transfer_nb else min_total_transfer_nb  
        max_total_transfer_nb = float(performance["total_transfer_nb"+str(i)]) if float(performance["total_transfer_nb"+str(i)]) > max_total_transfer_nb else max_total_transfer_nb  
        # min_total_metaupdate_time = float(performance["total_metaupdate_time"+str(i)]) if float(performance["total_metaupdate_time"+str(i)]) < min_total_metaupdate_time else min_total_metaupdate_time  
        # max_total_metaupdate_time = float(performance["total_metaupdate_time"+str(i)]) if float(performance["total_metaupdate_time"+str(i)]) > max_total_metaupdate_time else max_total_metaupdate_time  

        total_aggregation_time += float(performance["total_aggregation_time"+str(i)])
        total_update_time += float(performance["total_update_time"+str(i)])
        min_aggregation_time = float(performance["total_aggregation_time"+str(i)]) if float(performance["total_aggregation_time"+str(i)]) < min_aggregation_time else min_aggregation_time  
        max_aggregation_time = float(performance["total_aggregation_time"+str(i)]) if float(performance["total_aggregation_time"+str(i)]) > max_aggregation_time else max_aggregation_time  
    
        # num_perf += 1
        # if style_idx == 1:
        #     data.append((float(performance["total_peersampling_time"+str(i)])+float(performance["total_update_time"+str(i)])+float(performance["total_aggregation_time"+str(i)])) * 283 / 400)
        # elif style_idx == 2:
        #     data.append(float(performance["total_peersampling_time"+str(i)])+float(performance["total_update_time"+str(i)])+float(performance["total_aggregation_time"+str(i)])+
        #     float(performance["total_metaupdate_time"+str(i)]))
        # else:
        #     data.append(float(performance["total_peersampling_time"+str(i)])+float(performance["total_update_time"+str(i)])+float(performance["total_aggregation_time"+str(i)]))

   
    # cdf(data, style_idx)



    print("Num participants :",num_perf)
    print("Min Update Time =",min_local_update)
    print("Max Update Time =",max_local_update)

    print("Min Meta Update Time =",min_total_metaupdate_time)
    print("Max Meta Update Time =",max_total_metaupdate_time)
    
    print("Min Peer-sampling Time =",min_total_peersampling_time)
    print("Max Peer-sampling Time =",max_total_peersampling_time)
    print("Min Exchanges =",min_total_transfer_nb)
    print("Max Exchanges =",max_total_transfer_nb)
    
    print("Min Aggregation Time =",min_aggregation_time)
    print("Max Aggregation Time =",max_aggregation_time)
    print(total_update_time / 400 * 283)
    print(total_aggregation_time / 400 * 283)
    print("total average overhead :", (total_aggregation_time + total_update_time) * 147 / 283 / 100)
# overhead("overheadDJ6.ini")


# averagePerf()

# for i,f in enumerate(files):
#     overhead(f,i)


#53.54 sec






# X = expon.rvs(scale = 283, size = 1000, random_state = 1)
# Y = expon.rvs(scale = 410, size = 1000, random_state = 1)
# Z = expon.rvs(scale = 490, size = 1000, random_state = 1)


# cdf(Y,0)
# cdf(X,1)
# cdf(Z,2)
# plt.legend()
# plt.savefig('CommunicationRounds.pdf') 
# plt.ylabel("CDF")
# plt.xlabel("Communications rounds")
# plt.xlim(0,3000)
# plt.show()



# # X = np.hstack([X,Y])
# colors = ["darkorange","royalblue"]
# plt.hist([X,Y], 20, density = True,histtype = "bar",color = colors, edgecolor='black')
# # plt.hist(Y)
# plt.legend(legends, fontsize = 18)
# # plt.margins(x=1)
# plt.tick_params(axis='x', which='major', labelsize=18)
# plt.tick_params(axis='y', which='major', labelsize=18)
# plt.xticks([150,300,600,1200,1800,2400])



# print(nodes_performance)
# print(nodes_performance.items())
# size = 0
# avg = 0
# nodes = []


# for user, perf in nodes_performance_onlyitems.items():
    # if nodes_performance_allparams[user] > perf:
        # size += 1
        # avg += perf
        # nodes.append((user,perf))
        # avg += perf
        # nodes.append(user)

    # nodes.append((user,perf - nodes_performance_allparams[user]))

# # print(nodes_performance_onlyitems)
# print(nodes)
# print(len(nodes))
# # print(size)
# # print(avg/size)
# with open("outliers","rb") as input:
#     outliers = pickle.load(input)

# print(outliers)
# print(len(outliers))
# print(len([x for x in nodes if int(x) in outliers]))


# 40% of nodes have the same performance
# 20% regressed 72% : if we consider the 10% max outliers 90% of them are included here; if we consider 15% max outliers, 66% of thems are included in these; still need to find an explanation for others but a good start neverthless 
# 40% improved, these nodes have an average perf of 89%




