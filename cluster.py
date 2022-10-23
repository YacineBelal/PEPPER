from Dataset import Dataset
from sklearn.cluster import KMeans
import numpy as np
from yellowbrick.cluster.elbow import kelbow_visualizer
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
import pickle
import os

dataset = Dataset("ml-100k")
# dataset.trainMatrix, dataset.testRatings, dataset.testNegatives,dataset.validationRatings, dataset.validationNegatives

def get_user_vector(user):
    positive_instances = []
    # nb_user = 0
    # last_u = list(train.keys())[0]
    
    for (u,i) in dataset.trainMatrix.keys():
        # if(u != last_u):
        #     nb_user +=1
        #     last_u = u
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
    #alternatively you can calulate any other indicators
    max = np.max(data, axis=1)
    std = np.std(data, axis=1)
    return max, std

users = []
for u in range(100):
    vector = get_user_vector(u)
    users.append(get_distribution_by_genre(vector))

users = np.array(users)

# to find best K for clustering
#  kelbow_visualizer(KMeans(random_state = 0),users,k=(2,30))

model = KMeans(n_clusters=11,random_state = 0).fit(users) 
_labels = list(model.labels_)
print("labeeels = ",_labels)

# print("labels :",_labels)

centroids = model.cluster_centers_

users_distances = []


for i,u in enumerate(users):
    users_distances.append(euclidean(centroids[_labels[i]],u))


threshold = 0.9

threshold_max_distance = (sorted(users_distances, reverse=True)[int((1-threshold) * len(users_distances)):])[0]
print(threshold_max_distance)
print(max(users_distances))
outliers =[x for x in range(len(users)) if users_distances[x] > threshold_max_distance]
for x in outliers:
    model.labels_[x] = 11


print("Outliers :",outliers)
print("len outliers :",len(outliers))
with open("outliers","wb") as output:
    pickle.dump(outliers, output)



# # dimensionnality reduction 
# old_users = users.copy()
# pca = PCA(n_components = 3)
# users = pca.fit_transform(users)
# explanation = pca.explained_variance_ratio_


# fig = plt.figure(figsize = (10, 7))
# ax = plt.axes(projection ="3d")
 
# # Creating plot
# ax.scatter3D(users[model.labels_== 0,0], users[model.labels_== 0,1], users[model.labels_== 0,2], c='brown', label ='Cluster 1')
# ax.scatter3D(users[model.labels_== 1,0], users[model.labels_== 1,1], users[model.labels_== 1,2], c='blue', label ='Cluster 2')
# ax.scatter3D(users[model.labels_== 2,0], users[model.labels_== 2,1], users[model.labels_== 2,2], c='red', label ='Cluster 3')

# ax.scatter3D(users[model.labels_== 3,0], users[model.labels_== 3,1], users[model.labels_== 3,2], c='green', label ='Cluster 4')
# ax.scatter3D(users[model.labels_== 4,0], users[model.labels_== 4,1], users[model.labels_== 4,2], c='grey', label ='Cluster 5')
# ax.scatter3D(users[model.labels_== 5,0], users[model.labels_== 5,1], users[model.labels_== 5,2], c='cyan', label ='Cluster 6')
# ax.scatter3D(users[model.labels_== 6,0], users[model.labels_== 6,1], users[model.labels_== 6,2], c='magenta', label ='Cluster 7')
# ax.scatter3D(users[model.labels_== 7,0], users[model.labels_== 7,1], users[model.labels_== 7,2], c='purple', label ='Cluster 8')
# ax.scatter3D(users[model.labels_== 8,0], users[model.labels_== 8,1], users[model.labels_== 8,2], c='maroon', label ='Cluster 9')
# ax.scatter3D(users[model.labels_== 9,0], users[model.labels_== 9,1], users[model.labels_== 9,2], c='pink', label ='Cluster 10')
# ax.scatter3D(users[model.labels_== 10,0], users[model.labels_== 10,1], users[model.labels_== 10,2], c='black', label ='Cluster 11')
# ax.scatter3D(users[model.labels_== 11,0], users[model.labels_== 11,1], users[model.labels_== 11,2], c='yellow', label ='Cluster 11')

# # ax.scatter3D(users[outliers,0], users[outliers,1], users[outliers,2], c='yellow', label ='Cluster 11')


# plt.title("Clusters of users based on their movies' genre distribution")
# plt.show()


# count = Counter(_labels)
# biggest_cluster = list(count)[0]
# print(biggest_cluster)

nodes = []
for i in range(len(_labels)):
    if _labels[i] == 2:
        # print("user ",i)
        # print(old_users[i])
        nodes.append(i)

print(nodes)


# drama 8, romance 14, children's 4