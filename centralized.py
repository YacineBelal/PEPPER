from utility import get_model, reset_random_seeds
from evaluate import evaluate_model
from Dataset import Dataset
from keras.optimizers import Adam, SGD
import numpy as np

from scipy.spatial import distance
import multiprocessing as mp
from collections import defaultdict, Counter
import seaborn as sns
import matplotlib
import time 

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

topK_clustering = 20
num_items = 1682
num_users = 100

reset_random_seeds()

def get_user_vector(user, dataset):
    positive_instances = []

    for (u, i) in dataset.trainMatrix.keys():
        if u == user:
            positive_instances.append(i)
        if u > user:
            break

    return positive_instances


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))


def get_topK(*topks, reverse_=True):
    if len(topks) == 0:
        return
    if len(topks) == 1:
        users_topk = topks[0]
        for u in range(num_users):
            users_topk[u].sort(key=lambda x: x[1], reverse=reverse_)
            users_topk[u] = [x[0] for x in users_topk[u]][:topK_clustering]
        return users_topk
    # combine similarities
    users_topk = topks[0]
    for u in range(num_users):
        for v in range(num_users - 1):
            for tk in range(1, len(topks)):
                users_topk[u][v] = (users_topk[u][v][0], users_topk[u][v][1] * topks[tk][u][v][1])
        users_topk[u].sort(key=lambda x: x[1], reverse=reverse_)
        users_topk[u] = [x[0] for x in users_topk[u]][:topK_clustering]

    return users_topk


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


def groundturth_metadata(dataset):
    users = []
    for u in range(num_users):
        vector = get_user_vector(u, dataset)
        dist = get_distribution_by_genre(vector)
        s = sum(dist)
        dist = [x / s for x in dist]
        users.append(dist)

    # scaler = StandardScaler(with_mean=False)
    # users = scaler.fit_transform(users)
    users_topk = defaultdict(list)
    for u1 in range(num_users):
        for u2 in range(num_users):
            if u1 != u2:
                users_topk[u1].append((u2, 1 - distance.cosine(users[u1], users[u2])))

    return users_topk


def groundtruth_topkitemsliked(dataset):
    users = []
    users_topk = defaultdict(list)
    for u in range(num_users):
        users.append(get_user_vector(u, dataset))
    for u in range(num_users):
        for v in range(num_users):
            if u != v:
                users_topk[u].append((v, jaccard_similarity(users[u], users[v])))

    return users_topk


def topk_per_embedding(model):
    users_topk = defaultdict(list)
    # considering only users embeddings
    users_embeddings = model.get_layer('user_embedding').get_weights()[0]
    for i in range(num_users):
        for j in range(i + 1, num_users):
            if i != j:
                # dist = np.linalg.norm(users_embeddings[i] - users_embeddings[j])
                dist = 1 - distance.cosine(users_embeddings[i], users_embeddings[j])
                users_topk[i].append((j, dist))
                users_topk[j].append((i, dist))
        users_topk[i].sort(key=lambda x: x[1], reverse=True)
        users_topk[i] = [x[0] for x in users_topk[i][:topK_clustering]]
    return users_topk


def compute_metrics(ground_truthK, users_topK):
    accs = []
    for i in range(len(ground_truthK)):
        accs.append(len(set(ground_truthK[i]) & set(users_topK[i])) / topK_clustering)

    plt.hist(accs, bins='auto')
    plt.xlabel("Accuracy of attack")
    plt.ylabel("Users")
    plt.grid(True)
    plt.show()
    # print(accs)
    return sum(accs) / num_users


def get_train_instances(train, num_negatives=4):
    user_input, item_input, labels = [], [], []
    for (u, i) in train.keys():
        if u >= num_users:
            break
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels


evaluation_threads = 1 #mp.cpu_count()
topK = 20
epochs = 9
batch_size = 128
verbose = 1

dataset = Dataset("ml-100k")
train, testRatings, testNegatives, _, _ = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives, \
                                          dataset.validationRatings, dataset.validationNegatives
testRatings = testRatings[:1000]
testNegatives = testNegatives[:1000]

model = get_model(num_items, num_users)  # each node initialize its own model

# itm_embd = model.get_layer("item_embedding").get_weights()[0]
# print(len(model.get_layer("item_embedding").get_weights()))
# print(len(model.get_weights()[0]))
# exit(1)

model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy')
groundtruthK = get_topK(groundturth_metadata(dataset),
                        groundtruth_topkitemsliked(dataset))  #

print(len(model.get_weights()[0]))
exit(1)
# evaluation before start of training
print("starting evaluation")
start = time.time()
(hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
print("it lasted ", time.time() - start, " seconds")


hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
print('Init: HR = %.4f, NDCG = %.4f\t' % (hr, ndcg))
acc = 0
# acc = compute_metrics(topk_per_embedding(model), groundtruthK)
# print('Init Acc = %3.f' % acc)
best_hr, best_ndcg, best_iter = hr, ndcg, -1

for epoch in range(epochs):
    user_input, item_input, labels = get_train_instances(train)

    # Training
    hist = model.fit([np.array(user_input), np.array(item_input)],
                     np.array(labels),
                     batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)

    if epoch % verbose == 0:
        (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
        hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
        # acc = compute_metrics(topk_per_embedding(model), groundtruthK)
        print('Iteration %d: HR = %.4f, NDCG = %.4f, loss = %.4f, ACC = %.4f'
              % (epoch, hr, ndcg, loss, acc))
        if hr > best_hr:
            best_hr, best_ndcg, best_iter = hr, ndcg, epoch

print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
