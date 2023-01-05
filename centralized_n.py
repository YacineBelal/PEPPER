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
from sklearn.manifold import TSNE
import plotly.express as ex

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

topK_clustering = 5
num_items =  1682 #38333
num_users = 943

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


def topk_per_items_embedding(model_n, train):
    users_topk = defaultdict(list)
    for u in range(len(model_n)):
        self_embedding = model_n[u].get_layer('user_embedding').get_weights()[0][u]
        _, items, labels = get_individual_train_instances(u, train)
        positives_u = []
        for i in range(len(items)):
            if labels[i] == 1:
                positives_u.append(items[i])

        for v in range(u + 1, len(model_n)):
            if v != u:
                items_embedding = model_n[v].get_layer('item_embedding').get_weights()[0]
                dist = 0
                for item in range(len(items_embedding)):
                    dist += 1 - distance.cosine(self_embedding, items_embedding[item])
                dist /= len(items_embedding)
                users_topk[u].append((v, dist))
                users_topk[v].append((u, dist))
        users_topk[u].sort(key=lambda x: x[1], reverse=True)
        users_topk[u] = [x[0] for x in users_topk[u]][:topK_clustering]

    return users_topk


def topk_per_evaluation_itemsonly(model_n, Ratings, Negatives):
    users_topk = defaultdict(list)
    for u in range(len(model_n)):
        u_ratings, u_negatives = get_individual_set(u, Ratings, Negatives)
        u_model = model_n[u]
        for v in range(len(model_n)):
            if u != v:
                v_ratings = u_ratings.copy()
                for i in range(len(v_ratings)):
                    v_ratings[i][0] = v
                v_model = model_n[v]
                u_model.get_layer("item_embedding").set_weights(v_model.get_layer("item_embedding").get_weights())
                hits, ndcgs = evaluate_model(u_model, u_ratings, u_negatives, K=20, num_thread=1)
                hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
                users_topk[u].append((v, ndcg))
        users_topk[u].sort(key=lambda x: x[1], reverse=True)
        users_topk[u] = [x[0] for x in users_topk[u]][:topK_clustering]

    return users_topk


def topk_per_evaluation(model_n, Ratings, Negatives):
    users_topk = defaultdict(list)
    for u in range(len(model_n)):
        u_ratings, u_negatives = get_individual_set(u, Ratings, Negatives)
        for v in range(len(model_n)):
            if u != v:
                v_ratings = u_ratings.copy()
                for i in range(len(v_ratings)):
                    v_ratings[i][0] = v
                v_model = model_n[v]
                hits, ndcgs = evaluate_model(v_model, v_ratings, u_negatives, K=20, num_thread=1)
                hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
                users_topk[u].append((v, ndcg))
        users_topk[u].sort(key=lambda x: x[1], reverse=True)
        users_topk[u] = [x[0] for x in users_topk[u]][:topK_clustering]

    return users_topk


def compute_individual_metrics(ground_truthK, users_topK):
    return len(set(ground_truthK) & set(users_topK))


def compute_metrics(ground_truthK, users_topK):
    accs = []
    for i in range(len(users_topK)):
        accs.append(len(set(ground_truthK[i]) & set(users_topK[i])) / topK_clustering)
    #
    plt.hist(accs, bins='auto')
    plt.xlabel("Accuracy of attack")
    plt.ylabel("Users")
    plt.grid(True)
    plt.show()
    print(accs)
    return sum(accs) / len(users_topK)


def get_individual_set(user, ratings, negatives):
    personal_Ratings = []
    personal_Negatives = []

    for i in range(len(ratings)):
        idx = ratings[i][0]
        if idx == user:
            personal_Ratings.append(ratings[i].copy())
            personal_Negatives.append(negatives[i].copy())
        elif idx > user:
            break

    return personal_Ratings, personal_Negatives


def remove_duplicates():
    users_liked = defaultdict(list)
    new_lines = []
    with open("foursquareNYC.train.rating","r") as f:
        lines = f.readlines()
        for line in lines:
            arr = line.split("\t")
            user, item = int(arr[0]), int(arr[1])
            if item not in users_liked[user]:
                users_liked[user].append(item) 
                new_lines.append(line)
        
    with open("foursquareNYC_new.train.rating","w") as w:
        w.writelines(new_lines)
        print('here')


def create_train_negatives(user, train):
    _, item_input, _ = get_individual_train_instances(user, train, only_positives=True)
    lines = []
    with open("ml-100k.train.negative", "a") as f:
        for item in item_input:
            negatives = []
            for _ in range(99):
                j = np.random.randint(0, num_items)
                while j in item_input:
                    j = np.random.randint(0, num_items)
                negatives.append(j)
            negatives_str = "\t".join(str(n) for n in negatives)
            line = "(" + str(user) + "," + str(item) + ")" + " \t " + negatives_str + "\n"
            lines.append(line)
        f.writelines(lines)


def get_training_as_list(train):
    trainingList = []
    for (u, i) in train.keys():
        if u >= num_users:
            break
        trainingList.append([u, i])

    return trainingList


def get_individual_train_instances(user, train, only_positives=False, num_negatives=4):
    user_input, item_input, labels = [], [], []
    for (u, i) in train.keys():
        if u == user:
            user_input.append(u)
            item_input.append(i)
            labels.append(1)
            # if not only_positives:
            #     for _ in range(num_negatives):
            #         j = np.random.randint(num_items)
            #         while (u, j) in train:
            #             j = np.random.randint(num_items)
            #         user_input.append(u)
            #         item_input.append(j)
            #         labels.append(0)
    return user_input, item_input, labels


def get_train_instances(train, num_negatives=4):
    user_input, item_input, labels, ratingList = [], [], [], []
    for (u, i) in train.keys():
        if u >= num_users:
            break
        user_input.append(u)
        item_input.append(i)
        ratingList.append([u, i])
        labels.append(1)
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels


evaluation_threads = 1  # mp.cpu_count()
topK = 20
epochs = 4
batch_size = 64
verbose = 1

dataset = Dataset("ml-100k")


train, testRatings, testNegatives, _, _, _ = dataset.trainMatrix, dataset.testRatings, \
                                                          dataset.testNegatives, dataset.trainNegatives, \
                                                          dataset.validationRatings, dataset.validationNegatives
# remove_duplicates()
# size = 0 
for user in range(num_users):
    create_train_negatives(user, train)

exit(1)

testRatings = testRatings[:1000]
testNegatives = testNegatives[:1000]

model = get_model(num_items, num_users)  # each node initialize its own model

init_weights = model.get_weights().copy()

tsne = TSNE(n_components=2, random_state=0)

# model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy')
#
# # evaluation before start of training
# (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
# hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
# print('Init: HR = %.4f, NDCG = %.4f\t' % (hr, ndcg))
#
# best_hr, best_ndcg, best_iter = hr, ndcg, -1
#
# for epoch in range(epochs):
#     user_input, item_input, labels = get_train_instances(train)
#
#     # Training
#     hist = model.fit([np.array(user_input), np.array(item_input)],
#                      np.array(labels),
#                      batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)
#
#     if epoch % verbose == 0:
#         (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
#         hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
#         print('Iteration %d: HR = %.4f, NDCG = %.4f, loss = %.4f'
#               % (epoch, hr, ndcg, loss))
#         if hr > best_hr:
#             best_hr, best_ndcg, best_iter = hr, ndcg, epoch
#
# print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
#

# groundtruthK = topk_per_embedding(model)
# groundtruthK = get_topK(groundturth_metadata(dataset),
#                       groundtruth_topkitemsliked(dataset))
groundtruthK = get_topK(groundtruth_topkitemsliked(dataset))

models_n = []
for user in range(100):
    model = get_model(num_items, num_users)
    model.set_weights(init_weights)
    model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy')
    print('model of user %d' % user)

    user_input, item_input, labels = get_individual_train_instances(user, train)
    # Individual Training
    hist = model.fit([np.array(user_input), np.array(item_input)],
                     np.array(labels),
                     batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)
    while (hist.history['loss'][0] > 0.25):
        user_input, item_input, labels = get_individual_train_instances(user, train)
        # Individual Training
        hist = model.fit([np.array(user_input), np.array(item_input)],
                         np.array(labels),
                         batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)

    models_n.append(model)




accs = []
for i in range(100):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(models_n[i].get_layer("item_embedding").get_weights()[0])
    _, items, _ = get_individual_train_instances(i, train, only_positives=True)
    c1 = np.where(kmeans.labels_ == 0)[0]
    c2 = np.where(kmeans.labels_ == 1)[0]
    print("items = ",len(items))
    print("c1 ", c1.shape)
    print("c2 ", c2.shape)
    if c2.shape[0] < c1.shape[0]:
        accs.append(len(set(items) & set(list(c2))) / len(items))
    else:
        accs.append(len(set(items) & set(list(c1))) / len(items))

print(accs)
print("accuracy :", sum(accs) / len(accs))

# labels = np.full(num_items, "not liked")
# labels[items] = "liked"
#
# projections = tsne.fit_transform(models_n[0].get_layer("user_embedding").get_weights()[0])
# fig = ex.scatter(
#     projections, x=0, y=1, color=labels, labels={"color": "items"},
#     title="items embedding for user 0"
# )
# fig.show()

# labels = ["user 0", "user 91", "user 10"]
# embeddings = []
# embeddings.append(models_n[0].get_layer("user_embedding").get_weights()[0][0])
# embeddings.append(models_n[91].get_layer("user_embedding").get_weights()[0][91])
# embeddings.append(models_n[10].get_layer("user_embedding").get_weights()[0][10])
#
# projections = tsne.fit_transform(embeddings)
# fig = ex.scatter(
#     projections, x=0, y=1, color=labels, labels={"color": "users"},
#     title="user embedding (0,10 and 91) projected in 2D"
# )
#
# fig.show()

# items_embedding_topk = topk_per_items_embedding(models_n, train)
items_embedding_topk = topk_per_evaluation_itemsonly(models_n, get_training_as_list(train), trainNegatives)

# items_embedding_topk = topk_per_evaluation_itemsonly(models_n, testRatings, testNegatives)
print("ACC = %.2f " % compute_metrics(groundtruthK, items_embedding_topk))
