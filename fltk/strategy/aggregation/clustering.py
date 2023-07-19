from typing import Dict

import numpy as np
import torch
from collections import Counter
import hdbscan
from sklearn.metrics import accuracy_score, precision_score, recall_score


def clustering(stored_gradients, client_index, clients_per_round=10, stored_rounds=10):
    client_index = np.array(client_index)
    labels, major_label = clustering_(stored_gradients[-stored_rounds*clients_per_round:])
    aggregation_id = np.where(labels[-clients_per_round:] == major_label)[0]
    clients_id = client_index[-clients_per_round:][aggregation_id]
    return aggregation_id, clients_id

def clustering_(stored_gradients):
    # transform the stored gradients into a numpy array
    stored_gradients = np.concatenate(stored_gradients, axis=0)
    #clf = DBSCAN(eps=0.5, min_samples=5).fit(stored_gradients)
    clf = hdbscan.HDBSCAN(min_cluster_size=3).fit(stored_gradients)
    major_label = find_majority_label(clf)
    labels = clf.labels_
    return labels, major_label

def find_majority_label(clf):
    counts = Counter(clf.labels_)
    major_label = max(counts, key=counts.get)
    # major_id = set(major_id.reshape(-1))
    return major_label

def calculate_cluster_metrics(client_index, mal_index, candidates):
    y_true = [1 if i in mal_index else 0 for i in client_index]
    y_pred = [0 if i in candidates else 1 for i in client_index]
    # calculate the metrics
    # acc score
    acc = accuracy_score(y_true, y_pred)
    # precision score
    pre = precision_score(y_true, y_pred)
    # recall score
    rec = recall_score(y_true, y_pred)
    return acc, pre, rec

if __name__ == '__main__':
    #test the clustering function
    stored_gradients = [torch.Tensor([[1.1, 2.3, 3.6], [4.2, 5.1, 6.3]]), torch.Tensor([[7, 8, 9], [10, 11, 12]]), torch.Tensor([[13, 14, 15], [16, 17.12, 18.2]]), torch.Tensor([[19, 20, 21], [22, 23, 24]]), torch.Tensor([[25, 26, 27], [28, 29.12, 30.1]]), torch.Tensor([[31, 32, 33], [34, 35, 36]])]
    print(clustering(stored_gradients))
