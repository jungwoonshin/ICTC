import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, precision_score, confusion_matrix
import scipy.stats
from scipy import sparse
import networkx as nx

from input_data import *
from preprocessing import *
from postprocessing import *
import pickle
import math

def sigmoid(x):
    return 1 / (1 + np.exp(-x))     
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def get_precision_scores(edges_pos, edges_neg, adj_rec, adj_orig, adj_norm):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


    precision, recall, thresholds = precision_recall_curve(adj_orig.flatten(), adj_rec.flatten())
    with np.errstate(divide='ignore'):
        f1 = 2/(1/precision+1/recall)
    # f1 = precision+recall
    max_index = np.argmax(f1)
    max_threshold = thresholds[max_index]
    # print(max_threshold)

    adj_rec[adj_rec > max_threshold] = 1.0
    adj_rec[adj_rec <= max_threshold] = 0.0

        # Predict on test set of edges
    preds = []
    pos = []


    predicted_to_be_false_but_true =[]
    for e in edges_pos:
        if (adj_rec[e[0], e[1]].item() == 0.0):
            predicted_to_be_false_but_true.append((e[0],e[1]))

        # preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
        preds.append(adj_rec[e[0], e[1]].item())    
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:

        preds_neg.append(adj_rec[e[0], e[1]].item())
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])

    cm = confusion_matrix(labels_all, preds_all,labels=[1,0])
    print(cm)


    return f1[max_index],max_threshold, predicted_to_be_false_but_true

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_scores(edges_pos, edges_neg, adj_rec, adj_orig):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    pos = []
    for e in edges_pos:
        # print(e)
        # print(adj_rec[e[0], e[1]])
        # score = sigmoid(adj_rec[e[0], e[1]].item())
        # print(score)
        # preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
        
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:

        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].item()))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    # if epoch > 100:
    #     print("ANSWERS")
    #     print(labels_all)
    #     print("PREDICTIONS")
    #     print(preds_all)

    return roc_score, ap_score

def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy
def get_acc2(adj_rec, adj_label):
    labels_all = adj_label.view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy

def max1(t1, t2):
    combined = torch.cat((t1.unsqueeze(2), t2.unsqueeze(2)), dim=2)
    return torch.max(combined, dim=2)[0].squeeze(2)

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

def get_similarity_and_number_of_neighbors(adj_train,  adj_train_norm, u2id):
    A = adj_train_norm[:len(u2id),:len(u2id)]
    B = adj_train_norm[len(u2id):,len(u2id):]

    indice1 = largest_indices(A,A.shape[0])
    indice2 = largest_indices(B,B.shape[0])
    indice2 = (indice2[0]+len(u2id), indice2[1]+len(u2id))

    top5_indice1_x = indice1[0][:int(len(indice1[0])/2.)]
    top5_indice1_y = indice1[1][:int(len(indice1[0])/2.)]

    number_of_common_neighbor = []
    for (x,y) in zip(top5_indice1_x, top5_indice1_y):
        # print(x)
        idx_x = neighbor_x = set(adj_train[x].indices)
        idx_y = neighbor_y = set(adj_train[y].indices)
        intersection= neighbor_x.intersection(neighbor_y)
        # print(len(intersection))

        two_hop_neighbor_x = [adj_train[x].indices.tolist() for x in idx_x]
        two_hop_neighbor_x = set([item for sublist in two_hop_neighbor_x for item in sublist])

        two_hop_neighbor_y = [adj_train[x].indices.tolist() for x in idx_y]
        two_hop_neighbor_y = set([item for sublist in two_hop_neighbor_y for item in sublist])
        intersection2 = two_hop_neighbor_x.intersection(two_hop_neighbor_y)

        number_of_common_neighbor.append(len(intersection)+len(intersection2))
        # print(neighbor_y)
        # print(adj_train)
        
        # exit()
    mean_ncn = np.mean(number_of_common_neighbor)
    print(mean_ncn)

    top5_indice1_x = indice1[0][int(len(indice1[0])/2.):]
    top5_indice1_y = indice1[1][int(len(indice1[0])/2.):]

    number_of_common_neighbor = []
    for (x,y) in zip(top5_indice1_x, top5_indice1_y):
        # print(x)
        idx_x = neighbor_x = set(adj_train[x].indices)
        idx_y = neighbor_y = set(adj_train[y].indices)
        intersection= neighbor_x.intersection(neighbor_y)
        # print(len(intersection))

        two_hop_neighbor_x = [adj_train[x].indices.tolist() for x in idx_x]
        two_hop_neighbor_x = set([item for sublist in two_hop_neighbor_x for item in sublist])

        two_hop_neighbor_y = [adj_train[x].indices.tolist() for x in idx_y]
        two_hop_neighbor_y = set([item for sublist in two_hop_neighbor_y for item in sublist])
        intersection2 = two_hop_neighbor_x.intersection(two_hop_neighbor_y)

        number_of_common_neighbor.append(len(intersection)+len(intersection2))
        # print(neighbor_y)
        # print(adj_train)
        
        # exit()
    mean_ncn = np.mean(number_of_common_neighbor)
    print(mean_ncn)

def get_homo_scores(adj_train, u2id, v2id):
    graph =nx.from_numpy_matrix(adj_train.toarray())
    size = adj_train.shape[0]
    adj_train = nx.to_dict_of_lists(graph)

    scores = np.zeros((size,size))
    for x in range(len(u2id)):
        for y in range(len(u2id)):
            Sx = set(adj_train[x])
            Sy = set(adj_train[y])
            intersection = Sx.intersection(Sy)
            score =0.0
            for i in intersection:
                try:
                    if len(adj_train[i]) == 1.0:
                        neighbor_z = 1./math.log(len(adj_train[i])+0.1)
                    else:
                        neighbor_z = 1./math.log(len(adj_train[i]))
                except ZeroDivisionError:
                    #neighbor_z = 1/math.log(0.0000001)
                    neighbor_z = 0.
                    # print('zerodivision')
                score += neighbor_z
            scores[x,y] = score
            # scores[x,y] = sigmoid(len(intersection))

    for x in range(len(u2id),len(u2id)+len(v2id)):
        for y in range(len(u2id),len(u2id)+len(v2id)):
            Sx = set(adj_train[x])
            Sy = set(adj_train[y])
            intersection = Sx.intersection(Sy)
            score =0.0
            for i in intersection:
                try:
                    if len(adj_train[i]) == 1.0:
                        neighbor_z = 1./math.log(len(adj_train[i])+0.1)
                    else:
                        neighbor_z = 1./math.log(len(adj_train[i]))
                except ZeroDivisionError:
                    #neighbor_z = 1/math.log(0.0000001)
                    neighbor_z = 0.
                    # print('zerodivision')
                score += neighbor_z
            scores[x,y] = score
            # scores[x,y] = sigmoid(len(intersection))

    return scores