import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam, SGD
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp

import numpy as np
import os
import time

from input_data import *
from preprocessing import *
from postprocessing import *
import args
import model
import pickle
from scipy import linalg
# from OrderedSet import OrderedSet
from sklearn.utils.extmath import randomized_svd
from sklearn import metrics
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""



def sigmoid(x):
    return 1 / (1 + np.exp(-x))     

def get_scores(edges_pos, edges_neg, adj_rec):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    pos = []
    for e in edges_pos:
        # print(e)
        # print(adj_rec[e[0], e[1]].item())
        # exit()
        if e[0] < len(u2id) and e[1] < len(u2id):
            print('warning11')
        if e[0] > len(u2id) and e[1] > len(u2id):
            print('warning12')
        if e[0] > len(u2id) and e[1] < len(u2id):
            print('warning1*')  

        # score = sigmoid(adj_rec[e[0], e[1]].item())
        # print(score)
        score = adj_rec[e[0], e[1]].item()
        preds.append(score)
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        if e[0] < len(u2id) and e[1] <len(u2id):
            print('warning21')
        if e[0] > len(u2id) and e[1] > len(u2id):
            print('warning22')
        if e[0] > len(u2id) and e[1] < len(u2id):
            print('warning23')   

        # preds_neg.append(sigmoid(adj_rec[e[0], e[1]].item()))
        # score = sigmoid(adj_rec[e[0], e[1]].item())
        score = adj_rec[e[0], e[1]].item()
        preds_neg.append(score)
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    fpr, tpr, thresholds = metrics.roc_curve(labels_all, preds_all)
    auc_score = metrics.auc(fpr, tpr)
    return roc_score, ap_score

def getBrAndBtriangle(adj_train):
    adj_tuple = sparse_to_tuple(adj_train)
    edges = adj_tuple[0]
    percentage = 0.9
    num_train = int(np.floor(edges.shape[0] * percentage)) # 10%

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    B_r_idx = all_edge_idx[:num_train] # 90% of training edges
    B_triangle_idx = all_edge_idx[num_train:] # 10% of training edges
    B_r_edges = edges[B_r_idx]
    B_triangle_edges = edges[B_triangle_idx]

    data = np.ones(B_r_edges.shape[0])
    data2 = np.ones(B_triangle_edges.shape[0])
    # Re-build adj matrix
    B_r = sp.csr_matrix((data, (B_r_edges[:, 0], B_r_edges[:, 1])), shape=adj_train.shape)
    B_triangle = sp.csr_matrix((data2, (B_triangle_edges[:, 0], B_triangle_edges[:, 1])), shape=adj_train.shape)

    return B_r, B_triangle
def getBiSPM(B_r,B_triangle):
    B_r = B_r.toarray()
    B_triangle = B_triangle.toarray()
    rank = np.linalg.matrix_rank(B_r)

    # print('rank r: ', rank)

    U, s, Vh = linalg.svd(B_r, full_matrices=False)
    S = np.diag(s)
    # print('b_r.shape:',B_r.shape)
    # val = np.zeros((B_r.shape))
    # for i in range(0, rank):
    #     val += np.multiply(S[i,i] , np.outer(U[:, i].reshape(-1,1), Vh[i,:].reshape(1,-1)))

    # print('print(np.allclose(adj_train, np.dot(U, np.dot(S, Vh)))): ',np.allclose(adj_train, np.dot(U, np.dot(S, Vh))))
    # print('adj_train.shape[0]*adj_train.shape[1]: ', adj_train.shape[0]*adj_train.shape[1])
    # print('  (val!=Br).sum(): ',(val!=B_r).sum())
    
    middle_val = np.matmul(np.transpose(B_r), B_triangle)+ np.matmul(np.transpose(B_triangle),B_r) # 877 * 877
    result = np.zeros((B_r.shape), dtype='float64')
    for i in range(0, rank):
        left = np.matmul(Vh[i,:].reshape(1,-1),middle_val)
        right = Vh[i,:].reshape(-1,1)

        numerator = np.matmul(left,right)
        coefficient = 2.0 * S[i,i]
        denominator = np.multiply( coefficient, np.matmul(Vh[i,:].reshape(1,-1),right ))

        delta_sigma = numerator/denominator
        delta_sigma = delta_sigma[0][0]

        result += (S[i,i] + delta_sigma) *  np.matmul(U[:, i].reshape(-1,1), Vh[i,:].reshape(1,-1))

    # print('new:',np.allclose(B_r.T @ B_r @ Vh[0,:].reshape(-1,1),S[i,i]**2.0 * Vh[0,:].reshape(-1,1)))
    # B_hat = sigmoid(result)
    B_hat = result

    Bi_adjacency_left = np.concatenate((np.zeros((B_hat.shape[0],B_hat.shape[0])), np.transpose(B_hat)), axis=0)
    Bi_adjacency_right = np.concatenate((B_hat, np.zeros((B_hat.shape[1],B_hat.shape[1]))), axis=0)
    Bi_adjacency = np.concatenate((Bi_adjacency_left, Bi_adjacency_right), axis=1)
    return Bi_adjacency

test_ap_list = []
test_roc_list = []
test_precision_list = []
for i in range(10):

    adj, features,\
            adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, edges_all, false_edges = get_data(args.dataset)

    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    with open('data/bipartite/id2name/'+ str(args.dataset)  +'u2id.pkl', 'rb') as f:
        u2id = pickle.load(f)
    with open('data/bipartite/id2name/'+ str(args.dataset) +'v2id.pkl', 'rb') as f:
        v2id = pickle.load(f)

    adj_train = adj_train[:len(u2id),len(u2id):]


    B_r, B_triangle = getBrAndBtriangle(adj_train)
    
    Bi_adjacency = getBiSPM(B_r,B_triangle)

    test_precision = get_precision(test_edges, test_edges_false, Bi_adjacency, adj_orig, sparse_to_tuple(sparse.csr_matrix(train_edges))[0], u2id, v2id)
    test_roc, test_ap = get_scores(test_edges, test_edges_false, Bi_adjacency)
    print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
              "test_ap=", "{:.5f}".format(test_ap), 
              'test precision=','{:.5f}'.format(test_precision))
    test_roc_list.append(test_roc)
    test_ap_list.append(test_ap)
    test_precision_list.append(test_precision)
    # break

mean_roc, ste_roc = np.mean(test_roc_list), np.std(test_roc_list)/(args.numexp**(1/2))
mean_ap, ste_ap = np.mean(test_ap_list), np.std(test_ap_list)/(args.numexp**(1/2))
mean_precision, ste_precision = np.mean(test_precision_list), np.std(test_precision_list)/(args.numexp**(1/2))

print('BiSPM')
print('mean_roc=','{:.5f}'.format(mean_roc),', ste_roc=','{:.5f}'.format(ste_roc))
print('mean_ap=','{:.5f}'.format(mean_ap),', ste_ap=','{:.5f}'.format(ste_ap))
print('mean_precision=','{:.5f}'.format(mean_precision),', ste_precision=','{:.5f}'.format(ste_precision))
