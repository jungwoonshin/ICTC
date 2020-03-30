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
from sklearn.utils.extmath import randomized_svd
from sklearn import metrics

# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA, NMF, non_negative_factorization
import nimfa
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

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
            print('warning1')
        if e[0] > len(u2id) and e[1] > len(u2id):
            print('warning1')

        score = sigmoid(adj_rec[e[0], e[1]].item())
        # print(score)
        # score = adj_rec[e[0], e[1]].item()
        preds.append(score)
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        if e[0] < len(u2id) and e[1] <len(u2id):
            print('warning2')
        if e[0] > len(u2id) and e[1] > len(u2id):
            print('warning2')
        # preds_neg.append(sigmoid(adj_rec[e[0], e[1]].item()))
        score = sigmoid(adj_rec[e[0], e[1]].item())
        # score = adj_rec[e[0], e[1]].item()
        preds_neg.append(score)
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    fpr, tpr, thresholds = metrics.roc_curve(labels_all, preds_all)
    auc_score = metrics.auc(fpr, tpr)
    # if epoch > 100:
    #     print("ANSWERS")
    #     print(labels_all)
    #     print("PREDICTIONS")
    #     print(preds_all)

    return roc_score, ap_score, auc_score


def getBrAndBtriangle02(adj_train):
    # adj_train += adj_train.T
    # adj_train = adj_train.toarray()

    adj_train = sp.csr_matrix(adj_train)
    # adj_train2[len(u2id):,:len(u2id)] = np.zeros((len(v2id),len(u2id)))
    adj_tuple = sparse_to_tuple(adj_train)
    edges = adj_tuple[0]
    percentage = 0.9
    num_train = int(np.floor(edges.shape[0] * percentage)) # 10%
    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)

    print('num_train: ',num_train)
    B_r_idx = all_edge_idx[:num_train] # 90% of training edges
    # B_r_idx = all_edge_idx # 100% of training edges
    B_triangle_idx = all_edge_idx[num_train:] # 10% of training edges
    B_r_edges = edges[B_r_idx]
    B_triangle_edges = edges[B_triangle_idx]

    print('len(B_r_edges): ', len(B_r_edges))
    print('len(B_triangle_edges): ', len(B_triangle_edges))

    data = np.ones(B_r_edges.shape[0])
    data2 = np.ones(B_triangle_edges.shape[0])

    B_r = sp.csr_matrix((data, (B_r_edges[:, 0], B_r_edges[:, 1])), shape=adj_train.shape).toarray()
    B_r = B_r + np.transpose(B_r)

    B_r += sigmoid(np.matmul(adj_train.toarray(),np.transpose(adj_train.toarray())))

    # B_r = sigmoid(B_r)
    adj_train += adj_train.T
    B_triangle = sp.csr_matrix((data2, (B_triangle_edges[:, 0], B_triangle_edges[:, 1])), shape=adj_train.shape).toarray()
    B_triangle += B_triangle.T

    print('B_r is symmetric: ', check_symmetric(B_r))
    return B_r, B_triangle
def getXY(S, adj_train):
    model = NMF(n_components=k, init='nndsvda',solver='mu',max_iter=400)
    X = model.fit_transform(adj_train.toarray())
    Y = model.components_
    print('finished computing factorizing XY')

    # print('X.shape:',X.shape)
    # print('Y.shape:',Y.shape)
    adj_train = adj_train.toarray()

    gamma = 0.5
    lamda = 2.0
    for i in range(200):
        new_X_num = adj_train@Y.T+ gamma*(S*adj_train) @ Y.T
        new_X_den = X@Y@Y.T + gamma * (S * (X@Y)) @ Y.T + lamda * X 
        # new_X_den[np.where(new_X_den == 0.)] = 0.0001
        new_X_den = np.where(new_X_den != 0, new_X_den, 0.0001 )
        new_X = new_X_num / new_X_den 

        new_Y_num = X.T@adj_train + gamma * X.T @(S*adj_train)
        new_Y_den = X.T@X@Y+gamma*X.T@(S*(X@Y))+lamda*Y
        new_Y_den = np.where(new_Y_den != 0, new_Y_den, 0.0001 )
        new_Y = new_Y_num / new_Y_den 

        X = np.multiply(X,new_X)
        Y = np.multiply(Y,new_Y)
    return X,Y

test_ap_list = []
test_roc_list = []
test_precision_list = []
for i in range(10):
    adj, features,\
            adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, edges_all, edges_false_all = get_data(args.dataset)

    with open('data/bipartite/id2name/' +args.dataset +'u2id.pkl', 'rb') as f:
        u2id = pickle.load(f)
    with open('data/bipartite/id2name/' +args.dataset +'v2id.pkl', 'rb') as f:
        v2id = pickle.load(f)

    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    adj_train += adj_train.T
    
    pca = PCA().fit(adj_train.toarray())
    cumulative_contribution_rate = list(np.cumsum(pca.explained_variance_ratio_))
    val = min(cumulative_contribution_rate, key=lambda x:abs(x-0.95))
    k = cumulative_contribution_rate.index(val)+1
    print(k)

    # S = get_aa_scores(adj_train,u2id,v2id)
    S = get_jc_scores(adj_train,u2id,v2id)
    # S = get_cpa_scores(adj_train,u2id,v2id)
    # S = get_cn_scores(adj_train,u2id,v2id)
    print('finished computing S')

    X, Y = getXY(S, adj_train)
    print('finished computing updating XY')

    B_hat = np.nan_to_num(X@Y)

    test_precision = get_precision(test_edges, test_edges_false, B_hat, adj_orig, sparse_to_tuple(sparse.csr_matrix(train_edges))[0], u2id, v2id)
    test_roc, test_ap, test_auc = get_scores(test_edges, test_edges_false, B_hat)
    print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
              "test_ap=", "{:.5f}".format(test_ap), 
              'test precision=','{:.5f}'.format(test_precision))
    # exit()

    test_roc_list.append(test_roc)
    test_ap_list.append(test_ap)
    test_precision_list.append(test_precision)
    # break

mean_roc, ste_roc = np.mean(test_roc_list), np.std(test_roc_list)/(args.numexp**(1/2))
mean_ap, ste_ap = np.mean(test_ap_list), np.std(test_ap_list)/(args.numexp**(1/2))
mean_precision, ste_precision = np.mean(test_precision_list), np.std(test_precision_list)/(args.numexp**(1/2))

print('mean_roc=','{:.5f}'.format(mean_roc),', ste_roc=','{:.5f}'.format(ste_roc))
print('mean_ap=','{:.5f}'.format(mean_ap),', ste_ap=','{:.5f}'.format(ste_ap))
print('mean_precision=','{:.5f}'.format(mean_precision),', ste_precision=','{:.5f}'.format(ste_precision))
