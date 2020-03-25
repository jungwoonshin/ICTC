'''
****************NOTE*****************
CREDITS : Thomas Kipf
since datasets are the same as those in kipf's implementation, 
Their preprocessing source was used as-is.
*************************************
'''
import numpy as np
import scipy.sparse as sp
import pickle
import torch
import args
import time

from input_data import *
from preprocessing import *

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def preprocess_graph_numpy(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized.toarray()
    
def getMaskForBiAdjanceyMatrix():
    top_left = np.zeros((len(u2id),len(u2id)))
    bottom_left = np.ones((len(v2id),len(u2id)))
    left = np.concatenate((top_left, bottom_left))

    top_right = np.ones((len(u2id),len(v2id)))
    bottom_right = np.zeros((len(v2id),len(v2id)))
    right = np.concatenate((top_right, bottom_right))

    adj_unnormalized = np.concatenate((left, right),axis=1)
    adj_unnormalized = sp.csr_matrix(adj_unnormalized)
    adj_unnormalized = sparse_to_tuple(adj_unnormalized)
    adj_unnormalized = torch.sparse.FloatTensor(torch.LongTensor(adj_unnormalized[0].T), 
                                torch.FloatTensor(adj_unnormalized[1]), 
                                torch.Size(adj_unnormalized[2]))
    return adj_unnormalized


def mask_bipartite_perturbation_test_edges(adj):
    print('args.dataset: ', args.dataset)
    with open(str(args.dataset) +'u2id.pkl', 'rb') as f:
        u2id = pickle.load(f)
    with open(str(args.dataset) +'v2id.pkl', 'rb') as f:
        v2id = pickle.load(f)
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]

    ''' original training/test'''
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))
    all_edge_idx = list(range(edges.shape[0]))
    np.random.seed(args.edge_idx_seed)
    np.random.shuffle(all_edge_idx)
    args.edge_idx_seed += 1
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    # Re-build adj matrix
    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    def isSetMember(a,b):
        setA = set()
        setB = set()

        for (x,y) in a:
            setA.add((x,y))
        for index in range(b.shape[0]):
            setB.add((b[index,0],b[index,1]))
        return len(setA.intersection(setB)) == 0

    if args.use_saved_edge_false:
        with open(str(args.dataset) +'_test_edges_false.pkl', 'rb') as f:
            test_edges_false = pickle.load(f)
        with open(str(args.dataset) +'_val_edges_false.pkl', 'rb') as f:
            val_edges_false = pickle.load(f)

        print('len(train_edges): ',len(train_edges))
        print('len(test_edges): ',len(test_edges))
        print('len(edges): ', len(edges))

        assert isSetMember(test_edges_false, edges)
        print('isSetMember(test_edges_false, edges) is True')
        assert isSetMember(val_edges_false, edges)
        print('isSetMember(val_edges_false, edges) is True')
        assert isSetMember(val_edges, train_edges)
        print('isSetMember(val_edges, train_edges) is True')
        assert isSetMember(test_edges, train_edges)
        print('isSetMember(test_edges, train_edges) is True')
        assert isSetMember(val_edges, test_edges)
        print('isSetMember(val_edges, test_edges) is True')

        return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, edges_all, None

    test_edges_false = []
    val_edges_false = []

    ''' only for movie 1million '''
    if args.dataset == 'movie1m' or args.dataset == 'movie100k' or args.dataset == 'pubmed' or args.dataset == 'nanet':

        top_right_adj = adj[:len(u2id),len(u2id):].toarray()
        indexes = np.where(top_right_adj==0.0)
        np.random.seed(args.edge_idx_seed)
        np.random.shuffle(indexes[0])
        np.random.seed(args.edge_idx_seed)
        np.random.shuffle(indexes[1])

        val_index_i = indexes[0][:num_val]
        val_index_j = np.array(indexes[1][:num_val]) + len(u2id)

        test_index_i = indexes[0][num_val:num_test+num_val]
        test_index_j =  np.array(indexes[1][num_val:num_test+num_val]) + len(u2id)

        false_edges = []
        for i in range(len(indexes[0])):
            idx_i = indexes[0][i]
            idx_j = indexes[1][i]
            false_edges.append([idx_i, idx_j])

        for i in range(len(val_edges)):
            idx_i = val_index_i[i]
            idx_j = val_index_j[i]
            val_edges_false.append([idx_i, idx_j])

        for i in range(len(test_edges)):
            idx_i = test_index_i[i]
            idx_j = test_index_j[i]
            test_edges_false.append([idx_i, idx_j])

        # print(test_edges_false)
        # print(val_edges_false)

        # print(np.hstack([val_edges_false, test_edges_false]))
        train_false_edges = np.delete(false_edges, val_edges_false + test_edges_false, axis=0)
        train_false_edges = train_false_edges[:len(train_edges)]

        assert isSetMember(test_edges_false, edges)
        print('isSetMember(test_edges_false, edges) is True')
        assert isSetMember(val_edges_false, edges)
        print('isSetMember(val_edges_false, edges) is True')
        assert isSetMember(val_edges, train_edges)
        print('isSetMember(val_edges, train_edges) is True')
        assert isSetMember(test_edges, train_edges)
        print('isSetMember(test_edges, train_edges) is True')
        assert isSetMember(val_edges, test_edges)
        print('isSetMember(val_edges, test_edges) is True')
        # print('train false edges!')
        return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, edges_all, false_edges

    np.random.seed(1)
    while len(test_edges_false) < len(test_edges):
        t = time.time()
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if idx_i < len(u2id) and idx_j < len(u2id):
            continue
        if idx_i >= len(u2id) and idx_j >= len(u2id):
            continue
        if idx_j < len(u2id) and idx_i < len(u2id):
            continue
        if idx_j >= len(u2id) and idx_i >= len(u2id):
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    np.random.seed(2)
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if idx_i < len(u2id) and idx_j < len(u2id):
            continue
        if idx_i >= len(u2id) and idx_j >= len(u2id):
            continue
        if idx_j < len(u2id) and idx_i < len(u2id):
            continue
        if idx_j >= len(u2id) and idx_i >= len(u2id):
            continue
        # if ismember([idx_i, idx_j], train_edges):
        #     continue
        # if ismember([idx_j, idx_i], train_edges):
        #     continue
        # if ismember([idx_i, idx_j], val_edges):
        #     continue
        # if ismember([idx_j, idx_i], val_edges):
        #     continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    top_right_adj = adj[:len(u2id),len(u2id):].toarray()
    indexes = np.where(top_right_adj==0.0)
    np.random.seed(0)
    np.random.shuffle(indexes[0])
    np.random.seed(0)
    np.random.shuffle(indexes[1])

    false_edges = []
    for i in range(len(indexes[0])):
        idx_i = indexes[0][i]
        idx_j = indexes[1][i] + len(u2id)
        if (idx_i, idx_j) in test_edges_false:
            continue
        if (idx_j, idx_i) in test_edges_false:
            continue
        false_edges.append([idx_i, idx_j])
        # false_edges.append([idx_j, idx_i])

    false_edges = false_edges[:len(edges_all)]

    # np.save('data/bipartite/watch_your_step/'+str(args.dataset)+'/train.txt.npy', train_edges)
    # np.save('data/bipartite/watch_your_step/'+str(args.dataset)+'/train.neg.txt.npy', false_edges[:len(train_edges)])
    # np.save('data/bipartite/watch_your_step/'+str(args.dataset)+'/test.txt.npy', test_edges)
    # np.save('data/bipartite/watch_your_step/'+str(args.dataset)+'/test.neg.txt.npy', test_edges_false)

    # # print(len(train_edges))
    # # print(len( false_edges[:len(train_edges)]))
    # # print(len(test_edges))
    # # print(len(test_edges_false))
    # # print(len(range(0,3)))

    # index = {'index':range(adj.shape[0])}
    # with open('data/bipartite/watch_your_step/'+str(args.dataset)+'/index.pkl', 'wb') as f:
    #     pickle.dump(index, f)

    # exit()

    if args.save_edge_false:
        with open(str(args.dataset) +'_test_edges_false.pkl', 'wb') as f:
            pickle.dump(test_edges_false, f)
        with open(str(args.dataset) +'_val_edges_false.pkl', 'wb') as f:
            pickle.dump(val_edges_false, f)
        print('test and valid edges saved!')
        exit()

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)


    warning=0
    for (i,j) in test_edges_false:
        # print(i,j)
        if (i < len(u2id) and j < len(u2id)) or (i > len(u2id) and j > len(u2id)):
            warning+=1
    print('warning:', warning)
    print('len(u2id): ',len(u2id))
    # exit()


    # NOTE: these edge lists only contain single direction of edge!
    print('len(train_edges): ',len(train_edges))
    print('len(val_edges): ',len(val_edges))
    print('len(test_edges): ',len(test_edges))
    print('len(edges): ', len(edges))
    print('len(val_edges_false):', len(val_edges_false))
    print('len(test_edges_false):', len(test_edges_false))
    print('len(false_edges):', len(false_edges))
    print('len(edges_all):', len(edges_all))
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, edges_all, false_edges

def mask_bipartite_fd_neg_test_edges(adj):
    with open("data/fd_neg/id2disease.pickle", 'rb') as f:
        id2disease = pickle.load(f)
    with open("data/fd_neg/id2ingredient.pickle", 'rb') as f:
        id2ingredient = pickle.load(f)
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.


    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.seed(args.edge_idx_seed)
    np.random.shuffle(all_edge_idx)
    args.edge_idx_seed += 1
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    np.random.seed(1)
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if idx_i <= len(id2disease) and idx_j <= len(id2disease):
            continue
        if idx_i >= len(id2disease) and idx_j >= len(id2disease):
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    np.random.seed(2)
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if idx_i <= len(id2disease) and idx_j <= len(id2disease):
            continue
        if idx_i >= len(id2disease) and idx_j >= len(id2disease):
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # for i in train_edges:
    #     print(i)
    # print(adj_train)
    # print(adj_train[0])
    # exit()

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, edges_all, None

def mask_bipartite_fd_test_edges(adj):
    with open("data/fd/id2disease.pickle", 'rb') as f:
        id2disease = pickle.load(f)
    with open("data/fd/id2ingredient.pickle", 'rb') as f:
        id2ingredient = pickle.load(f)
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.


    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.seed(args.edge_idx_seed)
    np.random.shuffle(all_edge_idx)
    args.edge_idx_seed += 1
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    np.random.seed(1)
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if idx_i < len(id2disease) and idx_j < len(id2disease):
            continue
        if idx_i >= len(id2disease) and idx_j >= len(id2disease):
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    np.random.seed(2)
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if idx_i < len(id2disease) and idx_j < len(id2disease):
            continue
        if idx_i >= len(id2disease) and idx_j >= len(id2disease):
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # for i in train_edges:
    #     print(i)
    # print(adj_train)
    # print(adj_train[0])
    # exit()

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, edges_all, None

def mask_bipartite_test_edges(adj):
    with open('u2id.pkl', 'rb') as f:
        u2id = pickle.load(f)
    with open('v2id.pkl', 'rb') as f:
        v2id = pickle.load(f)
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.seed(args.edge_idx_seed)
    np.random.shuffle(all_edge_idx)
    args.edge_idx_seed += 1
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    def isSetMember(a,b):
        setA = set()
        setB = set()

        for (x,y) in a:
            setA.add((x,y))
        for index in range(b.shape[0]):
            setB.add((b[index,0],b[index,1]))
        return len(setA.intersection(setB)) == 0

    top_right_adj = adj[:len(u2id),len(u2id):].toarray()
    indexes = np.where(top_right_adj==0.0)
    np.random.seed(args.edge_idx_seed)
    np.random.shuffle(indexes[0])
    np.random.seed(args.edge_idx_seed)
    np.random.shuffle(indexes[1])

    val_index_i = indexes[0][:num_val]
    val_index_j = np.array(indexes[1][:num_val]) + len(u2id)

    test_index_i = indexes[0][num_val:num_test+num_val]
    test_index_j =  np.array(indexes[1][num_val:num_test+num_val]) + len(u2id)

    false_edges = []
    val_edges_false = []
    test_edges_false = []

    for i in range(len(val_edges)):
        idx_i = val_index_i[i]
        idx_j = val_index_j[i]
        val_edges_false.append([idx_i, idx_j])

    for i in range(len(test_edges)):
        idx_i = test_index_i[i]
        idx_j = test_index_j[i]
        test_edges_false.append([idx_i, idx_j])

    for i in range(len(indexes[0])):
        idx_i = indexes[0][i]
        idx_j = indexes[1][i]
        if (idx_i, idx_j) in test_edges_false:
            continue
        if (idx_j, idx_i) in test_edges_false:
            continue
        false_edges.append([idx_i, idx_j])

    # print(test_edges_false)
    # print(val_edges_false)

    # print(np.hstack([val_edges_false, test_edges_false]))
    # train_false_edges = np.delete(false_edges, val_edges_false + test_edges_false, axis=0)
    # train_false_edges = train_false_edges[:len(train_edges)]

    assert isSetMember(test_edges_false, edges)
    print('isSetMember(test_edges_false, edges) is True')
    assert isSetMember(val_edges_false, edges)
    print('isSetMember(val_edges_false, edges) is True')
    assert isSetMember(val_edges, train_edges)
    print('isSetMember(val_edges, train_edges) is True')
    assert isSetMember(test_edges, train_edges)
    print('isSetMember(test_edges, train_edges) is True')
    assert isSetMember(val_edges, test_edges)
    print('isSetMember(val_edges, test_edges) is True')
    # print('train false edges!')
    false_edges = false_edges[:len(edges_all)]

    np.save('data/bipartite/watch_your_step/'+str(args.dataset)+'/train.txt.npy', train_edges)
    np.save('data/bipartite/watch_your_step/'+str(args.dataset)+'/train.neg.txt.npy', false_edges[:len(train_edges)])
    np.save('data/bipartite/watch_your_step/'+str(args.dataset)+'/test.txt.npy', test_edges)
    np.save('data/bipartite/watch_your_step/'+str(args.dataset)+'/test.neg.txt.npy', test_edges_false)

    # print(len(train_edges))
    # print(len( false_edges[:len(train_edges)]))
    # print(len(test_edges))
    # print(len(test_edges_false))
    # print(len(range(0,3)))

    index = {'index':range(adj.shape[0])}
    with open('data/bipartite/watch_your_step/'+str(args.dataset)+'/index.pkl', 'wb') as f:
        pickle.dump(index, f)

    # exit()

    np.random.seed(1)
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if idx_i <= len(u2id) and idx_j <= len(u2id) :
            continue
        if idx_i >= len(u2id)  and idx_j >= len(u2id) :
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    np.random.seed(2)
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if idx_i <= len(u2id)  and idx_j <= len(u2id) :
            continue
        if idx_i >= len(u2id)  and idx_j >= len(u2id) :
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    top_right_adj = adj[:len(u2id),len(u2id):].toarray()
    indexes = np.where(top_right_adj==0.0)
    np.random.seed(0)
    np.random.shuffle(indexes[0])
    np.random.seed(0)
    np.random.shuffle(indexes[1])

    false_edges = []
    for i in range(len(indexes[0])):
        idx_i = indexes[0][i]
        idx_j = indexes[1][i] + len(u2id)
        if (idx_i, idx_j) in test_edges_false:
            continue
        if (idx_j, idx_i) in test_edges_false:
            continue
        false_edges.append([idx_i, idx_j])
        # false_edges.append([idx_j, idx_i])

    false_edges = false_edges[:len(edges_all)]

    np.save('data/bipartite/watch_your_step/'+str(args.dataset)+'/train.txt.npy', train_edges)
    np.save('data/bipartite/watch_your_step/'+str(args.dataset)+'/train.neg.txt.npy', false_edges[:len(train_edges)])
    np.save('data/bipartite/watch_your_step/'+str(args.dataset)+'/test.txt.npy', test_edges)
    np.save('data/bipartite/watch_your_step/'+str(args.dataset)+'/test.neg.txt.npy', test_edges_false)

    # print(len(train_edges))
    # print(len( false_edges[:len(train_edges)]))
    # print(len(test_edges))
    # print(len(test_edges_false))
    # print(len(range(0,3)))

    index = {'index':range(adj.shape[0])}
    with open('data/bipartite/watch_your_step/'+str(args.dataset)+'/index.pkl', 'wb') as f:
        pickle.dump(index, f)

    # exit()

    # for i in train_edges:
    #     print(i)
    # print(adj_train)
    # print(adj_train[0])
    # exit()

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, edges_all, None

# mask_bipartite_fd_neg_test_edges("1")

def get_data(dataset):
    if dataset == 'citeseer' or dataset == 'cora' or dataset == 'pubmed':
        adj, features = load_data_citation(args.dataset)
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, edges_all,edges_false_all = mask_bipartite_perturbation_test_edges(adj)
    
    # if dataset == 'fdpos':
    #     adj, features = load_data_fd(args.dataset)
    #     adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, edges_all,edges_false_all = mask_bipartite_fd_test_edges(adj)
    # if dataset == 'fdneg':
    #     adj, features = load_data_fd_neg(args.dataset)
    #     adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, edges_all,edges_false_all = mask_bipartite_fd_neg_test_edges(adj)
    
    if dataset == 'ionchannel' or args.dataset == 'enzyme' or args.dataset == 'gpcr' or \
        args.dataset == 'movie100k' or args.dataset == 'sw' or args.dataset == 'movie1m' or \
        args.dataset =='malaria' or args.dataset=='nanet' or args.dataset == 'c2o':
        adj, features = load_data_biological(args.dataset)
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, edges_all, edges_false_all = mask_bipartite_perturbation_test_edges(adj)
    if dataset == 'drug':
        adj, features = load_data_drug(args.dataset)
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, edges_all, edges_false_all = mask_bipartite_perturbation_test_edges(adj)

    return adj, features,\
            adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, edges_all, edges_false_all