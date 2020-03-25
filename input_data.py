'''
****************NOTE*****************
CREDITS : Thomas Kipf
since datasets are the same as those in kipf's implementation, 
Their preprocessing source was used as-is.
*************************************
'''
import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from collections import defaultdict
from networkx.algorithms import bipartite
import pickle 
import collections
import matplotlib.pyplot as plt
from collections import OrderedDict
from random import shuffle
import random

import args

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data_fd_neg(dataset):
    with open("data/fd_neg/id2disease.pickle", 'rb') as f:
        id2disease = pkl.load(f)
    with open("data/fd_neg/id2ingredient.pickle", 'rb') as f:
        id2ingredient = pkl.load(f)

    total_number_nodes = len(id2disease) + len(id2ingredient)
    print('total_number_nodes:',total_number_nodes)

    row = np.array([x for x in range(total_number_nodes)])
    col = np.array([x for x in range(total_number_nodes)])
    data = np.array([1 for x in range(total_number_nodes)])
    whole_x = csr_matrix((data, (row, col)), shape=(total_number_nodes, total_number_nodes))
    # print('x:',x)
    idx = int(whole_x.shape[0]*0.9)

    x = whole_x[:idx,:]
    tx = whole_x[idx:,:]
    allx = x

    f = open("data/fd_neg/edgelist_neg", 'r')
    graph = defaultdict(list)
    for line in f:
        edge = line.strip('\n').split(' ')
        graph[int(edge[0])].append(int(edge[1]))
    # print(graph)

    features = sp.vstack((allx, tx)).tolil()
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    # row = np.array([x for x in range(total_number_nodes)])
    # col = np.array([x for x in range(total_number_nodes)])
    # data = np.random.uniform(low=-1, high=1, size=(total_number_nodes,))
    # features = csr_matrix((data, (row, col)), shape=(total_number_nodes, total_number_nodes))

    print(adj.shape)
    print(features.shape)

    return adj, features
def load_data_fd(dataset):
    with open("data/fd/id2disease.pickle", 'rb') as f:
        id2disease = pkl.load(f)
    with open("data/fd/id2ingredient.pickle", 'rb') as f:
        id2ingredient = pkl.load(f)

    total_number_nodes = len(id2disease) + len(id2ingredient)
    print('total_number_nodes:',total_number_nodes)

    row = np.array([x for x in range(total_number_nodes)])
    col = np.array([x for x in range(total_number_nodes)])
    data = np.array([1 for x in range(total_number_nodes)])
    whole_x = csr_matrix((data, (row, col)), shape=(total_number_nodes, total_number_nodes))
    # print('x:',x)
    idx = int(whole_x.shape[0]*0.9)

    x = whole_x[:idx,:]
    tx = whole_x[idx:,:]
    allx = x

    f = open("data/fd/edgelist", 'r')
    graph = defaultdict(list)
    for line in f:
        edge = line.strip('\n').split(' ')
        graph[int(edge[0])].append(int(edge[1]))
    # print(graph)

    features = sp.vstack((allx, tx)).tolil()
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # print(type(adj))
    # print((adj!=adj.T).nnz==0)

    # row = np.array([x for x in range(total_number_nodes)])
    # col = np.array([x for x in range(total_number_nodes)])
    # data = np.random.uniform(low=-1, high=1, size=(total_number_nodes,))
    # features = csr_matrix((data, (row, col)), shape=(total_number_nodes, total_number_nodes))

    print(adj.shape)
    print(features.shape)

    return adj, features

def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)
    # print('x:',x.shape)
    # print('tx:',tx.shape)
    # print('allx:',allx)
    # print('graph:',graph)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    # total_number_nodes = len(graph.keys())
    # print(total_number_nodes)
    # row = np.array([x for x in range(total_number_nodes)])
    # col = np.array([x for x in range(total_number_nodes)])
    # data = np.array([1 for x in range(total_number_nodes)])
    # features = csr_matrix((data, (row, col)), shape=(total_number_nodes, total_number_nodes)).tolil()
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    print(adj.shape)
    print(features.shape)
    # exit()
    return adj, features

def load_data_featureless(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)
    # print('x:',x.shape)
    # print('tx:',tx.shape)
    # print('allx:',allx)
    # print('graph:',graph)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    total_number_nodes = adj.shape[0]
    print(total_number_nodes)
    row = np.array([x for x in range(total_number_nodes)])
    col = np.array([x for x in range(total_number_nodes)])
    data = np.array([1 for x in range(total_number_nodes)])
    features = csr_matrix((data, (row, col)), shape=(total_number_nodes, total_number_nodes)).tolil()
    # features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    print(adj.shape)
    print(features.shape)
    # exit()
    return adj, features




def load_data_biological(dataset, change_seed=False):
    
    f = open("data/bipartite/edgelist_"+dataset+"_bp", 'r')
    u_list = []
    v_list = []
    for line in f:
         # malaria case
        if len(line.strip('\n').split('\t')) == 4:
            edge = line.strip('\n').split('\t')
            edge1 = edge[1]
            edge2 = edge[2]
            u_list.append(edge1)
            v_list.append(edge2)
            continue 
        elif len(line.strip('\n').split('\t')) > 1:
            edge = line.strip('\n').split('\t')
        elif len(line.strip('\n').split(' '))> 1:
            edge = line.strip('\n').split(' ')
        else:
            edge = line.strip('\n').split('::')
        edge1 = edge[0]
        edge2 = edge[1]
        u_list.append(edge1)
        v_list.append(edge2)

    u_list= list( OrderedDict.fromkeys(u_list) )
    v_list = list( OrderedDict.fromkeys(v_list) )
    
    random.seed(0)
    shuffle(u_list)
    random.seed(0)
    shuffle(v_list)

    id2u = OrderedDict()
    u2id = OrderedDict()
    for index, val in enumerate(u_list):
        id2u[index] = val
        u2id[val] = index

    id2v = {}
    v2id = {}
    for index, val in enumerate(v_list):
        index = index+len(u_list)
        id2v[index] = val
        v2id[val] = index

    with open(str(args.dataset) +'id2v.pkl', 'wb') as f:
        pickle.dump(id2v, f)
    with open(str(args.dataset) +'id2u.pkl', 'wb') as f:
        pickle.dump(id2u, f)
    with open(str(args.dataset) +'u2id.pkl', 'wb') as f:
        pickle.dump(u2id, f)
    with open(str(args.dataset) +'v2id.pkl', 'wb') as f:
        pickle.dump(v2id, f)

    f = open("data/bipartite/edgelist_"+dataset+"_bp", 'r')
    graph = defaultdict(list)

    edge_list = []
    for line in f:
        # malaria case
        if len(line.strip('\n').split('\t')) == 4:
            edge = line.strip('\n').split('\t')
            edge1 = u2id[edge[1]]
            edge2 = v2id[edge[2]]
            edge_list.append((edge1,edge2))
            graph[edge1].append(edge2)
            continue 
        elif len(line.strip('\n').split('\t')) > 1:
            edge = line.strip('\n').split('\t')
        elif len(line.strip('\n').split(' '))> 1:
            edge = line.strip('\n').split(' ')
        else:
            edge = line.strip('\n').split('::')
        edge1 = u2id[edge[0]]
        edge2 = v2id[edge[1]]
        edge_list.append((edge1,edge2))
        graph[edge1].append(edge2)

    graph_dict_if_lists = graph
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph_dict_if_lists))

    # check if 0 is value is set on top-right and bottom-left side of adjacency matrix.
    warning=0
    warning += adj[:len(u2id),:len(u2id)].sum()
    warning += adj[len(u2id):,len(u2id):].sum()
    print('number of elements that are not 0: ', warning)
    
    total_number_nodes = adj.shape[0]
    print(total_number_nodes)
    row = np.array([x for x in range(total_number_nodes)])
    col = np.array([x for x in range(total_number_nodes)])
    data = np.array([1 for x in range(total_number_nodes)])
    features = csr_matrix((data, (row, col)), shape=(total_number_nodes, total_number_nodes)).tolil()
    
    print(adj.shape)
    print(features.shape)
    # exit()
    return adj, features

def load_data_citation(dataset):
    

    fileName = 'data/bipartite/bip_'+dataset+'_node_list'
    u_list = [int(line.rstrip('\n')) for line in open(fileName)]
    fileName = 'data/bipartite/bip_'+dataset+'_group_list'
    v_list = [int(line.rstrip('\n')) for line in open(fileName)]

    id2u = {}
    u2id = {}
    for index, val in enumerate(u_list):
        id2u[index] = val
        u2id[val] = index

    id2v = {}
    v2id = {}
    for index, val in enumerate(v_list):
        index = index+len(u_list)
        id2v[index] = val
        v2id[val] = index

    with open(str(args.dataset) +'id2v.pkl', 'wb') as f:
        pickle.dump(id2v, f)
    with open(str(args.dataset) +'id2u.pkl', 'wb') as f:
        pickle.dump(id2u, f)
    with open(str(args.dataset) +'u2id.pkl', 'wb') as f:
        pickle.dump(u2id, f)
    with open(str(args.dataset) +'v2id.pkl', 'wb') as f:
        pickle.dump(v2id, f)

    f = open("data/bipartite/edgelist_"+dataset+"_bp", 'r')
    graph = defaultdict(list)

    edge_list = []
    for line in f:
        edge = line.strip('\n').split('\t')
        edge1 = u2id[int(edge[0])]
        edge2 = v2id[int(edge[1])]
        edge_list.append((edge1,edge2))
        graph[edge1].append(edge2)


    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # G = nx.from_dict_of_lists(graph)
    # bipartite.biadjacency_matrix(G, G.nodes())
    # print(adj[0:len(id2u), 0: len(id2u)].shape)
    # print(type(adj.toarray()))
    # print(np.zeros((len(id2u),len(id2u))).shape)

    # adj = adj[0:len(id2u), len(id2u):]
    # to_compare = np.zeros((len(id2u),len(id2v)))
    # print(adj.shape)
    # print(to_compare.shape)
    # b = adj[len(id2u):, len(id2u):].toarray()
    # print(np.array_equal(adj.toarray(), to_compare))

    total_number_nodes = adj.shape[0]
    print(total_number_nodes)
    row = np.array([x for x in range(total_number_nodes)])
    col = np.array([x for x in range(total_number_nodes)])
    data = np.array([1 for x in range(total_number_nodes)])
    features = csr_matrix((data, (row, col)), shape=(total_number_nodes, total_number_nodes)).tolil()
    
    print(adj.shape)
    print(features.shape)
    # exit()
    return adj, features

def show_node_degree(dataset):


    f = open("data/bipartite/edgelist_"+dataset+"_bp", 'r')
    u_list = []
    v_list = []
    for line in f:
        if dataset == 'fdpos' or dataset =='fdneg':
            edge = line.strip('\n').split(' ')
        else:
            edge = line.strip('\n').split('\t')

        edge1 = edge[0]
        edge2 = edge[1]
        u_list.append(edge1)
        v_list.append(edge2)

    np.random.shuffle(u_list)
    u_list = set(u_list)

    np.random.shuffle(v_list)
    v_list = set(v_list)


    id2u = {}
    u2id = {}
    for index, val in enumerate(u_list):
        id2u[index] = val
        u2id[val] = index

    id2v = {}
    v2id = {}
    for index, val in enumerate(v_list):
        index = index+len(u_list)
        id2v[index] = val
        v2id[val] = index

    with open('id2v.pkl', 'wb') as f:
        pickle.dump(id2v, f)
    with open('id2u.pkl', 'wb') as f:
        pickle.dump(id2u, f)
    with open('u2id.pkl', 'wb') as f:
        pickle.dump(u2id, f)
    with open('v2id.pkl', 'wb') as f:
        pickle.dump(v2id, f)

    f = open("data/bipartite/edgelist_"+dataset+"_bp", 'r')
    graph = defaultdict(list)

    edge_list = []
    for line in f:
        if dataset == 'fdpos' or dataset=='fdneg':
            edge = line.strip('\n').split(' ')
        else:
            edge = line.strip('\n').split('\t')
        edge1 = u2id[edge[0]]
        edge2 = v2id[edge[1]]
        edge_list.append((edge1,edge2))
        graph[edge1].append(edge2)

    graph = nx.Graph()

    graph.add_nodes_from(id2u.keys(), bipartite=0)
    graph.add_nodes_from(id2v.keys(), bipartite=1)
    graph.add_edges_from(edge_list)

    G = graph

    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    # print "Degree sequence", degree_sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    # plt.bar(deg, cnt, width=0.80, color='b')
    plt.plot(deg, cnt)

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    
    left, right = plt.xlim()  # return the current xlim
    plt.xlim((left, right))   # set the xlim to left, right
    plt.xlim(left, right)
    # ax.set_xticks([d + 0.4 for d in deg])
    # ax.set_xticklabels(deg)

    plt.savefig('node degree distribution of '+ str(dataset))
    # plt.show()




def load_data_drug(dataset, change_seed=False):
    with open('data/drug_graph.pickle', 'rb') as f:
        drug_graph = pickle.load(f)
    with open('data/drug_adjacency.pickle', 'rb') as f:
        adj = pickle.load(f)

    file = open('data/drug_adjacency matrix.txt')
    next(file)
    u_name_list = []
    for line in file:
        u_name = line.strip('\n').split('\t')[0]
        u_name_list.append(u_name)

    with open(str(args.dataset) +'u2id.pkl', 'wb') as f:
        pickle.dump(u_name_list, f)

    v_name_list = range(adj.shape[0]-len(u_name_list))
    with open(str(args.dataset) +'v2id.pkl', 'wb') as f:
        pickle.dump(v_name_list, f)
    # check if 0 is value is set on top-right and bottom-left side of adjacency matrix.
    # warning=0
    # for i in range(len(u2id)):
    #     for j in range(len(u2id)):
    #         if adj[i,j] != 0:
    #             warning+=1

    # for i in range(len(u2id),len(u2id)+len(v2id)):
    #     for j in range(len(u2id),len(u2id)+len(v2id)):
    #         if adj[i,j] != 0:
    #             warning+=1

    # print(warning)
    # exit()
    total_number_nodes = adj.shape[0]
    print(total_number_nodes)
    row = np.array([x for x in range(total_number_nodes)])
    col = np.array([x for x in range(total_number_nodes)])
    data = np.array([1 for x in range(total_number_nodes)])
    features = csr_matrix((data, (row, col)), shape=(total_number_nodes, total_number_nodes)).tolil()
    
    print(adj.shape)
    print(features.shape)
    # exit()
    return adj, features










# def main():
#     # show_node_degree('fdpos')
#     # show_node_degree('fdneg')
    

#     show_node_degree('cora')
#     # show_node_degree('citeseer')
#     # show_node_degree('pubmed')

#     # show_node_degree('enzyme')
#     # show_node_degree('ionchannel')
#     # show_node_degree('gpcr')

#     # load_data_cora_bp_block_adjacency('cora')
# #     # adj, features = load_data_fd('cora')
# #     # adj, features = load_data('cora')
    
# main()

