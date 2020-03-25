import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam, SGD
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import scipy.sparse as sp
import pickle
import scipy.stats
from collections import defaultdict
import numpy as np
import os
import time

from input_data import *
from preprocessing import *
from postprocessing import *

import args
import model


# Train on CPU (hide GPU) due to memory constraints
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def learn_train_adj(seed):
    global adj_train, adj, features, adj_norm, adj_label, weight_mask, weight_tensor, pos_weight, norm, num_feature, features_nonzero, num_nodes
    global train_edges, train_false_edges, val_edges, val_edges_false, test_edges, test_edges_false
    global u2id, v2id
    global adj_orig, adj_unnormalized, adj_norm_first
    # train model

    # init model and optimizer
    if torch.cuda.is_available():
        adj_norm = adj_norm.cuda()

    torch.manual_seed(seed)
    model_adj_norm = getattr(model,args.model1)(adj_norm, adj_unnormalized)
    optimizer = Adam(model_adj_norm.parameters(), lr=args.learning_rate1)

    if torch.cuda.is_available():
        features = features.cuda()
        adj_label = adj_label.cuda()
        model_adj_norm.cuda()
        weight_tensor = weight_tensor.cuda()

    for epoch in range(args.num_epoch1):
    # for epoch in range(args.num_epoch):

        t = time.time()
        A_pred = model_adj_norm(features.to_dense())
        optimizer.zero_grad()

        loss = log_lik = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
        if args.model1 == 'VGAE':
            kl_divergence = 0.5/ A_pred.size(0) * (1 + 2*model_adj_norm.logstd - model_adj_norm.mean**2 - torch.exp(model_adj_norm.logstd)).sum(1).mean()
            loss -= kl_divergence

        loss.backward(retain_graph=True)
        optimizer.step()

        train_acc = get_acc(A_pred,adj_label)

        val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred.cpu(), adj_orig)

        if args.print_val:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
                  "train_acc=", "{:.5f}".format(train_acc),
                  "val_roc=", "{:.5f}".format(val_roc),
                  "val_ap=", "{:.5f}".format(val_ap),
                  "time=", "{:.5f}".format(time.time() - t))

    test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred, adj_orig)
    print("End of training!", "test_pretrain_roc=", "{:.5f}".format(test_roc),
          "test_pretrain_ap=", "{:.5f}".format(test_ap))

    learn_train_adj = A_pred.cpu().detach().numpy()
    learn_train_adj = sp.csr_matrix(learn_train_adj)
    # sp.save_npz('data/'+str(args.model1) +'_' +str(args.dataset)+'_reconstructed_matrix.npz', learn_train_adj)
    # print('feature matrix saved!')
    # exit()
    return learn_train_adj, test_roc, test_ap

def run():

    global adj_train, adj, features, adj_norm, adj_label, weight_mask, weight_tensor, pos_weight, norm, num_feature, features_nonzero, num_nodes
    global train_edges, train_false_edges, val_edges, val_edges_false, test_edges, test_edges_false, edges_all, adj_all
    global u2id, v2id, adj_unnormalized
    global adj_orig, adj_norm_first

    print('cuda device= '+ str(args.device))     
    print('model1= '+ str(args.model1))     
    print('model2= '+ str(args.model2))
    print('dataset=' + str(args.dataset))
    print('learning rate1= '+ str(args.learning_rate1))
    print('learning rate2= '+ str(args.learning_rate2))
    print('numexp= '+ str(args.numexp))
    print('epoch1= '+ str(args.num_epoch1))
    print('epoch2= '+ str(args.num_epoch2))
    # train model
    test_ap_list = []
    test_roc_list = []
    
    test_ap_pretrain_list = []
    test_roc_pretrain_list = []

    for seed in range(args.numexp):
        adj, features,\
            adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, edges_all, edges_false_all = get_data(args.dataset)

        with open(str(args.dataset) +'u2id.pkl', 'rb') as f:
            u2id = pickle.load(f)
        with open(str(args.dataset) +'v2id.pkl', 'rb') as f:
            v2id = pickle.load(f)

        adj_orig = adj  
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()

        adj = adj_train

        num_nodes = adj.shape[0]

        features = sparse_to_tuple(features.tocoo())
        num_features = features[2][1]
        features_nonzero = features[1].shape[0]

        # Create Model
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

        features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), 
                                    torch.FloatTensor(features[1]), 
                                    torch.Size(features[2]))
        
        # Some preprocessing
        adj_norm = preprocess_graph(adj)
        adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), 
                                    torch.FloatTensor(adj_norm[1]), 
                                    torch.Size(adj_norm[2]))
        # Create Model
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)


        adj_label = adj_train + sp.eye(adj_train.shape[0])
        # adj_label = adj_label + get_homo_scores(adj_train, u2id, v2id)
        # adj_label = get_homo_scores(adj_train, u2id, v2id)
        # adj_label = sparse.csr_matrix(adj_label)
        adj_label = sparse_to_tuple(adj_label)
        adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), 
                                    torch.FloatTensor(adj_label[1]), 
                                    torch.Size(adj_label[2]))

        # weight_mask = adj_label.to_dense()[0:len(u2id), len(u2id):].contiguous().view(-1) == 1
        weight_mask = adj_label.to_dense().view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0)) 
        weight_tensor[weight_mask] = pos_weight

        adj_unnormalized = sp.coo_matrix(adj)
        adj_unnormalized = sparse_to_tuple(adj_unnormalized)
        adj_unnormalized = torch.sparse.FloatTensor(torch.LongTensor(adj_unnormalized[0].T), 
                                    torch.FloatTensor(adj_unnormalized[1]), 
                                    torch.Size(adj_unnormalized[2]))


        print('='*88)
        print(str(seed)+' iteration....' + str(args.learning_rate1) + ', '+ str(args.learning_rate2))
        print('='*88)

        adj_train_norm, test_roc_pretrain, test_ap_pretrain = learn_train_adj(seed)
        # adj_train_norm.tolil().setdiag(np.zeros(adj_train_norm.shape[0]))
        adj_train_norm = adj_train_norm.toarray()

        adj_norm = adj_norm.cpu().to_dense().numpy()
        
        A = np.matmul(adj_norm,adj_train_norm)
        AT = np.transpose(A)
        A_pred = (A+AT)/2.0 
        
        test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred, adj_orig)
        print("2nd model End of training!", "test_roc=", "{:.5f}".format(test_roc),
              "test_ap=", "{:.5f}".format(test_ap))

        test_roc_list.append(test_roc)
        test_ap_list.append(test_ap)

        test_ap_pretrain_list.append(test_ap_pretrain)
        test_roc_pretrain_list.append(test_roc_pretrain)

    mean_roc, ste_roc = np.mean(test_roc_list), np.std(test_roc_list)/(args.numexp**(1/2))
    mean_ap, ste_ap = np.mean(test_ap_list), np.std(test_ap_list)/(args.numexp**(1/2))

    mean_roc_pretrain, ste_roc_pretrain = np.mean(test_roc_pretrain_list), np.std(test_roc_pretrain_list)/(args.numexp**(1/2))
    mean_ap_pretrain, ste_ap_pretrain = np.mean(test_ap_pretrain_list), np.std(test_ap_pretrain_list)/(args.numexp**(1/2))

    print('cuda device= '+ str(args.device))     
    print('model1= '+ str(args.model1))     
    print('model2= '+ str(args.model2))
    print('dataset=' + str(args.dataset))
    print('learning rate1= '+ str(args.learning_rate1))
    print('learning rate2= '+ str(args.learning_rate2))
    print('numexp= '+ str(args.numexp))
    print('epoch1= '+ str(args.num_epoch1))
    print('epoch2= '+ str(args.num_epoch2))
    print('mean_roc=','{:.5f}'.format(mean_roc),', ste_roc=','{:.5f}'.format(ste_roc))
    print('mean_ap=','{:.5f}'.format(mean_ap),', ste_ap=','{:.5f}'.format(ste_ap))
    print('mean_roc_pretrain=','{:.5f}'.format(mean_roc_pretrain),', ste_roc=','{:.5f}'.format(ste_roc_pretrain))
    print('mean_ap_pretrain=','{:.5f}'.format(mean_ap_pretrain),', ste_ap=','{:.5f}'.format(ste_ap_pretrain))


run()
