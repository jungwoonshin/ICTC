import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
import numpy as np

import args

class VGAE(nn.Module):
	def __init__(self, adj):
		super(VGAE,self).__init__()
		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)
		self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

	def encode(self, X):
		hidden = self.base_gcn(X)
		self.mean = self.gcn_mean(hidden)
		self.logstd = self.gcn_logstddev(hidden)
		gaussian_noise = torch.randn(X.size(0), args.hidden2_dim)
		sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
		return sampled_z

	def forward(self, X):
		Z = self.encode(X)
		A_pred = dot_product_decode(Z)
		return A_pred

class GraphConvSparse(nn.Module):
	def __init__(self, input_dim, output_dim, adj, activation = F.relu, **kwargs):
		super(GraphConvSparse, self).__init__(**kwargs)
		self.weight = glorot_init(input_dim, output_dim) 
		self.adj = adj
		self.activation = activation

	def forward(self, inputs):
		x = inputs
		# print(type(self.adj))
		# print(type(x))
		''' AB multiplication first '''
		# x = torch.mm(self.adj.to_dense(), x)
		# x = torch.mm(x,self.weight)

		''' BC multiplication first '''
		x = torch.mm(x,self.weight)
		x = torch.mm(self.adj, x)
		outputs = self.activation(x)
		return outputs

def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred

def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	# print('seed: ', args.seed)
	torch.manual_seed(args.weight_seed)
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	args.weight_seed += 1

	return nn.Parameter(initial)


class GAE(nn.Module):
	def __init__(self, adj, unnormalized_adj):
		super(GAE,self).__init__()
		# adj real-time update
		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

	def encode(self, X):
		hidden = self.base_gcn(X)
		z = self.mean = self.gcn_mean(hidden)
		return z

	def forward(self, X):
		# X = torch.cat((X, adj), 1)
		# X = torch.mm(X, self.resizing_weight)
		# print(feature_adj.shape)
		# exit()
		Z = self.encode(X)
		A_pred = dot_product_decode(Z)
		return A_pred

class LGAE(nn.Module):
	def __init__(self, adj, unnormalized_adj):
		super(LGAE,self).__init__()
		# adj real-time update
		self.first_layer =  TwohopGraphConvSparse(args.input_dim1, args.hidden1_dim, adj, activation=lambda x:x)
		# self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

	def encode(self, X):
		z = self.first_layer(X)
		return z

	def forward(self, X):
		# X = torch.cat((X, adj), 1)
		# X = torch.mm(X, self.resizing_weight)
		# print(feature_adj.shape)
		# exit()
		Z = self.encode(X)
		A_pred = dot_product_decode(Z)
		return A_pred
