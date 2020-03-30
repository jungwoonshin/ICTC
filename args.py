### CONFIGS ###
# dataset = 'ionchannel'
# dataset='enzyme'
dataset = 'gpcr'
# dataset= 'malaria'
# dataset = 'nanet'
# dataset = 'c2o'

# dataset = 'cora'
# dataset ='citeseer'
# dataset ='pubmed'
# dataset = 'drug'
# dataset = 'movie100k'
# dataset = 'sw'
# dataset = 'movie1m'

# dataset = 'fdpos'
# dataset = 'fdneg'


model1 = 'LGAE' # GAE, LGAE
model2 = 'GAE' # MGAE, MLGAE, LGAE, GAE
# model2 = 'HLGAE'

# input_dim1 = 318 # gpcr bipartite
# input_dim1 = 1109 # enzyme bipartite
# input_dim1 = 1103 # malaria
# input_dim1 =1880 # nanet
# input_dim1 = 261 # country2org

# input_dim1 = 414 # ion channel bip
input_dim1 = 350 # drug
# input_dim1 = 2625 # movie100k
# input_dim1 = 32 # southernwomen
# input_dim1 = 9746 # movie1m
# input_dim = 243 # food disease negative
# input_dim = 175 # food disease positive
device = 1

learning_rate1 = 0.01
learning_rate2 = 0.01
# learning_rate = 0.00001 # fd pos neg, bipartite graphs

num_epoch1 = 200
num_epoch2 = 0

print_val = False
use_saved_edge_false = False
save_edge_false = False

hidden1_dim = 32
hidden2_dim = 16

numexp = 10

weight_seed = 100
edge_idx_seed = 100


# input_dim1 = 1123 # citeseer bipartite
# input_dim1 = 1611 #/ cora bipartite
# input_dim1 = 16859 # pubmed bipartite

# input_dim = 3327 # citeseer featureless
# input_dim =2708 # cora featureless 
