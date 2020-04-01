### CONFIGS ###

similarity = 'srnmf_cn'
# similarity = 'srnmf_jc'
# similarity = 'srnmf_cpa'

# dataset = 'gpcr'
# dataset='enzyme'
# dataset = 'ionchannel'
# dataset= 'malaria'
dataset = 'drug'
# dataset = 'sw'
# dataset = 'nanet'
# dataset = 'movie100k'

model1 = 'LGAE'
model2 = 'GAE' 

# input_dim1 = 318 # gpcr bipartite
# input_dim1 = 1109 # enzyme bipartite
# input_dim1 = 414 # ion channel bip
# input_dim1 = 1103 # malaria
input_dim1 = 350 # drug
# input_dim1 = 32 # southernwomen
# input_dim1 =1880 # nanet
# input_dim1 = 2625 # movie100k

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
num_test = 10./1. # 10/1 means 10% means 10% is used as test sets., 10/2 means 20% is used as test sets. 

weight_seed = 100
edge_idx_seed = 100
