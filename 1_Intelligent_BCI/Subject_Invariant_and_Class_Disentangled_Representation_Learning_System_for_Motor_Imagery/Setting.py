import os

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

path = "/Data/ejjeon/"
model_path = create_dir(path + "Projects/TNNLS_2019_GIST/SV_model/")


PSD_data_path = path + "GIST_data/"
TIME_data_path = path + "GIST_data/TIME/"

n_ch = 64 # channel
n_cl = 2 # class
n_freq = 37 # frequency
n_time = 1024 # timepoint

learning_rate = 1e-4

bs = 253 # batch size
act = "elu"
bn = True
drop = True
is_train = True
in_norm = False
lrelu_alpha = 0.2
eval_epoch = 1
n_epoch = 500
alpha = 0.05

nfeat1 = 50
nfeat2= 100
nfeat3 = 120

