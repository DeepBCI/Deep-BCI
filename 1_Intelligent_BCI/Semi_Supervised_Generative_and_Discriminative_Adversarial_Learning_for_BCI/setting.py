import os

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# Pathes
path = "/home/wonjun/Desktop/pycharm-2017.2.3/PycharmProjects"
data_path = create_dir(path + "/DATA/New_one")
baseline_path = create_dir(path + "/Adversarial_Training/models/baselines")
adversarial_path = create_dir(path + "/Adversarial_Training/models/adversarial")
generation_path = create_dir(path + "/Adversarial_Training/figures/generation")
activation_path = create_dir(path + "/Adversarial_Training/figures/activation")
tsne_path = create_dir(path + "/Adversarial_Training/features")

# Experiment conditions
subject = 2
batch_size = 32
total_epoch = 30
n_noise = 400
learning_rate = 1e-5
window_size = 512
