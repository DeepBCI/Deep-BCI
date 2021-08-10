import numpy as np
from torch.utils.data import Dataset

class GigaDataset(Dataset):
    def __init__(self, dataset, subj_id,istrain=True):
        self.dataset = dataset
        self.len = len(dataset[1])
        self.subj_id = subj_id
        self.istrain = istrain
        self.targets = dataset[1]
        self.ch_idx = np.array(list(range(7, 11)) + list(range(12, 15)) + list(range(17, 21)) + list(range(32, 41)))

    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        y = self.targets[idx]
        X = self.dataset[0][idx, :, :, 0:250]
        X = X[:,self.ch_idx,:]

        return [X, y, self.subj_id, idx]

    def train(self):
        self.istrain = True

    def eval(self):
        self.istrain = False
