import os
import pickle
import jittor as jt
from jittor.dataset.dataset import Dataset


class RegressionDataset(Dataset):
    def __init__(self, path, batch=64):
        super(RegressionDataset, self).__init__()
        assert os.path.exists(path)
        with open(path, 'rb') as file:
            self.x, self.y = pickle.load(file)
        self.x, self.y = jt.array(self.x), jt.array(self.y)
        self.set_attrs(batch_size=batch, shuffle=True)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.x.shape[0]
