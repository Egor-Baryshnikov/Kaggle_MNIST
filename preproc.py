import numpy as np
from PIL import Image
import torch
from torch.utils import data as tdata
from torchvision import transforms as tt

def raw_to_img(raw, shape=(28,28)):
    raw = np.array(raw, dtype=np.int8)
    arr = raw.reshape(shape)
    img = Image.fromarray(arr, mode='L')
    return img

class MyDataset(tdata.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        
        self.transform = transform
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y
    
class DatasetFromSubset(tdata.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
    
def end_to_end_preprocess(data):
    X = data.iloc[:,1:].apply(raw_to_img, axis=1).to_dict()
    y = data.iloc[:,0].to_list()

    data = MyDataset(X, y)
    
    return data


