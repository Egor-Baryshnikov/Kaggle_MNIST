import numpy as np
from PIL import Image
from torch.utils import data as tdata
from torchvision import transforms as tt

def raw_to_img(raw, shape=(28,28)):
    raw = np.array(raw, dtype=np.int8)
    arr = raw.reshape(shape)
    img = Image.fromarray(arr, mode='L')
    return img

augm = tt.Compose([
    tt.Resize(size=(32, 32)),
    tt.RandomRotation(30),
    tt.RandomResizedCrop(size=32, scale=(.8, 1.2)),
    tt.ToTensor()
])

class MyDataset(tdata.Dataset):
    def __init__(self, data, targets, transform):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        x = self.transform(x)
            
        return x, y
    
def end_to_end_preprocess(data):
    X = data.iloc[:,1:].apply(raw_to_img, axis=1).to_dict()
    y = data.iloc[:,0].to_list()

    data = MyDataset(X, y, transform=augm)
    
    return data


