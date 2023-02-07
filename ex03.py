import torch
import torch.nn as nn
import torch.optim as optim
import h5py
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR



file_name = "SynthText_train.h5"

db = h5py.File(file_name, 'r')
im_names = list(db['data'].keys())



# 1. Build a computation graph
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output
net = Net()

optimizer = optim.Adadelta(net.parameters(), lr=1.)  # 2. Setup optimizer
criterion = nn.NLLLoss()  # 3. Setup criterion


class HDF5Dataset(Dataset):

    transform = transforms.Compose([
    transforms.Resize((8, 8)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

    def __init__(self, h5_path):
        self.h5_path = "SynthText_train.h5"
        self.train = h5py.File(h5_path, 'r')
        self.length = len(h5py.File(h5_path, 'r'))

    def __getitem__(self, index): #to enable indexing
        # record = self.train[str(index)]
        im = im_names[0]
        img = db['data'][im][:]
        font = db['data'][im].attrs['font']
        # image = record['X'].value

        # transform to PIL image
        image = Image.fromarray(img, 'RGB') # assume your data is  uint8 rgb
        # label = record['y'].value

        # transformation here
        # torchvision PIL transformations accepts one image as input
        image = self.transform(image)
        return (
                image,
                font,
        )

    def __len__(self):
        return self.length

train_loader = DataLoader(HDF5Dataset("SynthText_train.h5")[:5], batch_size=512)

for inputs, target in train_loader:
    output = net(inputs)
    loss = criterion(output, target)
    print(round(loss.item(), 2))

    net.zero_grad()
    loss.backward()
    optimizer.step()