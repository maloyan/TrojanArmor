# datasets/gtsrb.py
import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

class GTSRB(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.download = download

        if self.download:
            self.download_dataset()

        if self.train:
            data_file = os.path.join(self.root, 'GTSRB/Final_Training/Images/train.csv')
        else:
            data_file = os.path.join(self.root, 'GTSRB/Final_Test/Images/test.csv')

        self.data = pd.read_csv(data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.data.iloc[index]['path'])
        img = Image.open(img_path)
        label = self.data.iloc[index]['class_id']

        if self.transform:
            img = self.transform(img)

        return img, label

    def download_dataset(self):
        # Implement dataset downloading and extraction here
        pass
