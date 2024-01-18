import numpy as np
import pandas as pd
from tqdm import tqdm

from skimage import io, transform

import torch
# import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from PIL import Image
from torchvision import models,transforms

from PIL import Image

class HAM10000Dataset(Dataset):
    def __init__(self, data, label_column='dx_cat',img_col='image_path', transform=None):
        self.data = data
        self.label_column = label_column
        self.img_col = img_col
        self.transform = transform 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx][self.img_col]
        image = Image.open(image_path) #RGB format

        label = torch.tensor(self.data.iloc[idx][self.label_column])

        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)

        return image, label
    
