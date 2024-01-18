import numpy as np
import pandas as pd
from tqdm import tqdm

from skimage import io, transform
# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import confusion_matrix

# import optuna

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from torchsummary import summary
import torch.optim as optim
from torch.autograd import Variable
from PIL import Image
import torchvision
from torchvision import models,transforms

from PIL import Image
import cv2

def compute_img_mean_std(image_paths,img_h,img_w):
    """
    Computing the mean and standard deviation of three channels on the whole dataset,
    without loading all images into memory at once. More time consuming but does not 
    fill up all the RAM.

    Args: image_paths (list): list of image paths
          img_h (int): height in pixels to resize
          img_w (int): width in pixels to resize

    Returns: 
        means (list): mean values of RGB channels
        stdevs (list): standard deviations of RGB channels
    """
    n_images = len(image_paths)
    channel_sum = np.zeros(3)
    channel_sum_squared = np.zeros(3)

    for path in tqdm(image_paths):
        img = cv2.imread(path) # Default BGR image
        img = cv2.resize(img, (img_h, img_w)) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #BGR to RGB
        img = img.astype(np.float32) / 255.0

        channel_sum += np.sum(img, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(img), axis=(0, 1))

    # Calculate mean and standard deviation
    means = channel_sum / (img_h * img_w * n_images)
    stdevs = np.sqrt(channel_sum_squared / (img_h * img_w * n_images) - np.square(means))

    print("Means = {}".format(means))
    print("Stds = {}".format(stdevs))
    return means, stdevs
    
