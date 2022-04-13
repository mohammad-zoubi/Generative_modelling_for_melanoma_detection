import zipfile
import cv2
import os
import numpy as np
import json
import albumentations as A
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from PIL import Image
import os, os.path
import csv
from tqdm import tqdm 
import seaborn as sns
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from matplotlib.colors import ListedColormap

### create train and val ###############################################

train_folder= '/ISIC256/ISIC_pool/malignant_all/Cropped_resized/test'
val_folder = '/ISIC256/ISIC_pool/malignant_all/Cropped_resized/validation'

folder ='/ISIC256/ISIC_pool/malignant_all/Cropped_resized/all_non_train'

file_list = os.listdir(folder)
print(len(file_list))
train_part = int(0.5*len(file_list))
print(train_part)


for idx in tqdm(range(0,len(file_list))):
    img = cv2.imread(os.path.join(folder,file_list[idx]))
    if img is  None:
        continue

    path_out_train = os.path.join(train_folder, file_list[idx])
    path_out_val = os.path.join(val_folder, file_list[idx])

    if idx <= train_part:
        cv2.imwrite(path_out_train, img)
    else :
        cv2.imwrite(path_out_val, img)
