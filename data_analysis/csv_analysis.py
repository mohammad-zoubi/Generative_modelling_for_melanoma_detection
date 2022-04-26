''' Looking at synth100k images directories
    Here we have labels of images that are fed into melanoma_real classifier and we want to see how
    different softmax differ in image quality'''


import pandas as pd
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from image_samples  import image_grid
from tqdm import tqdm
from PIL import Image


CSV_PATH = "/ISIC256/ISIC256_ORIGINAL/synth100k_mal/shifted_imgs_dir/shifted_frames.csv"
# CSV_PATH_mel_class = '/ISIC256/ISIC256_ORIGINAL/synth100k_mal/'

labels_biases = ["label_hair_dense", "label_hair_short", "label_hair_medium", "label_black_frame", "label_ruler_mark", "label_other"] 

softmax_labels = []
image_paths = []

for i in (range(1,4)):
    # path_to_csv = f'/ISIC256/ISIC256_ORIGINAL/synth100k_mal/synth100k{i}_labels.csv'
    path_to_csv = f'/ISIC256/synth60k/labels_real_model/synth60k{i}_labels.csv'
    # path_to_bises = f'/ISIC256/ISIC256_ORIGINAL/synth100k_mal/synth100k_anno{i}.csv'
    path_to_bises = f'/ISIC256/synth60k/synth60k_anno{i}.csv'
    df = pd.read_csv(path_to_csv)
    df = df.sort_values(by=['image_name'])
    df_bias = pd.read_csv(path_to_bises, header=None)
    # non_biased_idx = np.where(df_bias.iloc[:, [1,2,3,4,5,6]].sum(axis=1) == 0)[0]
    non_biased_idx = np.where(df_bias.iloc[:, [1,2,3,4,5]].sum(axis=1) == 0)[0]
    
    tmp = np.asarray(df['predicted_labels'])[non_biased_idx]
    # images_names = df.image_name.tolist()
    images_names = np.asarray(df.image_name)[non_biased_idx]
    images_names = images_names.tolist()
    softmax_labels.append(tmp.tolist())
    # parent_path = f'/ISIC256/ISIC256_ORIGINAL/synth100k_mal/img_dirs/imgs_dirs_mal/imgs{i}/'
    parent_path = f'/ISIC256/synth60k/img_dir/imgs{i}/'
    
    image_paths.append([parent_path + s for s in images_names])
print(tmp)
image_paths = [item for sublist in image_paths for item in sublist]
softmax_labels = [item for sublist in softmax_labels for item in sublist]
image_paths = np.asarray(image_paths)
softmax_labels = np.array(softmax_labels)

idx1 = np.where(softmax_labels>=0.8888)[0]
idx2 = np.where((softmax_labels<0.8888) & (softmax_labels>=0.7777))[0]
idx3 = np.where((softmax_labels<0.7777) & (softmax_labels>=0.6666))[0]
idx4 = np.where((softmax_labels<0.6666) & (softmax_labels>=0.5555))[0]
idx5 = np.where((softmax_labels<0.5555) & (softmax_labels>=0.4444))[0]
idx6 = np.where((softmax_labels<0.4444) & (softmax_labels>=0.3333))[0]
idx7 = np.where((softmax_labels<0.3333) & (softmax_labels>=0.2222))[0]
idx8 = np.where((softmax_labels<0.2222) & (softmax_labels>=0.1111))[0]
idx9 = np.where(softmax_labels<0.1111)[0]
idxs = [idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, idx9]
print(np.where(np.asarray(softmax_labels)))
for h in range(10):
    sub_pths = []
    for idx in (idxs):
        sub_pths.append(image_paths[np.random.choice(idx, size=20, replace=False)].tolist())
    image_grid(sub_pths, f'/ISIC256/grid{h}.jpg')

softmax_labels = np.array(softmax_labels)
# softmax_labels = softmax_labels.flatten()
print(softmax_labels.shape)

df = pd.read_csv(CSV_PATH, header=None)
# labels_df = pd.read_csv(CSV_PATH_mel_class, header=None)
prec_unbiased = np.sum(np.array(df[4]))/len(df)
print("percentage of removed frames ", prec_unbiased)


fig = plt.figure(figsize=(12,12))
fig = plt.hist(softmax_labels.squeeze(),bins=np.linspace(0,1,10))
plt.show()

