''' Looking at synth100k images directories
    Here we have labels of images that are fed into melanoma_real classifier and we want to see how
    different softmax differ in image quality'''

import pandas as pd
import numpy as np
import os, sys
import matplotlib.pyplot as plt


CSV_PATH = "/ISIC256/ISIC256_ORIGINAL/synth100k_mal/shifted_imgs_dir/shifted_frames.csv"
# CSV_PATH_mel_class = '/ISIC256/ISIC256_ORIGINAL/synth100k_mal/'

labels_biases = ["label_hair_short", "label_hair_medium", "label_black_frame", "label_ruler_mark", "label_other"] 

softmax_labels = []
image_paths = []

for i in range(1,6):
    path_to_csv = f'/ISIC256/ISIC256_ORIGINAL/synth100k_mal/synth100k{i}_labels.csv'
    df = pd.read_csv(path_to_csv)
    tmp = np.asarray(df['predicted_labels'])
    images_names = df.image_name.tolist()
    # print(images_names)
    softmax_labels.append(tmp)
    parent_path = f'/ISIC256/ISIC256_ORIGINAL/synth100k_mal/img_dirs/imgs_dirs_mal/imgs{i}/'
    
    image_paths.append([parent_path + s for s in images_names])

image_paths = [item for sublist in image_paths for item in sublist]

idx1 = np.where(np.asarray(softmax_labels)>=0.8888)[0]
idx2 = np.where((np.asarray(softmax_labels)<0.8888) & (np.asarray(softmax_labels)>=0.7777))[0]
idx3 = np.where((np.asarray(softmax_labels)<0.7777) & (np.asarray(softmax_labels)>=0.6666))[0]
idx4 = np.where((np.asarray(softmax_labels)<0.6666) & (np.asarray(softmax_labels)>=0.5555))[0]
idx5 = np.where((np.asarray(softmax_labels)<0.5555) & (np.asarray(softmax_labels)>=0.4444))[0]
idx6 = np.where((np.asarray(softmax_labels)<0.4444) & (np.asarray(softmax_labels)>=0.3333))[0]
idx7 = np.where((np.asarray(softmax_labels)<0.3333) & (np.asarray(softmax_labels)>=0.2222))[0]
idx8 = np.where((np.asarray(softmax_labels)<0.2222) & (np.asarray(softmax_labels)>=0.1111))[0]
idx9 = np.where(np.asarray(softmax_labels)<0.1111)[0]

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

