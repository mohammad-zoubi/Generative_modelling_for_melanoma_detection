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

classifier = '/ISIC256/ISIC_pool/malignant_all/multiclassifier_cropped_resized.csv'

data = open(classifier)
feature = csv.reader(data)

hair_dense = 0
hair_short = 0
hair_medium = 0
black_frame = 0
ruler_mark = 0
other = 0

feature_list = []
row_1 = ['image_name','dense hair','short hair','medium hair','black frame','ruler','other']
feature_list.append(row_1)
for row in feature:
        feature_list.append(row)

print(len(feature_list))

# sns.histplot(data=feature_list)

for idx in range(1, len(feature_list)):
    hair_dense += int(feature_list[idx][1])
    hair_short += int(feature_list[idx][2])
    hair_medium += int(feature_list[idx][3])
    black_frame += int(feature_list[idx][4])
    ruler_mark += int(feature_list[idx][5])
    other += int(feature_list[idx][6])

hair = hair_dense+hair_medium+hair_short

print(feature_list[0])

fig = plt.figure(figsize=(12,12))
ax = fig.add_axes([0,0,1,1])
labels = ['hair','black frame','ruler','other']
numbers = [hair, black_frame,ruler_mark,other]
ax.bar(labels,numbers)
plt.show()
# fig.savefig('test.jpg')

# fig = plt.figure(figsize= (12,12))
# fig = sns.heatmap(x, cmap=cmap)
# fig = fig.get_figure()
# fig.savefig('test.jpg')
# 
# fig = plt.figure()

# N = 5
# menMeans = (20, 35, 30, 35, 27)
# womenMeans = (25, 32, 34, 20, 25)
# ind = np.arange(N) # the x locations for the groups
# width = 0.35
# fig = plt.figure()
# ax = fig.add_axes([-0.1,-0.1,1,1])
# ax.bar(ind, menMeans, width, color='r')
# ax.bar(ind, womenMeans, width,bottom=menMeans, color='b')
# ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
# ax.set_xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
# ax.set_yticks(np.arange(0, 81, 10))
# ax.legend(labels=['Men', 'Women'])
# plt.show()
fig.savefig('test.jpg', bbox_inches='tight')

### create train and val ###############################################

train_folder= '/ISIC256/ISIC_pool/malignant_all/Cropped_resized/train'
val_folder = '/ISIC256/ISIC_pool/malignant_all/Cropped_resized/val'

folder ='/ISIC256/ISIC_pool/malignant_all/Cropped_resized'

file_list = os.listdir(folder)
print(len(file_list))
train_part = int(0.8*len(file_list))
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


