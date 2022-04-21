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

# construct cmap
colors = ["#ddc7af","#d3a293", "#ba707e", "#995374", "#4f3159", "#231f37"]
sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
my_cmap = ListedColormap(sns.color_palette(colors).as_hex())

fig = plt.figure(figsize=(12,12))
ax = fig.add_axes([0,0,1,1])
labels = ['hair','black frame','ruler','other']
numbers = [hair, black_frame,ruler_mark,other]
plot = ax.bar(labels,numbers,color =["#ddc7af","#d3a293", "#ba707e", "#995374"]) 
# ax.bar_label(ax.containers[0],size=16)
ppp=[]
ppp = [round(x / len(feature_list)*100) for x in numbers]

print(ppp)
pp=0
for p in plot:
    height = p.get_height()
    # print(p)
    
    if pp == 0:
        ax.text(x=p.get_x() + p.get_width() / 2, y=height+.10,
        s="32%".format(height),
        ha='center', size=16)
    if pp == 1:
        ax.text(x=p.get_x() + p.get_width() / 2, y=height+.10,
        s="38%".format(height),
        ha='center', size=16)
    if pp == 2:
        ax.text(x=p.get_x() + p.get_width() / 2, y=height+.10,
        s="9%".format(height),
        ha='center', size=16)
    if pp == 3:
        ax.text(x=p.get_x() + p.get_width() / 2, y=height+.10,
        s="12%".format(height),
        ha='center', size=16)
    pp +=1

ax.set_title('Feature distribution', fontsize=20)
# plt.xlabel('', fontsize=18)
plt.ylabel('# Images', fontsize=16)
plt.xticks(fontsize= 16)
plt.yticks(fontsize= 16)
plt.show()


# # construct cmap
# colors = ["#ddc7af","#d3a293", "#ba707e", "#995374", "#4f3159", "#231f37"]
# sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
# my_cmap = ListedColormap(sns.color_palette(colors).as_hex())

# N = 500
# data1 = np.random.randn(N)
# data2 = np.random.randn(N)
# colors = np.linspace(0,1,N)
# fig = plt.figure(figsize=(12,12))
# plt.scatter(data1, data2, c=colors, cmap=my_cmap)
# plt.colorbar()
# plt.show()

# fig.savefig('test.jpg', bbox_inches='tight')


### create train and val ###############################################

# train_folder= '/ISIC256/ISIC_pool/malignant_all/Cropped_resized/train'
# val_folder = '/ISIC256/ISIC_pool/malignant_all/Cropped_resized/val'

# folder ='/ISIC256/ISIC_pool/malignant_all/Cropped_resized'

# file_list = os.listdir(folder)
# print(len(file_list))
# train_part = int(0.8*len(file_list))
# print(train_part)


# for idx in tqdm(range(0,len(file_list))):
#     img = cv2.imread(os.path.join(folder,file_list[idx]))
#     if img is  None:
#         continue

#     path_out_train = os.path.join(train_folder, file_list[idx])
#     path_out_val = os.path.join(val_folder, file_list[idx])

#     if idx <= train_part:
#         cv2.imwrite(path_out_train, img)
#     else :
#         cv2.imwrite(path_out_val, img)


