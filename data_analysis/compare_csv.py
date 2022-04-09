import os
from xml.sax.handler import DTDHandler
import pandas as pd
import numpy as np 
from tqdm import tqdm
from collections import Counter

EXT_DIR = '/ISIC256/ISIC256_ORIGINAL/train_concat.csv'
ISIC_DIR = '/ISIC256/ISIC_pool/malignant_all/ISIC_POOL_ORIGINAL.csv'

ext_all = pd.read_csv(EXT_DIR)
isic_all = pd.read_csv(ISIC_DIR)

# ext = pd.DataFrame(ext_all, columns=['image_name','target'])
# isic = pd.DataFrame(isic_all, columns=['image_name','target'])
matches = 0

ext_list = []
isic_list = []

# print(ext.loc[2][0])
# print(len(isic.index))


ext_df = pd.read_csv(EXT_DIR, header=None)
isic_df = pd.read_csv(ISIC_DIR, header=None)


tmp = np.array(np.asarray(ext_df[5])[1:], dtype=int)
indices = np.where(tmp == 1)[0]
ext_list.append(ext_df.iloc[indices][0][1:].tolist()) # list of image names, ISIC_ext

for i in range(len(ext_list[0])):
    ext_list[0][i] = ext_list[0][i][:12]

isic_list.append(isic_df[0][1:].tolist())

# get two dicts with frequencies 
counter_ext = Counter(ext_list[0])
counter_isic = Counter(isic_list[0])

# subtract frequencies to remove similar entries
# we have 570 more images in ext :-( 
print("number of external images:", len(counter_ext))
print("number of isic pool images:", len(counter_isic))
print("number of total images we have now:", len((counter_isic + counter_ext))) 
print("number of total new images in isic pool:", len((counter_ext - counter_isic))) 
print("number of non overlapping images:", sum(1 for v in (counter_isic + counter_ext).values() if v == 1))


# # for i, row_i in isic.iterrows(): # 
# for j in tqdm(range(0,len(ext.index))):
#     # isic_name = row_i['image_name']
#     ext_name = ext.loc[j][0]
#     ext_label = ext.loc[j][1]
#     if ext_label ==0:
#         continue
    
#     # print(isic_label)
#     # for j, row_e in ext.iterrows():
#     for i in range(0,len(isic.index)):
#         # ext_label = row.iloc[0]['label']
#         # ext_name = row_e['image_name']
#         isic_name = isic.loc[i][0]
#         isic_label = isic.loc[i][1]
        
#         # print(ext_name)
#         if ext_label==1:
#             if ext_name.find(isic_name):
#                 matches = matches+1
#                 # print(ext_name, isic_name)

# print(matches)

        


# print(isic.size)


