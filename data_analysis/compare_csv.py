import os
import pandas as pd
import numpy as np 
from tqdm import tqdm
EXT_DIR = '/ISIC256/ISIC256_ORIGINAL/train_concat.csv'
ISIC_DIR = '/ISIC256/ISIC_pool/malignant_all/ISIC_POOL_ORIGINAL.csv'

ext_all = pd.read_csv(EXT_DIR)
isic_all = pd.read_csv(ISIC_DIR)

ext = pd.DataFrame(ext_all, columns=['image_name','target'])
isic = pd.DataFrame(isic_all, columns=['image_name','target'])
matches = 0

# print(ext.loc[2][0])
print(len(isic.index))

# for i, row_i in isic.iterrows(): # 
for j in tqdm(range(0,len(ext.index))):
    # isic_name = row_i['image_name']
    ext_name = ext.loc[j][0]
    ext_label = ext.loc[j][1]
    if ext_label ==0:
        continue

    # print(isic_label)
    # for j, row_e in ext.iterrows():
    for i in range(0,len(isic.index)):
        # ext_label = row.iloc[0]['label']
        # ext_name = row_e['image_name']
        isic_name = isic.loc[i][0]
        isic_label = isic.loc[i][1]
        
        # print(ext_name)
        if ext_label==1:
            if ext_name.find(isic_name):
                matches = matches+1
                # print(ext_name, isic_name)

print(matches)

        


# print(isic.size)


