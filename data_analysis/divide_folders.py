import pandas as pd
import numpy as np
import os, os.path
from tqdm import tqdm 

### create train and val ###############################################

# train_folder= '/ISIC256/ISIC_pool/malignant_all/Cropped_resized/test'
train_folder= '/ISIC256/train_ISIC256_orig/imgs/'
# val_folder = '/ISIC256/ISIC_pool/malignant_all/Cropped_resized/validation'
val_folder = '/ISIC256/train_ISIC256_orig/val_set/'
df = pd.read_csv('/ISIC256/real_val.csv')
val_image_list = df.image_name.tolist()
all_images_list = os.listdir(train_folder)
# folder ='/ISIC256/ISIC_pool/malignant_all/Cropped_resized/all_non_train'

# file_list = os.listdir(folder)
# print(len(file_list))
# train_part = int(0.5*len(file_list))
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
dg = 0
for i in tqdm(range(len(all_images_list))):
    ext = os.path.splitext(all_images_list[i])[-1]
    if ext == '.jpg':
        if all_images_list[i] in val_image_list:
            os.replace(train_folder+all_images_list[i], val_folder+all_images_list[i])
