import os
from xml.sax.handler import DTDHandler
import pandas as pd
import numpy as np 
from tqdm import tqdm
from collections import Counter
from zipfile import ZipFile
import cv2

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
print("number of total images we have now:", (len(counter_isic + counter_ext))) 
print("number of total new images in isic pool:", len((counter_isic - counter_ext))) 
print("number of non overlapping images:", sum(1 for v in (counter_isic + counter_ext).values() if v == 1))


total_images = (counter_isic + counter_ext) # dict of all images
total_images_list = list(total_images)
overpalpping = {}
for i in range(len(total_images)):
    if total_images[total_images_list[i]] == 2:
        overpalpping[total_images_list[i]] = 1

non_overlapping_ext = (total_images - counter_isic - Counter(overpalpping))
val_list_ext = list(non_overlapping_ext.keys())
print("validation set from train external", len(val_list_ext))
# loop through dict keys 
# look for the images
zip_path = "/ISIC256/ISIC_POOL.zip"

PATH_src1 = "/ISIC256/ISIC256_train/"
PATH_src2 = "/ISIC256/ISIC256_val/"
PATH_src3 = "/ISIC256/ISIC_pool/malignant_all/Cropped_resized/"

PATH_list1 = os.listdir(PATH_src1)
PATH_list2 = os.listdir(PATH_src2)
PATH_list3 = os.listdir(PATH_src3)
# print(PATH_list1)

PATH_dest = "/ISIC256/ISIC_pool/malignant_all/Cropped_resized/valish/"
PATH_dest2 = "/ISIC256/ISIC_pool/malignant_all/Cropped_resized/val/"
list1 = os.listdir(PATH_dest)
list2 = os.listdir(PATH_dest2)

counter_ext = Counter(list1)
counter_isic = Counter(list2)
print("number of overlapping images:", sum(1 for v in (counter_isic + counter_ext).values() if v == 2))

exit()

# processed_img_zip = ZipFile(zip_path, "w")
print("loop started...")
c = 0
for i in range(len(val_list_ext)):
    image_name = str(val_list_ext[i] + '.jpg')
    print(image_name)
    # print(os.path.join( PATH_src1, image_name))
    if image_name in PATH_list1:
        path_in = os.path.join(PATH_src1, image_name)
    elif image_name in PATH_list2:
        path_in = os.path.join(PATH_src2, image_name)
    elif image_name in PATH_list3:
        path_in = os.path.join(PATH_src3, image_name)
    else: 
        continue
    c +=1 
    img = cv2.imread(path_in)
    # print(path_in)
    cv2.imwrite(PATH_dest +image_name , img)
print(c)
# zipped_path = "/ISIC256/ISIC256_ORIGINAL/train.zip"
# folder_path = "/ISIC256/ISIC_pool/malignant_all/Cropped_resized/valish/"
# # print(val_list_ext)
# c = 0
# with ZipFile(zipped_path, "r") as zip_ref:
#     list_of_files = zip_ref.namelist()

#     for i in range(len(list_of_files)):
#         # print(list_of_files[i][-4:])
#         if list_of_files[i][-4:] == '.jpg':

#             list_of_files[i] = list_of_files[i].split('/')[1].split('.')[0]
#             if list_of_files[i] not in val_list_ext:
#                 continue
#             print( list_of_files[i])
#             # print(i)
#             c = c + 1
#             in_bytes = zip_ref.read(list_of_files[i])
#             img = cv2.imdecode(np.frombuffer(in_bytes, np.uint8), cv2.IMREAD_COLOR)
#             print(img)
#             # cv2.imwrite(zipped_path, list_of_files[i]+'.jpg')
#             c = c + 1
#     zip_ref.close()
# print(c)

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


