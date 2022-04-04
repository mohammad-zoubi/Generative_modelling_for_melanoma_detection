import os
import zipfile 
import pandas as pd
import cv2
import numpy as np

# path_csv = '/ISIC256/ISIC256_ORIGINAL/train_concat.csv'
# path_img = '/ISIC256/ISIC256_ORIGINAL/train.zip'
# # path_img = '/ISIC256/ISIC256_ORIGINAL/ISIC256_benign.zip'
# path_zip = '/ISIC256/ISIC256_ORIGINAL/'

# zip_name_benign = 'ISIC256_benign.zip'
# zip_name_malignant = 'ISIC256_malignant.zip'

# zip_malignant = zipfile.ZipFile(os.path.join(path_zip,zip_name_malignant), "w")
# zip_benign = zipfile.ZipFile(os.path.join(path_zip,zip_name_benign), "w")

# anno = pd.read_csv(path_csv)
# df = pd.DataFrame(anno, columns=['image_name','target'])
# print(len(df))

# with zipfile.ZipFile(path_img, mode='r') as zip_ref:
#     # Get list of files names in zip
#     list_of_files = zip_ref.namelist()
#     # print(list_of_files)


#     for idx in range(len(list_of_files)):
#         img = zip_ref.read(list_of_files[idx])
#         if os.path.splitext(list_of_files[idx])[1] == '.jpg':
#             img_name = os.path.splitext(list_of_files[idx])[0].split('/')[1]
#             # print(os.path.splitext(list_of_files[idx]))
#             # print("img", img)
#             # print("img_name", img_name)
#             # df.loc[img_name]
#             row = df.loc[df['image_name']==img_name]
#             label = row.iloc[0]['target']
#             if label == 0:
#                 zip_benign.writestr(img_name+".jpg", img)
#             else :
#                 zip_malignant.writestr(img_name+".jpg", img)
#             # print(row.iloc[0]['target'])
#             # print(row)


          


#     print(len(zip_benign.namelist()))
#     print(len(zip_malignant.namelist()))

#     zip_malignant.close()
#     zip_benign.close()


##################################################################

## FOR SAM DATA
path_csv = '/SAM256/SAMdata2.csv'
path_img = '/SAM256/SAM_ORIGINAL_TV/256_full_cropped/'
# path_img = '/ISIC256/ISIC256_ORIGINAL/ISIC256_benign.zip'
path_zip = '/SAM256/'

zip_name_inv = 'SAM256_inv_full_cropped.zip'
zip_name_insitu = 'SAM256_insitu_full_cropped.zip'

zip_insitu = zipfile.ZipFile(os.path.join(path_zip,zip_name_insitu), "w")
zip_inv = zipfile.ZipFile(os.path.join(path_zip,zip_name_inv), "w")

anno = pd.read_csv(path_csv)
df = pd.DataFrame(anno, columns=['name','label'])

print(df)

# with zipfile.ZipFile(path_img, mode='r') as zip_ref:
    # Get list of files names in zip
    # list_of_files = zip_ref.namelist()
    # print(list_of_files)

list_of_files = os.listdir(path_img)
# print(list_of_files[1].split('.')[0])

for idx in range(len(list_of_files)):
    new_image = cv2.imread(path_img+list_of_files[idx])

    if os.path.splitext(list_of_files[idx])[1] == '.png':
        img_name = list_of_files[idx].split('.')[0]
        # print(os.path.splitext(list_of_files[idx]))
        # print("img", img)
        # print("img_name", img_name)
        # df.loc[img_name]

        img_encoded = cv2.imencode(".png", new_image)[1].tobytes() 
        nparr = np.frombuffer(img_encoded, np.byte)
        row = df.loc[df['name']==img_name]
        label = row.iloc[0]['label']
        if label == 0:
            zip_insitu.writestr(img_name+".jpg", nparr)
        else :
            zip_inv.writestr(img_name+".jpg", nparr)
        # print(row.iloc[0]['target'])
        # print(row)


          


print(len(zip_inv.namelist()))
print(len(zip_insitu.namelist()))

zip_inv.close()
zip_insitu.close()


