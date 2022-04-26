'''extracts a zip file, because why not'''

import sys, os
from zipfile import ZipFile
import cv2
import numpy as np

PATH_ZIP = '/ISIC256/ISIC256_ORIGINAL/train.zip'
DEST = '/ISIC256/train_set_trainzip/imgs/'

with ZipFile(PATH_ZIP, "r") as zip_ref:    
    # Get list of files names in zip
    list_of_files = zip_ref.namelist()
    for idx in range(len(list_of_files)): # list of files has all the files in zip includin the folders
    # for idx in range(10): # list of files has all the files in zip includin the folders
        ext = os.path.splitext(list_of_files[idx])[-1]
        image_name = list_of_files[idx].split('.')[0].split('/')[1]+'.jpg'
        if ext == ".jpg":
            in_bytes = zip_ref.read(list_of_files[idx])
            img = cv2.imdecode(np.frombuffer(in_bytes, np.uint8), cv2.IMREAD_COLOR)
            # img = hair_removal(img, kern=10, intensity=5)
            cv2.imwrite(DEST+image_name, img)