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

def circle_center(mask): # returns the center of a white circle in a black frame
  centerx = np.argmax(np.sum(mask, axis=1))
  centery = np.argmax(np.sum(mask, axis=0))
  return centerx, centery

# folder = '/ISIC256/ISIC_pool/malignant_all/ORIGINAL_IMAGES'
folder = '/ISIC256/ISIC_pool/malignant_all/resized'

new_folder = '/ISIC256/ISIC_pool/malignant_all/Cropped_resized'
# new_folder = '/ISIC256/ISIC_pool/malignant_all/resized'

classifier = '/ISIC256/ISIC_pool/multiclassifier.csv'

size = 256

######## CSV FILE ###############################################
file_list = os.listdir(folder)
data = open(classifier)
feature = csv.reader(data)

feature_list = []
for row in feature:
        feature_list.append(row)

# print(file_list)



# want to add cropping if multiclassifier determines its a frame

images = []
for idx in tqdm(range(0,10)): #len(file_list))):
    img = cv2.imread(os.path.join(folder,file_list[idx]))
   
    if img is  None:
        continue

    path_out = os.path.join(new_folder, file_list[idx])
    dimensions = img.shape

### CONVERT ALL RECTANGULAR IMAGES ###############################

    # if dimensions[0]!=dimensions[1]:
    #     # print('Need cropping')

    #     if dimensions[0]<dimensions[1]:
    #         crop = dimensions[0]/2
    #         img = img[0:dimensions[0],round(dimensions[1]/2-crop):round(dimensions[1]/2+crop)]
    #     else:
    #         crop = dimensions[1]/2
    #         img = img[round(dimensions[0]/2-crop):round(dimensions[0]/2+crop),0:dimensions[1]]

###### cropping frames ###########################################
    #if multiclassifier determined there was a frame -> remove it

    if int(feature_list[idx][4]) == 1:
        # print('current image ,',file_list[idx], ', row, ', feature_list[idx])

        gray = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
        gray[gray >= 100] = 255
        kernel = np.ones((5,5),np.float32)
        dst = cv2.filter2D(gray,-1,kernel)
        center = circle_center(gray)
        pi = gray[np.average(gray, axis=0).argmax(), :].tolist().index(255)
        piHalve = gray[:, np.average(gray, axis=1).argmax()].tolist().index(np.max(gray[:, np.average(gray, axis=1).argmax()]))
        zeroPi = len(gray[np.average(gray, axis=0).argmax(),:]) - 1 - gray[np.average(gray, axis=0).argmax(),:][::-1].tolist().index(255)
        onePiHalve = len(gray[np.average(gray, axis=1).argmax(),:]) - 1 - gray[np.average(gray, axis=1).argmax(),:][::-1].tolist().index(255)
        
        diameter = max((zeroPi- pi), (onePiHalve - piHalve))
        radius = round(diameter/2)
        x1 = int(pi) #int(pi+(radius-radius*np.sqrt(2)/2))
        y1 = int(piHalve) #int(piHalve+(radius-radius*np.sqrt(2)/2))
        x2 = int(zeroPi) #int(zeroPi - (radius-radius*np.sqrt(2)/2))
        y2 = int(onePiHalve) #int(onePiHalve -(radius-radius*np.sqrt(2)/2))

        img = A.Crop(x_min=x1,
                            y_min=y1, 
                            x_max=x2,
                            y_max=y2).apply(img)


    # img_encoded = cv2.imencode(".png", new_image)[1].tobytes() 
    # nparr = np.frombuffer(img_encoded, np.byte)

    # print('Original Dimensions : ',img.shape)
#     size = int(size)

##### RESIZE ##############################################

    width = size
    height = size
    # print(isinstance(size, str))
    dim = (width, height)
    
    # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_LANCZOS4)
    
    # WRITE TO FOLDER
    cv2.imwrite(path_out, img)
