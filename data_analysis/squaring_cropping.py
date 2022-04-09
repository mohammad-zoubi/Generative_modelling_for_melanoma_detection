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

def circle_center(mask): # returns the center of a white circle in a black frame
  centerx = np.argmax(np.sum(mask, axis=1))
  centery = np.argmax(np.sum(mask, axis=0))
  return centerx, centery

folder = '/ISIC256/ISIC_pool/malignant_all/ORIGINAL_IMAGES'
new_folder = '/ISIC256/ISIC_pool/malignant_all/Cropped_resized'

size = 256

file_list = os.listdir(folder)
# print(file_list)

### CONVERT ALL RECTANGULAR IMAGES ###############################

# want to add cropping if multiclassifier determines its a frame

images = []
for idx in range(0,10): #len(file_list)):
    img = cv2.imread(os.path.join(folder,file_list[idx]))
   
    if img is  None:
        continue

    path_out = os.path.join(new_folder, file_list[idx])
    dimensions = img.shape

    if dimensions[0]!=dimensions[1]:
        # print('Need cropping')

        if dimensions[0]<dimensions[1]:
            crop = dimensions[0]/2
            img = img[0:dimensions[0],round(dimensions[1]/2-crop):round(dimensions[1]/2+crop)]
        else:
            crop = dimensions[1]/2
            img = img[round(dimensions[0]/2-crop):round(dimensions[0]/2+crop),0:dimensions[1]]



    # gray = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
    # gray[gray >= 100] = 255
    # kernel = np.ones((5,5),np.float32)
    # dst = cv2.filter2D(gray,-1,kernel)
    # center = circle_center(gray)
    # pi = gray[np.average(gray, axis=0).argmax(), :].tolist().index(255)
    # piHalve = gray[:, np.average(gray, axis=1).argmax()].tolist().index(np.max(gray[:, np.average(gray, axis=1).argmax()]))
    # zeroPi = len(gray[np.average(gray, axis=0).argmax(),:]) - 1 - gray[np.average(gray, axis=0).argmax(),:][::-1].tolist().index(255)
    # onePiHalve = len(gray[np.average(gray, axis=1).argmax(),:]) - 1 - gray[np.average(gray, axis=1).argmax(),:][::-1].tolist().index(255)
    
    # diameter = max((zeroPi- pi), (onePiHalve - piHalve))
    # radius = round(diameter/2)
    # x1 = int(pi+(radius-radius*np.sqrt(2)/2))
    # y1 = int(piHalve+(radius-radius*np.sqrt(2)/2))
    # x2 = int(zeroPi - (radius-radius*np.sqrt(2)/2))
    # y2 = int(onePiHalve -(radius-radius*np.sqrt(2)/2))

    # new_image = A.Crop(x_min=x1,
    #                     y_min=y1, 
    #                     x_max=x2,
    #                     y_max=y2).apply(img)
    # img_encoded = cv2.imencode(".png", new_image)[1].tobytes() 
    # nparr = np.frombuffer(img_encoded, np.byte)

    # print('Original Dimensions : ',img.shape)
#     size = int(size)
    width = size
    height = size
    # print(isinstance(size, str))
    dim = (width, height)
    
    # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_LANCZOS4)
    
    # WRITE TO FOLDER
    cv2.imwrite(path_out, img)


        
    



# image_path = '/SAM256/SAM_ORIGINAL_TV/Full_size/'
# path_out = '/SAM256/SAM_ORIGINAL_TV/256_full_cropped'
# size = 256
# ###################################
# # ADD ext to json file 

# def circle_center(mask): # returns the center of a white circle in a black frame
#   centerx = np.argmax(np.sum(mask, axis=1))
#   centery = np.argmax(np.sum(mask, axis=0))
#   return centerx, centery

# with open('dataset.json', 'r') as file:
#     # print("json.load(file): ", json.load(file))
#     labels = json.load(file)["labels"]
#     for img in labels:
#         img[0] = img[0]+'.png'
# labels_list_dict = { "labels" : labels}

# with open("/SAM256/SAM_ORIGINAL_TV/dataset.json", "w") as outfile:
#     json.dump(labels_list_dict, outfile)

# file_list = os.listdir(image_path)
# print(file_list)
###################################

# list_of_files = zip_ref.namelist()
# Loop through the list of files 
# for idx in range(1,len(file_list)): # list of files has all the files in zip including the folders
#     # Get the extension of the the given file and if it is an image, excute what is inside the if statement 

#     path_in = os.path.join(image_path, file_list[idx]) 
#     # if os.path.isfile(path_in) == False:
#     #     continue
#     img = cv2.imread(path_in)



#     # in_bytes = zip_ref.read(list_of_files[idx])
#     # img = cv2.imdecode(np.frombuffer(in_bytes, np.uint8), cv2.IMREAD_COLOR)
#     # img = hair_removal(img, kern=10, intensity=5)

#     # img = hair_removal(img, kern=10, intensity=5) # is it really needed ???
#     # get mask of an image -> get center of a circle -> get square indices 
#     # cut the square out 
#     # profit ??
#     # mask = mask_extractor(img, kern, can, pad)
#     # cx, cy = circle_center(mask)
#     # x1, y1, x2, y2 = circle_to_square_crop(mask, cx, cy, shift)

#     gray = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
#     gray[gray >= 100] = 255
#     kernel = np.ones((5,5),np.float32)
#     dst = cv2.filter2D(gray,-1,kernel)
#     center = circle_center(gray)
#     pi = gray[np.average(gray, axis=0).argmax(), :].tolist().index(255)
#     piHalve = gray[:, np.average(gray, axis=1).argmax()].tolist().index(np.max(gray[:, np.average(gray, axis=1).argmax()]))
#     zeroPi = len(gray[np.average(gray, axis=0).argmax(),:]) - 1 - gray[np.average(gray, axis=0).argmax(),:][::-1].tolist().index(255)
#     onePiHalve = len(gray[np.average(gray, axis=1).argmax(),:]) - 1 - gray[np.average(gray, axis=1).argmax(),:][::-1].tolist().index(255)
    
#     diameter = max((zeroPi- pi), (onePiHalve - piHalve))
#     radius = round(diameter/2)
#     x1 = int(pi+(radius-radius*np.sqrt(2)/2))
#     y1 = int(piHalve+(radius-radius*np.sqrt(2)/2))
#     x2 = int(zeroPi - (radius-radius*np.sqrt(2)/2))
#     y2 = int(onePiHalve -(radius-radius*np.sqrt(2)/2))

#     new_image = A.Crop(x_min=x1,
#                         y_min=y1, 
#                         x_max=x2,
#                         y_max=y2).apply(img)
#     # img_encoded = cv2.imencode(".png", new_image)[1].tobytes() 
#     # nparr = np.frombuffer(img_encoded, np.byte)

#      # print('Original Dimensions : ',img.shape)
#     size = int(size)
#     width = size
#     height = size
#     # print(isinstance(size, str))
#     dim = (width, height)
    
#     # resize image
#     resized = cv2.resize(new_image, dim, interpolation = cv2.INTER_LANCZOS4)
    
#     # print('Resized Dimensions : ',resized.shape)
    
#     # plt.imshow(resized)
#     # plt.show()
    
#     # cv2_imshow(img)
#     if size ==256:
#         path_out = os.path.join('/SAM256/SAM_ORIGINAL_TV/256_full_cropped', file_list[idx])

#     # WRITE TO FOLDER
#     cv2.imwrite(path_out, resized)


    # image_name = os.path.splitext(list_of_files[idx])[0].split('/')[2]
    # processed_img_in_situ_zip.writestr(image_name+".png", nparr)


########################################################################

# image_path = "/ISIC256/ISIC_pool/malignant_all/ORIGINAL_IMAGES"
# file_list = os.listdir(image_path)

# for m in range(len(file_list)):
    
#     path_in = os.path.join(image_path, file_list[m]) 
#     # if os.path.isfile(path_in) == False:
#     #     continue
#     img = cv2.imread(path_in)
    
#     # print('Original Dimensions : ',img.shape)
#     # size = int(size)
#     # width = size
#     # height = size
#     # # print(isinstance(size, str))
#     # dim = (width, height)
    
#     # resize image
#     resized = cv2.resize(img, (256, 256), interpolation = cv2.INTER_LANCZOS4)
    
#     # print('Resized Dimensions : ',resized.shape)
    
#     # cv2_imshow(resized)
#     # cv2_imshow(img)
#     # if size ==256:
#     path_out = os.path.join("/ISIC256/ISIC_pool/malignant_all/RESIZED_IMAGES/", file_list[m])
#     # elif size == 512:
#     # path_out = os.path.join('//content/drive/MyDrive/thesis/ELLA_Eivor1.0_DERM_dataset.zip (Unzipped Files)/Sahlgrenska/processed_512', "{:04d}".format(m)+ '_derm_01.jpeg')
#     # else:
#     # print("wrong dimensions")
#     # sys.exit()

#     cv2.imwrite(path_out, resized)