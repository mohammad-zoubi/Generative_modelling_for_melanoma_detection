from PIL import Image
import numpy as np
import os
from tqdm import tqdm

def vector_to_mat(file_path, save): #save argument is Boolean
    vector_files = [os.path.join(file_path,vec)
                for vec in os.listdir(file_path)
                if (vec.endswith(".txt"))]
    print("writing vectors to matrix... ")
    w_matrix = np.zeros((len(vector_files), 512))
    for idx in tqdm(range(len(vector_files))):
        w_vector = np.transpose( np.loadtxt(vector_files[idx])[:, np.newaxis] ) 
        w_matrix[idx,:] = w_vector
    if save: 
        np.save('/data/w_matrix.npy', w_matrix)
    else:  
        return w_matrix

def npy_to_tsv(w_matrix):
    np.savetxt("thousand_frames.tsv", w_matrix, delimiter="\t")

def sprite(image_folder_path): # Creates a large montage of images 
    image_files = [os.path.join(image_folder_path,img)
                for img in os.listdir(image_folder_path)
                if (img.endswith(".jpg") or img.endswith(".png"))]

    images = [Image.open(os.path.join(image_folder_path, filename)).resize((256,256)) for filename in tqdm(image_files)]
    image_width, image_height = images[0].size
    one_square_size = int(np.ceil(np.sqrt(len(images))))
    master_width = (image_width * one_square_size) 
    master_height = image_height * one_square_size
    spriteimage = Image.new(
        mode='RGBA',
        size=(master_width, master_height),
        color=(0,0,0,0))  # fully transparent
    for count, image in tqdm(enumerate(images)):
        div, mod = divmod(count,one_square_size)
        h_loc = image_width*div
        w_loc = image_width*mod    
        spriteimage.paste(image,(w_loc,h_loc))
    spriteimage.convert("RGB").save('sprite.jpg', transparency=0) # ADD PATH AND CHANGE NAME OF SPRITE IMAGE 

# file_path = "/data/generated_imgs_frames/latent_code/"
# w_matrix = vector_to_mat(file_path, save=False)
# print(w_matrix)
# npy_to_tsv(w_matrix)
# sprite('/data/generated_imgs_frames/imgs/')