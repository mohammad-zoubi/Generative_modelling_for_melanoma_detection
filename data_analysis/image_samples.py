from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
import PIL


def image_grid(PATH_list, save_path): # input: 2d list of paths
    grid = []
    for row in tqdm(PATH_list):
        # print(Image.open(row[0]))
        # print()
        images = [Image.open(x) for x in row]
        # print(images)
        imgs_comb = np.hstack( (np.asarray(i) for i in images ) )
        grid.append(imgs_comb)
    imgs_comb = np.vstack( (np.asarray(i) for i in grid ) )
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    imgs_comb.save(save_path)
