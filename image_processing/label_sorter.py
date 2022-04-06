'''sorts labels after given a csv and image directory'''

import pandas as pd
import numpy as np
import sys, os

# Input: csv labels
#       image path

# Output: move images with target label to another folder 

source_folder = "/data/synth100k_mal/imgs_dirs"
def list_of_img_pths(source_folder): # returns a list of paths for images in a directorty with subdirectories
    file_path_list = []
    for root,_,imgs in os.walk(source_folder):
        imgs = [ f for f in imgs if os.path.splitext(f)[1] in ('.png', '.jpg') ]
        for filename in imgs:
            file_path_list.append(os.path.join(root, filename))
    return file_path_list


list_of_img_pths(source_folder, 1, 1)
