'''sorts labels after given a csv and image directory'''

from email import header
import pandas as pd
import numpy as np
import sys, os

# Input: csv labels
#       image path

# Output: move images with target label to another folder 

source_folder = "/data/synth100k_mal/imgs_dirs"
csv_file = "/data/synth100k_mal/synth100k1_labels.csv"
def list_of_img_pths(source_folder): # returns a list of paths for images in a directorty with subdirectories
    file_path_list = []
    for root,_,imgs in os.walk(source_folder):
        imgs = [ f for f in imgs if os.path.splitext(f)[1] in ('.png', '.jpg') ]
        for filename in imgs:
            file_path_list.append(os.path.join(root, filename))
    return file_path_list

def filter_list(file_path_list, csv_file):
    csv_file = pd.read_csv(csv_file, index_col=0)
    print(csv_file)
    for img in file_path_list:
        file_name = img.split('/')[-1]
        # print(file_name)
        # print(csv_file['image_name'].tolist())
        idx = csv_file.loc[csv_file["image_name"] == file_name].index.values
        # print("index is",idx)
        # if csv_file.index:
        #     print(idx)

def move_imgs(file_path_list, target_folder):
    pass

# file_path_list = list_of_img_pths(source_folder)
# filter_list(file_path_list, csv_file)
# def csv_filter(path_to_csv): # returns csv with all images that has frames
bias_df = pd.DataFrame(columns=["image_name"])
tmp_list = []
for i in range(1,6):
    path_to_csv = f'/data/synth100k_mal/synth100k_anno{i}.csv'
    df = pd.read_csv(path_to_csv, header=None)
    tmp = np.asarray(df[4])
    indices = np.where(tmp == 1)[0]
    # print(type(df.iloc[indices][0].tolist()))
    tmp_list.append(df.iloc[indices][0].tolist())
tmp_list = [j for sub in tmp_list for j in sub]
seed_number = []
for i in range(len(tmp_list)):
    seed_number.append(tmp_list[i][4:10])
print(np.asarray(seed_number, dtype=int))
print(len(tmp_list))
bias_df['image_name'] = tmp_list
# bias_df = pd.concat([bias_df['image_name'],df.iloc[indices][0]], axis=0)


print(bias_df)
def script_runner(list_of_seeds):
    pass