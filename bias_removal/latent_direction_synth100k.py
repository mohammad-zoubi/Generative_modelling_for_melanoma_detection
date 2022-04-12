''' Use binary classifier to construct boundaries between biases '''
# This will be used to generate images for fool the doctor assignment
# 
# Layout
# place all 100k vectors in a matrix [n_samples, n_features] 
# get their corresponding bias labels, each in its own vector. Keep it ordered
# Train a binary classifer for each bias
# Add these classifiers as checkpoints when generating images, make a modified generate.py script for this
# Compare multiclassifier to the binary classifier

################################################################################################

''' Compare shifted images
    Goal: take images with frames and move them in a direction to remove frames then compare how many are still melanomas '''

import os, sys
import pandas as pd
import numpy as np

# import csv of bias
CSV_PATH = "/ISIC256/ISIC256_ORIGINAL/synth100k_mal/frames_synth100k.csv"
frame_df = pd.read_csv(CSV_PATH, header=None, index_col=0)
print(frame_df)
tmp_list = []
tmp_list.append(frame_df[1].tolist())

# print(tmp_list[0])
seed_name = []
seed_number = []
for i in range(1,len(tmp_list[0][:100])):
    seed_name.append(tmp_list[0][i][:10])
    seed_number.append(tmp_list[0][i][4:10])
tmp = ((str(seed_number).replace('[','').replace(']','').replace(' ','')))
DEST_PATH = "/ISIC256/ISIC256_ORIGINAL/synth100k_mal/shifted_imgs_dir/shifted_imgs1/"
# take 1 image
exit()
# .replace('[','').replace(']','').replace(' ',',')
execute = "python3 apply_factor.py " 
execute = execute + "/ISIC256/project_scripts/GITHUB_REPO/Generative_modelling_for_melanoma_detection/SeFa_matrices/mal_uncond_eig_vec.pt"
execute = execute + " --index=" + "6"
execute = execute + " --truncation=" + "1"
execute = execute + " --degree=" + "5"
execute = execute + " --ckpt=" + "/ISIC256/models/00001-ISIC256_malignant-auto2-bgcfnc-resumecustom/network-snapshot-009400.pkl"
execute = execute + " --seeds=" + tmp
execute = execute + " --output=" + "/ISIC256/ISIC256_ORIGINAL/synth100k_mal/experiments/"

os.system(execute)