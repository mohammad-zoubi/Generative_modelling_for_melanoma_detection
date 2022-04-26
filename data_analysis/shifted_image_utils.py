# import image_similarity_measures
# from image_similarity_measures.quality_metrics import rmse, psnr
import pandas as pd
import os, sys
from pathlib import Path

def seed_list():
    CSV_PATH = "/ISIC256/ISIC256_ORIGINAL/synth100k_mal/frames_synth100k.csv"
    frame_df = pd.read_csv(CSV_PATH, header=None, index_col=0)
    print(frame_df)
    tmp_list = []
    tmp_list.append(frame_df[1].tolist())

    # print(tmp_list[0])
    seed_name = []
    seed_number = []
    for i in range(1,len(tmp_list[0][:10])):
        seed_name.append(tmp_list[0][i][:10])
        seed_number.append(int(tmp_list[0][i][4:10]))
    return seed_number

shifted_list = seed_list()
print(shifted_list)
syn_data_path_mal_file = "/ISIC256/ISIC256_ORIGINAL/synth100k_mal/imgs_dirs"
shifted_data_path_mal_file = "/ISIC256/ISIC256_ORIGINAL/synth100k_mal/shifted_imgs"
input_images = [str(f) for f in sorted(Path(syn_data_path_mal_file).rglob('*.jpg')) if os.path.isfile(f)]

df = pd.DataFrame()
# for i in range(len(input_images[:10])):
#     if int(input_images[i][-14:-8]) in shifted_list:
#         print(input_images[i][-14:-8])
