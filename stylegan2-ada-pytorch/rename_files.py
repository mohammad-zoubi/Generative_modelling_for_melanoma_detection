import os
from tqdm import tqdm
mal_imgs_path = '/ISIC256/models/00001-ISIC256_malignant-auto2-bgcfnc-resumecustom/generated_imgs_ISIC_mal/imgs'
ben_imgs_path = '/ISIC256/models/00000-ISIC256_benign-auto1-bgcfnc/generated_imgs_ISIC_ben/imgs'

mal_list = os.listdir(mal_imgs_path)
os.chdir(mal_imgs_path)

for i in tqdm(range(len(mal_list))):
    if os.path.splitext(mal_list[i])[1] == '.jpg':
        os.rename(mal_list[i], mal_list[i].split('.')[0].split('_')[0] + 'm' + '.jpg')


ben_list = os.listdir(ben_imgs_path)
os.chdir(ben_imgs_path)

for i in tqdm(range(len(ben_list))):
    if os.path.splitext(ben_list[i])[1] == '.jpg':
        os.rename(ben_list[i], ben_list[i].split('.')[0].split('_')[0] + 'b' + '.jpg')