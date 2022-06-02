import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import os 

""" 
CSV_DIR = Path('/home/Data/melanoma_external_256/')
df = pd.read_csv(CSV_DIR/'train_concat.csv')
train_split, valid_split = train_test_split (df, stratify=df.target, test_size = 0.20, random_state=42)
train_df=pd.DataFrame(train_split)

labels_list = []
for n in range(len(train_df)):
    labels_list.append([train_df.iloc[n].image_name,int(train_df.iloc[n].target)])
"""


labels_list = []
# input_images = [str(f) for f in sorted(Path('/workspace/melanoma_isic_dataset/all_melanoma/SAM_Dataset').rglob('*')) if os.path.isfile(f)]
# input_images = [str(f) for f in sorted(Path('/ISIC256/train_set_synth/imgs/').rglob('*.jpg')) if os.path.isfile(f)]
df = pd.read_csv('/ISIC256/ISIC_orig_trainset.csv')
list_of_pths = df.image_name.tolist()
list_of_targets = df.target.tolist()
# for img_path in input_images:
#     label = img_path.split('.')[0][-1]
#     if label == 'b':
#         labels_list.append([img_path, '0'])
#     else:
#         labels_list.append([img_path, '1'])

for i in range(len(list_of_pths)):
    image_name = list_of_pths[i].split('/')[-1]
    label = list_of_targets[i]
    if label == 1:
        labels_list.append([image_name, '1'])
    else:
        labels_list.append([image_name, '0'])
labels_list_dict = { "labels" : labels_list}
with open("/ISIC256/train_ISIC256_orig/imgs/dataset.json", "w") as outfile:
    json.dump(labels_list_dict, outfile)
