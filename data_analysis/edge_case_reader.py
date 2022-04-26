''' Input: file with ground truth, image name and predicted value
    Output: copies edge case images to another folder '''

import shutil
import pandas as pd
import numpy as np


PATH_TSV = '/ISIC256/00000/default/metadata.tsv'
DEST = '/ISIC256/edge_cases/imgs/'
df = pd.read_csv(PATH_TSV, delimiter='\t')

gt = np.asarray([int(la.split('_')[0]) for la in df.label])
pred = np.asarray([int(np.around(float(la.split('_')[1]))) for la in df.label])

indoi = np.where(gt != pred)[0]

image_pths = np.array(df.image_name)[indoi]

for path in image_pths:
    shutil.copy(path, DEST)