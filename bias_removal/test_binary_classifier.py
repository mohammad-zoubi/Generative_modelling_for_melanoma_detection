from joblib import load
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

svm = load('/ISIC256/synth60k/svm_30k_imgs.joblib')

test_w_vectors_path = '/ISIC256/synth60k/w_latent_code/w_code2/'
w_matrix = np.zeros((20000, 512))
df_gt = pd.read_csv('/ISIC256/synth60k/synth60k_anno2.csv', header=None)


for idx in tqdm(range(20000)):
    # for idx in range(10):
    # image name without . 
    # image_name = csv_file[0][idx].split('.')[0].split('_')[0]
    image_name = df_gt[0][idx].split('.')[0][:10]
    w_vector = np.transpose( np.loadtxt(test_w_vectors_path + image_name + '.w.1.txt')[:, np.newaxis] ) 
    w_matrix[idx,:] = w_vector

frames_gt = np.asarray(df_gt.loc[:, 4])
frames_pred = svm.predict(w_matrix)

