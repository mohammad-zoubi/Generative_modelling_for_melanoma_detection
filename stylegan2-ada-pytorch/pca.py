''' WHat are the principal directions of the generated images '''

import numpy as np
import seaborn as sns
import os
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from latent_vector_manipulation import latent_manipulator
from tqdm import tqdm
from make_video import make_video

''' Input: many points in text format, make them as a matrix in the shape of [n_samples, n_features]
    Output: visualisations of how the data varys, principal direction eigenvectors '''

# Some paths
# file_path = "/data/generated_imgs_SAM_inv/latent_code"
# file_path = "/data/generated_uncond/generated_imgs_ISIC_ben/latent_code"
file_path = "/data/generated_uncond/generated_imgs_ISIC_mal/latent_code"

# Import the w-vectors and place them in a matrix

# list of files in the directory
vector_files = [os.path.join(file_path,vec)
               for vec in os.listdir(file_path)
               if (vec.endswith(".txt"))]

# w_matrix = np.zeros((len(vector_files), 512))
# for idx in tqdm(range(len(vector_files))):
#     w_vector = np.transpose( np.loadtxt(vector_files[idx])[:, np.newaxis] ) 
#     w_matrix[idx,:] = w_vector

# np.save('/data/stylegan2-ada-pytorch/w_matrix.npy', w_matrix)
w_matrix = np.load('/data/stylegan2-ada-pytorch/w_matrix.npy')
pca = PCA()

pca_matrix = pca.fit(w_matrix).get_covariance()
pca_eigenvalues = pca.fit(w_matrix).explained_variance_
pca_eigenvectors = pca.fit(w_matrix).components_
pca_data = pca.fit_transform(w_matrix)

print(pca_eigenvalues.shape)
print(pca_eigenvectors.shape)
print(pca_data.shape)
number_of_plots = 4
fig = plt.figure(figsize=(15,15))

# for i in tqdm(range(number_of_plots)):
#     plt.subplot(2,2,i+1)
#     # plt.scatter(pca_data[:,2*i], pca_data[:,2*i+1], alpha=0.4, s=10)
#     plt.scatter(pca_data[:,2*i], pca_data[:,2*i+1], alpha=0.4, s=10)
#     plt.xlabel("principal component"+str(2*i+1))
#     plt.ylabel("principal component"+str(2*i+2))
plt.scatter(pca_data[:,0], pca_data[:,3], alpha=0.4, s=10)
plt.savefig('pca_4_plts_mal_ISIC.png')

fig = plt.figure(figsize=(12,5))
components = min(w_matrix.shape)
components = np.arange(0, components)
# print(components)
plt.semilogy(components, pca_eigenvalues[:512], '-o', linewidth=0, markersize=2)
# plt.plot(np.ones((len(components))),components)
plt.xlabel("PCA components")
plt.ylabel("Eigenvalues")
plt.title("Scree plot")
plt.savefig('eigenvals_mal_ISIC.png')

print("Number of components with eigenvalue > 1:", len(pca_eigenvalues[pca_eigenvalues>=1]))
fig = plt.figure(figsize=(12,12))
fig = sns.heatmap(pca_matrix[:30,:30])
fig = fig.get_figure()
fig.savefig("cov_mal_ISIC.png") 

latent_manipulator(model_path='/ISIC256/models/00001-ISIC256_malignant-auto2-bgcfnc-resumecustom/network-snapshot-007200.pkl',
                     outdir='/data/generated_uncond/generated_imgs_ISIC_mal/directed_imgs/', 
                     w_vector_path='/data/generated_uncond/generated_imgs_ISIC_mal/latent_code/seed0002.class.None.txt', 
                    #  w_vector_path='/data/generated_uncond/generated_imgs_ISIC_mal/latent_code/seed0046.class.None.txt', 
                     classifier_path='/data/generated_uncond/generated_imgs_ISIC_mal/binary_classifiers/svm_10k_imgs_ISIC_MAL_200KIMG.joblib', 
                     threshold=False, 
                     steps=1000,
                     alpha=-0.1, 
                     diff_direction=False, 
                     eigenvector=pca_eigenvectors[23,:])

image_folder_path = '/data/generated_uncond/generated_imgs_ISIC_mal/directed_imgs/'
# image_folder_path = '/data/generated_uncond/generated_imgs_ISIC_mal/tmp_img/'
fps = 4
outdir = '/data/stylegan2-ada-pytorch/directed_imgsf23_no_frames.mp4'
make_video(image_folder_path, fps, outdir)