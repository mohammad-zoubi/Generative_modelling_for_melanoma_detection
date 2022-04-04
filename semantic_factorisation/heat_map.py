import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
matrix_path = "/data/Master_Thesis/semantic_factorization/SeFa_matrices/mal_uncond_eig_vec.pt"
w_matrix = np.load("/data/w_matrix.npy")

eig_vec_matrix = torch.load(matrix_path)['eigvec']
# ma = sns.color_palette("Spectral", as_cmap=True)
fig = plt.figure(figsize=(12,12))
fig = sns.heatmap(eig_vec_matrix[:30,:30], vmin=-0.1, vmax=1.2)
fig = fig.get_figure() 
fig.savefig("/data/Master_Thesis/semantic_factorization/heatmap_eig.jpg") 

print(np.trace(np.matmul(eig_vec_matrix, eig_vec_matrix.T)))
mul = np.matmul(eig_vec_matrix, eig_vec_matrix.T)
np.fill_diagonal(mul.numpy(),0)
if (np.abs(mul) > 1e-5).any(): 
    raise Exception("not orthogonal")

np.savetxt("foo.tsv", w_matrix[:512,:], delimiter="\t")